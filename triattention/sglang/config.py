"""sglang-specific configuration.

Reads environment variables and constructs the runtime configuration
objects used by all TriAttention hooks.  Algorithm-level configuration
(budget, divide_length, scoring params, ...) is handled by the shared
``triattention.vllm.runtime.config.TriAttentionRuntimeConfig``; this
module adds sglang-only knobs on top.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from triattention.vllm.runtime.config import TriAttentionRuntimeConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Environment variable name constants — sglang-only
# ---------------------------------------------------------------------------

ENV_ENABLE_TRIATTENTION: str = "ENABLE_TRIATTENTION"
"""Master switch.  When set to a falsy value, the integration is fully
disabled and ``install_sglang_integration()`` becomes a no-op."""

ENV_DISABLE_RADIX_CACHE_CHECK: str = (
    "TRIATTN_SGLANG_DISABLE_RADIX_CACHE_CHECK"
)
"""When set to a truthy value, skip the startup check that verifies
radix cache is disabled.  Use only when you know what you are doing."""

ENV_QUIET: str = "TRIATTENTION_QUIET"
"""When truthy, suppress integration startup banner and diagnostics."""


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_ENABLE_TRIATTENTION: bool = True
_DEFAULT_DISABLE_RADIX_CACHE_CHECK: bool = False
_DEFAULT_QUIET: bool = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_bool(raw: str) -> bool:
    """Parse a boolean from an environment variable string."""
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {raw!r}")


def _env_bool(name: str, default: bool) -> bool:
    """Read a boolean environment variable with a default."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return _parse_bool(raw)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class SglangIntegrationConfig:
    """Combines the shared ``TriAttentionRuntimeConfig`` with sglang-only
    flags.

    The shared config covers all algorithm parameters (budget, scoring,
    pruning mode, etc.).  This class adds:

    * ``enable_triattention`` — master on/off switch.
    * ``disable_radix_cache_check`` — bypass the radix-cache safety check.
    * ``quiet`` — suppress startup diagnostics.

    Use :meth:`from_env` to construct from environment variables.
    """

    def __init__(
        self,
        runtime_config: TriAttentionRuntimeConfig,
        *,
        enable_triattention: bool = _DEFAULT_ENABLE_TRIATTENTION,
        disable_radix_cache_check: bool = _DEFAULT_DISABLE_RADIX_CACHE_CHECK,
        quiet: bool = _DEFAULT_QUIET,
    ) -> None:
        self.runtime_config = runtime_config
        self.enable_triattention = enable_triattention
        self.disable_radix_cache_check = disable_radix_cache_check
        self.quiet = quiet

    @classmethod
    def from_env(cls) -> "SglangIntegrationConfig":
        """Build from environment variables.

        Algorithm parameters are delegated to
        ``TriAttentionRuntimeConfig.from_env()`` which reads all
        ``TRIATTN_RUNTIME_*`` variables.  sglang-only knobs are read
        here.

        Returns:
            A validated :class:`SglangIntegrationConfig`.

        Raises:
            ValueError: If any parameter fails validation.
        """
        # --- sglang-specific flags ---
        enable_triattention = _env_bool(
            ENV_ENABLE_TRIATTENTION, _DEFAULT_ENABLE_TRIATTENTION
        )
        disable_radix_cache_check = _env_bool(
            ENV_DISABLE_RADIX_CACHE_CHECK, _DEFAULT_DISABLE_RADIX_CACHE_CHECK
        )
        quiet = _env_bool(ENV_QUIET, _DEFAULT_QUIET)

        # --- shared algorithm config (reads TRIATTN_RUNTIME_* vars) ---
        # Force production-safe defaults for sglang: compaction and block
        # reclaim must be on.
        os.environ.setdefault(
            "TRIATTN_RUNTIME_ENABLE_EXPERIMENTAL_KV_COMPACTION", "1"
        )
        os.environ.setdefault(
            "TRIATTN_RUNTIME_ENABLE_EXPERIMENTAL_BLOCK_RECLAIM", "1"
        )
        # Default to the PyTorch scoring path, which is mathematically
        # equivalent to the Triton kernel and validated against the HF
        # reference.  Performance impact is minimal for the scoring-only path.
        os.environ.setdefault("TRIATTN_RUNTIME_REQUIRE_TRITON_SCORING", "0")

        runtime_config = TriAttentionRuntimeConfig.from_env()

        # Force window_size=0 to match the HF per_head_pruning behavior:
        # no window protection; all tokens compete fairly for the budget.
        # The sglang runtime inherits window_size=128 from vLLM defaults,
        # which silently reserves 6.25% of the budget for recent tokens.
        runtime_config.window_size = 0

        # Ensure the PyTorch scoring path is used regardless of env
        # overrides.  This is the authoritative override — the setdefault
        # above handles the normal case; this catches explicit user
        # TRIATTN_RUNTIME_REQUIRE_TRITON_SCORING=1 overrides.
        runtime_config.require_triton_scoring = False

        config = cls(
            runtime_config=runtime_config,
            enable_triattention=enable_triattention,
            disable_radix_cache_check=disable_radix_cache_check,
            quiet=quiet,
        )
        config.validate()
        return config

    def validate(self) -> None:
        """Check sglang-specific parameter constraints.

        The shared ``TriAttentionRuntimeConfig.validate()`` is already
        called during ``from_env()``.  This method only checks
        sglang-specific invariants.

        Raises:
            ValueError: If any constraint is violated.
            FileNotFoundError: If the stats file path is set but missing.
        """
        rc = self.runtime_config

        # If the integration is disabled, skip further checks — nothing
        # will run.
        if not self.enable_triattention:
            return

        # --- stats file existence ---
        if rc.sparse_stats_path is not None:
            p = Path(rc.sparse_stats_path)
            if not p.exists():
                raise FileNotFoundError(
                    f"Stats file not found: {p}  "
                    f"(set via TRIATTN_RUNTIME_SPARSE_STATS_PATH)"
                )

        # --- compaction requires stats ---
        if rc.enable_experimental_kv_compaction and rc.sparse_stats_path is None:
            raise ValueError(
                "KV compaction is enabled but no stats file is configured.  "
                "Set TRIATTN_RUNTIME_SPARSE_STATS_PATH to a valid .pt file."
            )

        # --- budget vs window_size sanity ---
        if rc.window_size >= rc.kv_budget:
            raise ValueError(
                f"window_size ({rc.window_size}) must be < kv_budget "
                f"({rc.kv_budget}); otherwise no tokens are eligible for "
                f"eviction."
            )

        # --- INV-20: Triton dispatch incompatibility warning ---
        # The Triton scoring kernel does not support disable_mlr=True.
        # When disable_mlr is set, scoring falls back to the PyTorch
        # path.  If require_triton_scoring is also True,
        # the user likely expects Triton acceleration but won't get it.
        if rc.disable_mlr and rc.require_triton_scoring:
            logger.warning(
                "disable_mlr=True is incompatible with Triton scoring — "
                "the Triton kernel does not support the simplified MLR "
                "formula and will fall back to the PyTorch path.  "
                "Set TRIATTN_RUNTIME_REQUIRE_TRITON_SCORING=0 to "
                "suppress this warning, or set "
                "TRIATTN_RUNTIME_DISABLE_MLR=0 to use Triton scoring."
            )

    def log_summary(self) -> None:
        """Emit a one-time startup summary at INFO level."""
        if self.quiet:
            return

        rc = self.runtime_config
        logger.info(
            "TriAttention sglang integration config:\n"
            "  enabled            = %s\n"
            "  kv_budget          = %d\n"
            "  divide_length      = %d\n"
            "  window_size        = %d\n"
            "  protect_prefill    = %s\n"
            "  pruning_mode       = %s\n"
            "  stats_path         = %s\n"
            "  model_path         = %s\n"
            "  radix_cache_check  = %s\n"
            "  triton_scoring     = %s",
            self.enable_triattention,
            rc.kv_budget,
            rc.divide_length,
            rc.window_size,
            rc.protect_prefill,
            rc.pruning_mode,
            rc.sparse_stats_path,
            rc.model_path,
            "skipped" if self.disable_radix_cache_check else "enabled",
            "required" if rc.require_triton_scoring else "optional",
        )


def load_sglang_config() -> SglangIntegrationConfig:
    """Build the sglang-specific config from environment variables.

    This is the preferred entry point for other modules.  It delegates
    to :meth:`SglangIntegrationConfig.from_env`.

    Returns:
        A fully-populated and validated :class:`SglangIntegrationConfig`.
    """
    return SglangIntegrationConfig.from_env()
