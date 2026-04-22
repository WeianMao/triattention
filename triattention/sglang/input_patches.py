"""Input tensor patches — seq_lens / positions decoupling.

sglang couples ``positions`` and ``seq_lens`` during decode:
``positions = clamp_position(seq_lens) = seq_lens - 1``.  TriAttention
needs them decoupled: ``seq_lens`` must reflect *effective* length
(so attention sees only retained KV), while ``positions`` must stay at
the *absolute* (logical) position (so RoPE encoding remains correct).

The patch has two injection points that cooperate via a module-level
staging variable (safe because sglang runs scheduler + worker in the
same process with a synchronous event loop):

1. ``prepare_for_decode`` patch — runs first.  Saves the logical
   seq_lens (= absolute positions for the upcoming token) and
   overwrites ``batch.seq_lens`` with effective values for compressed
   requests *before* ``alloc_for_decode`` runs.  This ensures the new
   KV slot is allocated at the correct (effective) position in
   ``req_to_token``.

2. ``ForwardBatch.init_new`` patch — runs second.  Reads the staged
   logical positions and overrides the ``positions`` tensor that would
   otherwise be derived from the (now-effective) seq_lens.  This keeps
   RoPE positional encoding at the true absolute position.

Uncompressed requests are never modified — the override tensor only
differs from the default at indices corresponding to compressed requests.
"""

from __future__ import annotations

import functools
import logging
from typing import Callable, Optional

import torch

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# Module-level staging area
# -----------------------------------------------------------------------
# Populated by the prepare_for_decode patch, consumed (and cleared) by
# the ForwardBatch.init_new patch.  The synchronous event loop guarantees
# that init_new always sees the value set by prepare_for_decode from the
# same scheduling round.
#
# Contains the *logical* seq_lens tensor BEFORE prepare_for_decode's +1
# increment.  In decode mode the absolute position for the new token
# equals this value (0-indexed: token at position N means N tokens
# already exist).
_pending_logical_seq_lens: Optional[torch.Tensor] = None


# -----------------------------------------------------------------------
# G.1a — ScheduleBatch.prepare_for_decode patch
# -----------------------------------------------------------------------


def _patched_prepare_for_decode(original_fn: Callable) -> Callable:
    """Return a wrapper around ``ScheduleBatch.prepare_for_decode`` that
    overwrites ``seq_lens`` with effective lengths for compressed requests
    before allocation, and stages logical positions for later use by the
    ``ForwardBatch.init_new`` patch.

    The original ``prepare_for_decode`` does (in order):
      1. ``alloc_for_decode(self)`` — writes new KV slot at
         ``req_to_token[req_pool_idx, seq_lens]``
      2. ``req.kv_committed_len += 1``; ``req.kv_allocated_len += 1``
      3. ``self.seq_lens += 1``

    If a request was compressed (effective < logical), we must override
    ``seq_lens`` to effective *before* step 1 so the new slot lands at
    the correct physical position.  The logical values are saved so that
    ``ForwardBatch.init_new`` can derive the correct absolute positions.
    """
    from triattention.sglang.effective_length import EffectiveLengthTracker

    @functools.wraps(original_fn)
    def wrapped(self):
        global _pending_logical_seq_lens

        # Fast path: no tracker or tracker has no overrides.
        tracker: Optional[EffectiveLengthTracker] = getattr(
            self, "_triattention_tracker", None
        )
        # The tracker is attached to the Scheduler, not to the
        # ScheduleBatch.  We look it up via the batch's reference to the
        # scheduler (not always available) or through a module-level
        # reference set during hook installation.
        if tracker is None:
            tracker = _get_tracker_for_batch(self)

        if tracker is None or not tracker.has_any_overrides():
            # No compressed requests — run original unmodified.
            _pending_logical_seq_lens = None
            return original_fn(self)

        # Build the override: for each request in the batch, check
        # whether the tracker has an effective-length override.
        reqs = getattr(self, "reqs", None)
        if not reqs:
            _pending_logical_seq_lens = None
            return original_fn(self)

        # Save the current (logical) seq_lens BEFORE any modification.
        # This is the absolute position for the next token.
        logical_seq_lens = self.seq_lens.clone()

        # Determine which requests need overriding.
        needs_override = False
        effective_values = []
        for i, req in enumerate(reqs):
            eff = tracker.get_effective_len(req.rid)
            if eff is not None:
                effective_values.append((i, eff))
                needs_override = True

        if not needs_override:
            _pending_logical_seq_lens = None
            return original_fn(self)

        # Stage the logical seq_lens for ForwardBatch.init_new.
        _pending_logical_seq_lens = logical_seq_lens

        # Overwrite seq_lens (GPU tensor) and seq_lens_cpu with
        # effective values for compressed requests.
        # Track the total delta to update seq_lens_sum without GPU sync.
        delta_sum = 0
        for idx, eff_len in effective_values:
            old_val = int(self.seq_lens_cpu[idx]) if (
                hasattr(self, "seq_lens_cpu")
                and self.seq_lens_cpu is not None
            ) else int(self.seq_lens[idx].item())
            delta_sum += eff_len - old_val
            self.seq_lens[idx] = eff_len
            if hasattr(self, "seq_lens_cpu") and self.seq_lens_cpu is not None:
                self.seq_lens_cpu[idx] = eff_len

        # Input patch logging.
        logger.debug(
            "TriAttention prepare_for_decode: overriding seq_lens for %d/%d "
            "requests; logical→effective: %s",
            len(effective_values),
            len(reqs),
            [(int(logical_seq_lens[idx].item()), eff) for idx, eff in effective_values],
        )

        # Update seq_lens_sum by the computed delta (avoids GPU sync).
        if hasattr(self, "seq_lens_sum"):
            self.seq_lens_sum += delta_sum

        # Run the original prepare_for_decode with effective seq_lens.
        # This will:
        #   - alloc_for_decode at effective positions (correct)
        #   - kv_committed_len += 1, kv_allocated_len += 1 (correct,
        #     since they were already synced to effective by compression)
        #   - seq_lens += 1 (now effective + 1, correct for attention)
        return original_fn(self)

    return wrapped


# -----------------------------------------------------------------------
# G.1b — ForwardBatch.init_new patch
# -----------------------------------------------------------------------


def _patched_forward_batch_init_new(original_init_new: Callable) -> Callable:
    """Return a wrapper around ``ForwardBatch.init_new`` that fixes
    ``positions`` for compressed requests.

    After ``prepare_for_decode`` overrides seq_lens to effective values,
    the +1 increment yields ``seq_lens = effective + 1``.  The default
    position calculation ``clamp_position(seq_lens) = seq_lens - 1``
    would produce ``effective`` as the position — but the correct
    absolute position is ``logical`` (the original seq_lens before
    override).

    This patch replaces ``positions`` with ``logical_seq_lens`` (staged
    by the ``prepare_for_decode`` patch) for decode-mode batches that
    have compressed requests.  The staged logical_seq_lens already
    represent the correct 0-indexed absolute position for the new token.
    """

    @functools.wraps(original_init_new)
    def wrapped(cls, batch, model_runner):
        global _pending_logical_seq_lens

        # Call the original init_new (unbound classmethod __func__).
        ret = original_init_new(cls, batch, model_runner)

        # Only intervene for decode mode when we have staged positions.
        if _pending_logical_seq_lens is None:
            return ret

        staged = _pending_logical_seq_lens
        # Consume the staged value (one-shot).  Always clear,
        # even if we end up not using it (e.g. extend mode).
        _pending_logical_seq_lens = None

        # Only override in decode (or target_verify) mode — extend mode
        # computes positions differently and is not affected.
        if not (ret.forward_mode.is_decode() or ret.forward_mode.is_target_verify()):
            return ret

        # If positions were already set by spec_info or dllm, don't
        # override — those paths have their own position logic.
        # In the normal decode path, positions = clamp_position(batch.seq_lens).
        # We replace it with the logical values.
        if ret.positions is None:
            # Should not happen in decode mode, but be safe.
            return ret

        device = ret.positions.device

        # staged is the logical seq_lens BEFORE the +1 in
        # prepare_for_decode.  The absolute position for the new token
        # is exactly this value (0-indexed).
        logical_positions = staged.to(device=device, dtype=torch.int64)

        # Sanity check: dimensions must match.
        if logical_positions.shape[0] != ret.positions.shape[0]:
            logger.warning(
                "TriAttention input_patches: staged logical positions "
                "shape %s != batch positions shape %s — skipping "
                "override.",
                logical_positions.shape,
                ret.positions.shape,
            )
            return ret

        # Input patch logging.
        n_overridden = int((logical_positions != (ret.positions)).sum().item())
        logger.debug(
            "TriAttention init_new: overriding positions for %d requests; "
            "sample effective→logical: %s",
            n_overridden,
            list(zip(
                ret.positions[:8].tolist(),
                logical_positions[:8].tolist(),
            )),
        )

        ret.positions = logical_positions
        return ret

    return wrapped


# -----------------------------------------------------------------------
# Tracker lookup helper
# -----------------------------------------------------------------------

# Module-level reference to the scheduler's tracker, set during
# hook installation (see integration.py).
_active_tracker: Optional["EffectiveLengthTracker"] = None  # noqa: F821


def set_active_tracker(tracker) -> None:
    """Called by integration.py to register the active tracker."""
    global _active_tracker
    _active_tracker = tracker


def _get_tracker_for_batch(batch) -> Optional["EffectiveLengthTracker"]:  # noqa: F821
    """Try to find the EffectiveLengthTracker for this batch.

    Lookup order:
      1. Module-level ``_active_tracker`` (set during hook installation).
      2. ``batch._triattention_tracker`` (if directly attached).
    """
    if _active_tracker is not None:
        return _active_tracker
    return getattr(batch, "_triattention_tracker", None)
