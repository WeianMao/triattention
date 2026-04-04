"""YAML configuration loader for TriAttention utility scripts."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import copy
import os

try:
    import yaml  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "PyYAML is required to load TriAttention configs. Install with 'pip install pyyaml'."
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class ConfigError(RuntimeError):
    """Raised when the configuration file is invalid."""


def _deep_merge(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in overrides.items():
        if (
            isinstance(value, dict)
            and key in base
            and isinstance(base[key], dict)
        ):
            base[key] = _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def _resolve_config_path(candidate: Optional[str]) -> Optional[Path]:
    path = candidate or os.environ.get("TRIATTENTION_CONFIG_PATH")
    if not path:
        return None
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = (PROJECT_ROOT / resolved).resolve()
    return resolved


def load_config(
    config_path: Optional[str] = None,
    *,
    defaults: Optional[Dict[str, Any]] = None,
    section: Optional[str] = None,
) -> Tuple[Dict[str, Any], Optional[Path]]:
    """Load YAML configuration and merge with defaults.

    Args:
        config_path: Optional explicit YAML path. If omitted, TRIATTENTION_CONFIG_PATH is used.
        defaults: Optional default dictionary merged before user config.
        section: Optional section name (e.g. "online").

    Returns:
        (merged_config, resolved_config_path)
    """
    config: Dict[str, Any] = copy.deepcopy(defaults) if defaults else {}
    resolved = _resolve_config_path(config_path)

    if not resolved:
        return config, None
    if not resolved.is_file():
        raise FileNotFoundError(f"Config file not found: {resolved}")

    with resolved.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    if not isinstance(data, dict):
        raise ConfigError(
            f"Expected a mapping at root of config file {resolved}, got {type(data).__name__}."
        )

    selected: Dict[str, Any]
    if section:
        common = data.get("common", {})
        if common and not isinstance(common, dict):
            raise ConfigError(
                f"Section 'common' must be a mapping in config file {resolved}."
            )
        if common:
            config = _deep_merge(config, copy.deepcopy(common))

        selected = data.get(section, {})
        if selected and not isinstance(selected, dict):
            raise ConfigError(
                f"Section '{section}' must be a mapping in config file {resolved}."
            )
    else:
        selected = data

    if selected:
        config = _deep_merge(config, copy.deepcopy(selected))

    return config, resolved


__all__ = ["load_config", "ConfigError", "PROJECT_ROOT"]
