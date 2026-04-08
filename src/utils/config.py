"""YAML-based configuration loader with CLI override support."""

import argparse

import yaml


def _deep_update(base: dict, override: dict) -> dict:
    """Recursively update base dict with override dict."""
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _set_nested(cfg: dict, dotted_key: str, value: str) -> None:
    """Set a value in a nested dict using dot-separated key.
    
    Attempts to cast the string value to int, float, or bool automatically.
    """
    keys = dotted_key.split(".")
    d = cfg
    for k in keys[:-1]:
        d = d.setdefault(k, {})

    # Auto-cast value
    casted: str | int | float | bool = value
    if value.lower() in ("true", "false"):
        casted = value.lower() == "true"
    else:
        try:
            casted = int(value)
        except ValueError:
            try:
                casted = float(value)
            except ValueError:
                pass  # keep as string
    d[keys[-1]] = casted


def load_config(path: str) -> dict:
    """Load a YAML configuration file."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg if cfg is not None else {}


def merge_configs(base_path: str, overrides: list[str] | None = None) -> dict:
    """Load config from YAML and apply CLI key=value overrides.
    
    Args:
        base_path: Path to the YAML config file.
        overrides: List of 'key=value' or 'dotted.key=value' strings.
    
    Returns:
        Merged configuration dictionary.
    """
    cfg = load_config(base_path)

    if overrides:
        for item in overrides:
            if "=" not in item:
                raise ValueError(f"Override must be in key=value format, got: {item}")
            key, value = item.split("=", 1)
            _set_nested(cfg, key.strip(), value.strip())

    return cfg


def parse_args(description: str = "Training/Evaluation") -> argparse.Namespace:
    """Parse command-line arguments for config path and overrides."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--override", nargs="*", default=[],
        help="Override config values as key=value pairs (supports dotted keys).",
    )
    return parser.parse_args()


def get_config(description: str = "Training/Evaluation") -> dict:
    """Parse CLI args and return the merged configuration dict."""
    args = parse_args(description)
    cfg = merge_configs(args.config, args.override)
    return cfg
