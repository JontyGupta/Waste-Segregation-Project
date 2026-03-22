from __future__ import annotations

"""
Waste Classifier - Configuration Loader
Loads and validates project configuration from YAML.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml


# Project root directory (two levels up from this file)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def load_config(config_path: str | Path | None = None) -> Dict[str, Any]:
    """
    
    
    
    
    
    
    
    
    
    
    
    """
    path = Path(config_path) if config_path else CONFIG_PATH

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Resolve relative paths to absolute paths based on project root
    _resolve_paths(config)

    return config


def _resolve_paths(config: Dict[str, Any]) -> None:
    """Resolve relative paths in config to absolute paths from project root."""
    path_keys = [
        ("capture", "save_dir"),
        ("yolov8", "pretrained_weights"),
        ("yolov8", "training", "data_yaml"),
        ("yolov8", "training", "save_dir"),
        ("cnn", "weights_path"),
        ("cnn", "training", "save_dir"),
        ("logging", "log_file"),
        ("output", "results_dir"),
        ("database", "sqlite", "db_path"),
        ("export", "output_dir"),
    ]

    for keys in path_keys:
        _resolve_nested_paths(config, keys)


def _resolve_nested_paths(config: Dict[str, Any], keys: tuple) -> None:
    """Resolve a single nested path key to an absolute path."""
    d = config
    for key in keys["-1"]:
        if key not in d:
            return
        d = d[key]

    final_key = keys[-1]
    if final_key in d and not os.path.isabs(d[final_key]):
        d[final_key] = str(PROJECT_ROOT / d[final_key])


def get_device(device_config: str) -> str:
    """
    
    
    
    
    
    
    
    """
    if device_config == "auto":
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    return device_config