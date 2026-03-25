"""
utils.py
========
Shared utilities: configuration loading, logging setup, file I/O helpers,
and common validation functions used across the entire pipeline.
"""

import logging
import os
import pickle
import sys
from pathlib import Path
from typing import Any

import colorlog
import joblib
import yaml

# ── Project root (two levels up from this file) ──────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# =============================================================================
# Configuration
# =============================================================================

def load_config(config_path: str | Path | None = None) -> dict:
    """
    Load the project YAML configuration file.

    Parameters
    ----------
    config_path : str or Path, optional
        Explicit path to config.yaml.  If None, looks for
        ``<PROJECT_ROOT>/config.yaml``.

    Returns
    -------
    dict
        Parsed configuration dictionary.
    """
    if config_path is None:
        config_path = PROJECT_ROOT / "config.yaml"

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as fh:
        cfg = yaml.safe_load(fh)

    return cfg


# =============================================================================
# Logging
# =============================================================================

def get_logger(name: str, config: dict | None = None) -> logging.Logger:
    """
    Return a colourised, file-backed logger.

    Parameters
    ----------
    name    : str  – logger name (usually ``__name__``)
    config  : dict – full project config; uses ``logging`` section if present

    Returns
    -------
    logging.Logger
    """
    cfg = (config or {}).get("logging", {})
    level_name = cfg.get("level", "INFO")
    log_dir = PROJECT_ROOT / cfg.get("log_dir", "logs")
    log_file = cfg.get("log_file", "pipeline.log")
    fmt = cfg.get(
        "format",
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    if logger.handlers:          # avoid duplicate handlers on re-import
        return logger

    level = getattr(logging, level_name.upper(), logging.INFO)
    logger.setLevel(level)

    # ── Console handler (colourised) ──────────────────────────────────────
    colour_fmt = (
        "%(log_color)s%(asctime)s%(reset)s | "
        "%(log_color)s%(levelname)-8s%(reset)s | "
        "%(cyan)s%(name)s%(reset)s | %(message)s"
    )
    console_handler = colorlog.StreamHandler(sys.stdout)
    console_handler.setFormatter(colorlog.ColoredFormatter(colour_fmt))
    console_handler.setLevel(level)
    logger.addHandler(console_handler)

    # ── File handler ──────────────────────────────────────────────────────
    file_handler = logging.FileHandler(log_dir / log_file)
    file_handler.setFormatter(logging.Formatter(fmt))
    file_handler.setLevel(level)
    logger.addHandler(file_handler)

    return logger


# =============================================================================
# Model / Artefact I/O
# =============================================================================

def save_artifact(obj: Any, filename: str, config: dict | None = None) -> Path:
    """
    Persist a Python object (model, scaler, etc.) with joblib.

    Parameters
    ----------
    obj      : Any  – the object to save
    filename : str  – basename, e.g. ``kmeans_model.pkl``
    config   : dict – project config (uses ``model.save_dir``)

    Returns
    -------
    Path – the saved file path
    """
    cfg = (config or {}).get("model", {})
    save_dir = PROJECT_ROOT / cfg.get("save_dir", "models")
    save_dir.mkdir(parents=True, exist_ok=True)

    out_path = save_dir / filename
    joblib.dump(obj, out_path)
    return out_path


def load_artifact(filename: str, config: dict | None = None) -> Any:
    """
    Load a persisted artefact from the models directory.

    Parameters
    ----------
    filename : str  – basename, e.g. ``kmeans_model.pkl``
    config   : dict – project config

    Returns
    -------
    Any – the loaded object
    """
    cfg = (config or {}).get("model", {})
    save_dir = PROJECT_ROOT / cfg.get("save_dir", "models")
    path = save_dir / filename

    if not path.exists():
        raise FileNotFoundError(f"Artifact not found: {path}")

    return joblib.load(path)


# =============================================================================
# Path Helpers
# =============================================================================

def resolve_path(relative: str) -> Path:
    """Return an absolute Path relative to PROJECT_ROOT."""
    return PROJECT_ROOT / relative


def ensure_dir(path: str | Path) -> Path:
    """Create directory (and parents) if it doesn't exist; return Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# =============================================================================
# Validation
# =============================================================================

REQUIRED_RAW_COLUMNS = {
    "InvoiceNo", "StockCode", "Description",
    "Quantity", "InvoiceDate", "UnitPrice", "CustomerID", "Country",
}


def validate_raw_dataframe(df, logger=None) -> bool:
    """
    Check that the raw DataFrame contains the expected columns.

    Parameters
    ----------
    df     : pd.DataFrame
    logger : logging.Logger, optional

    Returns
    -------
    bool – True if validation passes
    """
    missing = REQUIRED_RAW_COLUMNS - set(df.columns)
    if missing:
        msg = f"Raw data missing columns: {missing}"
        if logger:
            logger.error(msg)
        raise ValueError(msg)
    if logger:
        logger.info("Raw DataFrame validation passed ✓")
    return True


REQUIRED_RFM_COLUMNS = {"CustomerID", "Recency", "Frequency", "Monetary"}


def validate_rfm_dataframe(df, logger=None) -> bool:
    """Ensure the RFM feature DataFrame has minimum required columns."""
    missing = REQUIRED_RFM_COLUMNS - set(df.columns)
    if missing:
        msg = f"RFM DataFrame missing columns: {missing}"
        if logger:
            logger.error(msg)
        raise ValueError(msg)
    if logger:
        logger.info("RFM DataFrame validation passed ✓")
    return True


# =============================================================================
# Misc
# =============================================================================

def pretty_print_dict(d: dict, title: str = "") -> None:
    """Print a nested dict in a readable format."""
    if title:
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")
    for k, v in d.items():
        print(f"  {k:30s}: {v}")
    print()
