"""
Same API as HMMLSTM/cryptoregimeclassifier/src/split_bounds.py — discovers repo-root split_config.json
by walking parents from this file (RLModel/).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

SPLIT_CONFIG_NAME = "split_config.json"


def discover_split_config_path(start: Path | None = None) -> Path | None:
    root = Path(start).resolve() if start is not None else Path(__file__).resolve().parent
    env_p = os.environ.get("SPLIT_CONFIG_PATH")
    if env_p:
        p = Path(env_p).expanduser().resolve()
        return p if p.is_file() else None
    for d in [root, *root.parents]:
        cand = d / SPLIT_CONFIG_NAME
        if cand.is_file():
            return cand
    return None


def load_split_config(path: Path | str | None = None) -> dict[str, Any]:
    if path is not None:
        p = Path(path).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"split config not found: {p}")
    else:
        p = discover_split_config_path(Path(__file__).parent)
        if p is None:
            raise FileNotFoundError(
                f"{SPLIT_CONFIG_NAME} not found in parents of RLModel/; set SPLIT_CONFIG_PATH"
            )
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def try_load_split_config(path: Path | str | None = None) -> dict[str, Any] | None:
    try:
        return load_split_config(path) if path is not None else load_split_config()
    except FileNotFoundError:
        return None


def parse_split_bounds(raw: dict[str, Any]) -> dict[str, pd.Timestamp]:
    keys = ("hmm_fit_end", "lstm_train_end", "rl_train_end", "test_start")
    out: dict[str, pd.Timestamp] = {}
    for k in keys:
        if k not in raw or raw[k] is None:
            raise KeyError(f"split_config.json missing required key: {k}")
        ts = pd.Timestamp(raw[k])
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        out[k] = ts
    return out


def _ts_series(df: pd.DataFrame, ts_col: str) -> pd.Series:
    return pd.to_datetime(df[ts_col], utc=True, errors="coerce")


def lstm_train_mask(df: pd.DataFrame, bounds: dict[str, pd.Timestamp], ts_col: str = "timestamp") -> pd.Series:
    ts = _ts_series(df, ts_col)
    return ts <= bounds["lstm_train_end"]


def lstm_test_mask(df: pd.DataFrame, bounds: dict[str, pd.Timestamp], ts_col: str = "timestamp") -> pd.Series:
    ts = _ts_series(df, ts_col)
    return ts >= bounds["test_start"]


def lstm_oos_eval_export_mask(df: pd.DataFrame, bounds: dict[str, pd.Timestamp], ts_col: str = "timestamp") -> pd.Series:
    """Rows with ts > lstm_train_end (OOS for LSTM metrics / export aligned with RL)."""
    ts = _ts_series(df, ts_col)
    return ts.notna() & (ts > bounds["lstm_train_end"])


def effective_rl_test_start(bounds: dict[str, pd.Timestamp]) -> pd.Timestamp:
    """
    First timestamp at which RL backtest / holdout is allowed.
    Ensures test does not begin before ``rl_train_end`` even if ``test_start`` in JSON is earlier.
    """
    return max(bounds["test_start"], bounds["rl_train_end"])


def rl_train_mask(df: pd.DataFrame, bounds: dict[str, pd.Timestamp], ts_col: str = "timestamp") -> pd.Series:
    ts = _ts_series(df, ts_col)
    t_test = effective_rl_test_start(bounds)
    return (
        (ts > bounds["lstm_train_end"])
        & (ts <= bounds["rl_train_end"])
        & (ts < t_test)
    )


def rl_test_mask(df: pd.DataFrame, bounds: dict[str, pd.Timestamp], ts_col: str = "timestamp") -> pd.Series:
    return _ts_series(df, ts_col) >= effective_rl_test_start(bounds)
