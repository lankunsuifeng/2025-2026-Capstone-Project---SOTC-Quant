import json
import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import joblib
    from sklearn.preprocessing import RobustScaler
except ImportError:  # optional until normalize_features=True
    joblib = None  # type: ignore[misc, assignment]
    RobustScaler = None  # type: ignore[misc, assignment]


def _labels_to_state_ids(series: pd.Series, n_states: int) -> pd.Series:
    """
    Map LSTM/HMM label column to integer class ids in ``0 .. n_states-1``.

    Accepts: integers/floats, numeric strings, ``State_k`` / ``state_k`` (from LSTM export).
    Unparseable values become NaN.
    """
    out = np.full(len(series), np.nan, dtype=np.float64)
    idx = series.index
    for j, v in enumerate(series):
        if pd.isna(v):
            continue
        if isinstance(v, (bool, np.bool_)):
            continue
        if isinstance(v, (np.integer, int)):
            out[j] = int(v)
            continue
        if isinstance(v, (np.floating, float)):
            if np.isfinite(v):
                out[j] = int(round(v))
            continue
        s = str(v).strip()
        if s.isdigit():
            out[j] = int(s)
            continue
        m = re.match(r"(?i)^state_?(\d+)$", s)
        if m:
            out[j] = int(m.group(1))
            continue
        m = re.search(r"_(\d+)\s*$", s)
        if m:
            out[j] = int(m.group(1))
            continue
    ser = pd.Series(out, index=idx)
    ser = ser.where(ser.isna() | ((ser >= 0) & (ser < n_states)), np.nan)
    return ser


def _ensure_predicted_state_columns(
    df: pd.DataFrame,
    hmm_pred_state_col: str,
    lstm_pred_state_col: str,
    n_states: int,
    hmm_label_source_col: str | None,
    lstm_label_source_col: str | None,
) -> pd.DataFrame:
    """
    If canonical columns are missing, build them from HMMLSTM export conventions:

    - ``target`` → next-bar HMM regime class (same semantics as training target on page 4)
    - ``lstm_predicted_regime`` → LSTM predicted regime string or id
    """
    df = df.copy()
    if hmm_label_source_col and hmm_label_source_col in df.columns:
        if hmm_pred_state_col in df.columns:
            df = df.drop(columns=[hmm_label_source_col])
        else:
            df[hmm_pred_state_col] = _labels_to_state_ids(df[hmm_label_source_col], n_states)
            df = df.drop(columns=[hmm_label_source_col])
    if lstm_label_source_col and lstm_label_source_col in df.columns:
        if lstm_pred_state_col in df.columns:
            df = df.drop(columns=[lstm_label_source_col])
        else:
            df[lstm_pred_state_col] = _labels_to_state_ids(df[lstm_label_source_col], n_states)
            df = df.drop(columns=[lstm_label_source_col])
    return df


def _continuous_rl_feature_cols(
    feature_cols: list[str],
    *,
    hmm_pred_state_col: str,
    lstm_pred_state_col: str,
    onehot_pred_states: bool,
    pred_state_source: str,
) -> list[str]:
    """Numeric indicators only; exclude HMM/LSTM one-hot columns (leave 0/1 as-is)."""
    prefixes: list[str] = []
    if onehot_pred_states:
        _src = str(pred_state_source).strip().lower()
        if _src in {"both", "hmm"}:
            prefixes.append(f"{hmm_pred_state_col}_")
        if _src in {"both", "lstm"}:
            prefixes.append(f"{lstm_pred_state_col}_")
    out: list[str] = []
    for c in feature_cols:
        if any(str(c).startswith(p) for p in prefixes):
            continue
        out.append(c)
    return out


def _robust_scaler_fit_mask(
    df: pd.DataFrame,
    *,
    timestamp_col: str,
    scaler_train_ratio: float,
    split_config_path: str | None,
) -> tuple[np.ndarray, str]:
    """
    Boolean mask (length ``len(df)``): rows used to fit RobustScaler.
    Prefer ``split_config`` + ``rl_train_mask``; else first ``scaler_train_ratio`` fraction (row order).
    """
    from split_bounds import parse_split_bounds, rl_train_mask, try_load_split_config

    raw = try_load_split_config(split_config_path) if split_config_path else try_load_split_config()
    if raw is not None and timestamp_col in df.columns:
        try:
            bounds = parse_split_bounds(raw)
            m = rl_train_mask(df, bounds, ts_col=timestamp_col).to_numpy(dtype=bool)
            if bool(m.any()):
                return m, "split_config.rl_train_mask"
        except (KeyError, ValueError, TypeError) as e:
            warnings.warn(
                f"RobustScaler: split_config unusable ({e!r}); using scaler_train_ratio fallback on row order.",
                UserWarning,
                stacklevel=2,
            )

    n = len(df)
    if n == 0:
        return np.zeros(0, dtype=bool), "empty"
    tr = float(scaler_train_ratio)
    tr = min(max(tr, 1e-6), 1.0)
    split_idx = max(1, int(n * tr))
    mask = np.zeros(n, dtype=bool)
    mask[:split_idx] = True
    return mask, f"train_ratio_fallback(ratio={tr})"


def _train_winsor_bounds(
    out: pd.DataFrame,
    continuous_cols: list[str],
    mask_fit: np.ndarray,
    *,
    low_q: float,
    high_q: float,
) -> dict[str, tuple[float, float]]:
    """Per-column (lo, hi) from quantiles on RL train rows only."""
    low_q = float(low_q)
    high_q = float(high_q)
    if not (0.0 <= low_q < high_q <= 1.0):
        raise ValueError("winsor quantiles require 0 <= low_q < high_q <= 1")
    bounds: dict[str, tuple[float, float]] = {}
    for c in continuous_cols:
        if c not in out.columns:
            continue
        v = out.loc[mask_fit, c].to_numpy(dtype=np.float64)
        v = v[np.isfinite(v)]
        if v.size < 2:
            continue
        lo = float(np.quantile(v, low_q))
        hi = float(np.quantile(v, high_q))
        if lo > hi:
            lo, hi = hi, lo
        bounds[c] = (lo, hi)
    return bounds


def _apply_winsor_clip(out: pd.DataFrame, bounds: dict[str, tuple[float, float]]) -> None:
    for c, (lo, hi) in bounds.items():
        if c not in out.columns:
            continue
        arr = out[c].to_numpy(dtype=np.float64, copy=False)
        out[c] = np.clip(arr, lo, hi).astype(np.float32)


def _apply_rl_train_robust_scaler(
    out: pd.DataFrame,
    *,
    continuous_cols: list[str],
    timestamp_col: str,
    scaler_train_ratio: float,
    split_config_path: str | None,
    robust_scaler_path: str,
    fit_mask: np.ndarray | None = None,
    fit_split_mode: str | None = None,
) -> None:
    if joblib is None or RobustScaler is None:
        raise ImportError(
            "normalize_features=True requires scikit-learn and joblib. "
            "Install e.g. `pip install scikit-learn joblib`."
        )
    if not continuous_cols:
        warnings.warn("RobustScaler: no continuous feature columns to scale; skipping.", UserWarning, stacklevel=2)
        return

    if fit_mask is None:
        mask_fit, split_mode = _robust_scaler_fit_mask(
            out,
            timestamp_col=timestamp_col,
            scaler_train_ratio=scaler_train_ratio,
            split_config_path=split_config_path,
        )
    else:
        mask_fit = np.asarray(fit_mask, dtype=bool)
        split_mode = fit_split_mode or "provided"
        if len(mask_fit) != len(out):
            raise ValueError("fit_mask length must match out rows")

    if not bool(mask_fit.any()):
        warnings.warn("RobustScaler: fit mask is empty; skipping normalization.", UserWarning, stacklevel=2)
        return

    X_fit = out.loc[mask_fit, continuous_cols].to_numpy(dtype=np.float64, copy=True)
    scaler = RobustScaler()
    scaler.fit(X_fit)
    X_all = out[continuous_cols].to_numpy(dtype=np.float64, copy=True)
    out.loc[:, continuous_cols] = scaler.transform(X_all).astype(np.float32)

    os.makedirs(os.path.dirname(robust_scaler_path) or ".", exist_ok=True)
    joblib.dump(
        {"scaler": scaler, "columns": list(continuous_cols), "kind": "RobustScaler"},
        robust_scaler_path,
    )
    meta_path = os.path.splitext(robust_scaler_path)[0] + "_meta.json"
    meta = {
        "robust_scaler_path": robust_scaler_path,
        "columns_scaled": list(continuous_cols),
        "fit_mask_rule": split_mode,
        "n_rows_fit": int(mask_fit.sum()),
        "n_rows_total": int(len(out)),
        "timestamp_col": timestamp_col,
    }
    with open(meta_path, "w", encoding="utf-8") as fp:
        json.dump(meta, fp, indent=2, ensure_ascii=False)


def _add_multi_horizon_features(
    df: pd.DataFrame,
    *,
    close_col: str,
    adx_col: str = "adx_5m",
    vol_windows: tuple[int, ...] = (5, 20, 60),
    bars_1h: int = 12,
    bars_4h: int = 48,
) -> list[str]:
    """
    Causal multi-horizon features (in-place). Returns list of column names added.

    - Rolling log-return volatility over ``vol_windows`` bars.
    - Cumulative simple returns over ``bars_1h`` / ``bars_4h`` bars (5m bars → 1h / 4h).
    - ADX first difference as trend-strength slope (if ``adx_col`` present).
    """
    added: list[str] = []
    if close_col not in df.columns:
        return added

    close = pd.to_numeric(df[close_col], errors="coerce").astype(np.float64)
    prev = close.shift(1)
    log_ret = np.log(np.where(prev > 0, close / prev, np.nan))

    for w in vol_windows:
        name = f"roll_vol_{w}_5m"
        df[name] = pd.Series(log_ret, index=df.index).rolling(w, min_periods=w).std()
        added.append(name)

    name_1h = "cum_ret_1h_5m"
    df[name_1h] = close.pct_change(periods=bars_1h)
    added.append(name_1h)

    name_4h = "cum_ret_4h_5m"
    df[name_4h] = close.pct_change(periods=bars_4h)
    added.append(name_4h)

    if adx_col in df.columns:
        adx = pd.to_numeric(df[adx_col], errors="coerce")
        slope_name = "adx_slope_5m"
        df[slope_name] = adx.diff(1)
        added.append(slope_name)

    return added


def data_insight(data: pd.DataFrame, close_col: str = "close_5m") -> dict:
    df = data.copy()

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c != close_col]

    out = {
        "shape": df.shape,
        "n_numeric_cols": len(numeric_cols),
        "n_feature_cols": len(feature_cols),
        "feature_cols": feature_cols,
        "non_numeric_cols": df.columns.difference(numeric_cols).tolist(),
        "nan_ratio_overall": float(df.isna().mean().mean()),
        "nan_ratio_close": float(df[close_col].isna().mean()) if close_col in df.columns else None,
    }

    # simple return stats (optional)
    if close_col in df.columns:
        close = pd.to_numeric(df[close_col], errors="coerce")
        ret = close.pct_change()
        out["ret_mean"] = float(ret.mean(skipna=True))
        out["ret_std"] = float(ret.std(skipna=True))
        out["ret_nan_ratio"] = float(ret.isna().mean())

    # ---- scale analysis for features ----
    if len(feature_cols) > 0:
        X = df[feature_cols].to_numpy(dtype=float)
        finite = np.isfinite(X)

        absmax = np.nanmax(np.abs(np.where(finite, X, np.nan)), axis=0)
        mean = np.nanmean(np.where(finite, X, np.nan), axis=0)
        std = np.nanstd(np.where(finite, X, np.nan), axis=0)
        p01 = np.nanpercentile(np.where(finite, X, np.nan), 1, axis=0)
        p50 = np.nanpercentile(np.where(finite, X, np.nan), 50, axis=0)
        p99 = np.nanpercentile(np.where(finite, X, np.nan), 99, axis=0)

        dyn = np.abs(p99 - p01) / (np.abs(p50) + 1e-12)

        scale_df = pd.DataFrame({
            "feature": feature_cols,
            "absmax": absmax,
            "mean": mean,
            "std": std,
            "p01": p01,
            "p50": p50,
            "p99": p99,
            "dyn_range": dyn,
            "nonfinite_ratio": 1.0 - np.mean(finite, axis=0),
            "nan_ratio": np.mean(np.isnan(X), axis=0),
        }).sort_values("absmax", ascending=False)

        # flags: likely problematic for NN
        flags = scale_df[
            (scale_df["nonfinite_ratio"] > 0) |
            (scale_df["absmax"] > 1e6) |
            (scale_df["std"] > 1e3) |
            (scale_df["dyn_range"] > 1e3)
        ][["feature", "absmax", "std", "dyn_range", "nonfinite_ratio", "nan_ratio"]]

        out["scale_table"] = scale_df  # full table (pandas DataFrame)
        out["scale_flags"] = flags     # suspicious columns

        # quick global summary
        out["scale_summary"] = {
            "absmax_max": float(scale_df["absmax"].max()),
            "std_max": float(scale_df["std"].max()),
            "nonfinite_any": bool((scale_df["nonfinite_ratio"] > 0).any()),
            "n_flagged": int(len(flags)),
        }

    return out
def data_engineering(
    input_csv: str = "data/BTCUSDT_combined_klines_20210201_20260201_hmm_states4_labeled_lstm_nextregime_ts32_predictions.csv",
    output_csv: str = "data/data_e.csv",
    close_col: str = "close_5m",
    state_col: str = "state",
    hmm_pred_state_col: str = "hmm_predicted_state",
    lstm_pred_state_col: str = "lstm_predicted_state",
    pred_state_source: str = "both",  # "both" | "hmm" | "lstm"
    regime_col: str = "regime",  # ignored for output: RL uses HMM/LSTM one-hots only (no duplicate regime column)
    timestamp_col: str = "timestamp",
    n_states: int = 4,
    onehot_pred_states: bool = True,
    write_split_meta: bool = True,
    # HMMLSTM export: ``target`` = HMM next-bar class; ``lstm_predicted_regime`` = LSTM output
    hmm_label_source_col: str | None = "target",
    lstm_label_source_col: str | None = "lstm_predicted_regime",
    # RobustScaler on continuous RL features: fit on RL train window, transform full series (like LSTM scaler).
    normalize_features: bool = False,
    robust_scaler_path: str | None = None,
    scaler_train_ratio: float = 0.8,
    normalization_split_config_path: str | None = None,
    # Winsorize continuous features on RL train quantiles, clip full series (before RobustScaler).
    winsorize_features: bool = False,
    winsor_low_q: float = 0.01,
    winsor_high_q: float = 0.99,
    winsor_meta_path: str | None = None,
    # Multi-horizon RL features: rolling vol (5/20/60), 1h/4h cum return (12/48 bars on 5m), ADX slope.
    multi_horizon_features: bool = False,
) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    # --- drop bulky probability/confidence columns (keep predicted_state only) ---
    drop_prob_cols = []
    for c in df.columns:
        if c == "lstm_prediction_confidence":
            drop_prob_cols.append(c)
        elif c.startswith("lstm_prob_"):
            drop_prob_cols.append(c)
        elif c.startswith("hmm_prob_"):
            drop_prob_cols.append(c)
    if drop_prob_cols:
        df = df.drop(columns=drop_prob_cols)

    # --- align HMMLSTM column names → hmm_predicted_state / lstm_predicted_state ---
    df = _ensure_predicted_state_columns(
        df,
        hmm_pred_state_col=hmm_pred_state_col,
        lstm_pred_state_col=lstm_pred_state_col,
        n_states=n_states,
        hmm_label_source_col=hmm_label_source_col,
        lstm_label_source_col=lstm_label_source_col,
    )

    # --- build relative price features (avoid price level) ---
    # Use existing open/high/low if present; otherwise skip these features.
    if "open_5m" in df.columns:
        df["co_ret_5m"] = (df[close_col] / (df["open_5m"] + 1e-12)) - 1.0
    if "high_5m" in df.columns and "low_5m" in df.columns:
        df["hl_range_norm_5m"] = (df["high_5m"] - df["low_5m"]) / (df[close_col] + 1e-12)
        df["log_hl_5m"] = np.log((df["high_5m"] + 1e-12) / (df["low_5m"] + 1e-12))

    _mh_feat_cols: list[str] = []
    if multi_horizon_features:
        _mh_feat_cols = _add_multi_horizon_features(df, close_col=close_col)

    # Drop raw HMM latent ``state``. RL uses regime **one-hot** columns only; scalars would duplicate *_0..n-1.
    if state_col in df.columns:
        df = df.drop(columns=[state_col])

    # --- include predicted states for RL target/features ---
    _src = str(pred_state_source).strip().lower()
    if _src not in {"both", "hmm", "lstm"}:
        raise ValueError("pred_state_source must be one of: 'both', 'hmm', 'lstm'")

    if _src == "hmm":
        pred_state_cols = [c for c in [hmm_pred_state_col] if c in df.columns]
    elif _src == "lstm":
        pred_state_cols = [c for c in [lstm_pred_state_col] if c in df.columns]
    else:
        pred_state_cols = [c for c in [hmm_pred_state_col, lstm_pred_state_col] if c in df.columns]

    if onehot_pred_states and pred_state_cols:
        cats = list(range(n_states))
        for c in pred_state_cols:
            oh = pd.get_dummies(pd.Categorical(df[c], categories=cats), prefix=c)
            df = pd.concat([df, oh], axis=1)

    # --- columns to keep (relative + existing indicators) ---
    preferred_feats = [
        # your existing engineered indicators (keep if exist)
        "log_ret_1_5m",
        "ema_ratio_9_21_5m",
        "macd_hist_5m",
        "adx_5m",
        "atr_norm_5m",
        "bb_width_5m",
        "rsi_14_5m",
        "volume_zscore_50_5m",
        # newly created relative features
        "co_ret_5m",
        "hl_range_norm_5m",
        "log_hl_5m",
    ]
    if multi_horizon_features:
        preferred_feats += [c for c in _mh_feat_cols if c in df.columns]
    if onehot_pred_states:
        if _src in {"both", "hmm"}:
            preferred_feats += [c for c in df.columns if c.startswith(f"{hmm_pred_state_col}_")]
        if _src in {"both", "lstm"}:
            preferred_feats += [c for c in df.columns if c.startswith(f"{lstm_pred_state_col}_")]

    # keep only those that exist
    feature_cols = [c for c in preferred_feats if c in df.columns]

    # --- drop raw level columns (方案B核心) ---
    drop_level_cols = [c for c in ["open_5m", "high_5m", "low_5m", "volume_5m"] if c in df.columns]
    if drop_level_cols:
        df = df.drop(columns=drop_level_cols)

    # --- final column order ---
    keep = []
    for c in [timestamp_col, close_col]:
        if c in df.columns:
            keep.append(c)
    # Omit scalar predicted_state if full one-hots exist (RL / MoE use *_0..n-1 only).
    for c in pred_state_cols:
        if onehot_pred_states and all(f"{c}_{k}" in df.columns for k in range(n_states)):
            continue
        keep.append(c)
    keep += feature_cols

    out = df[keep].copy()

    # ensure numeric features are float32 where possible
    _num_cols = list(feature_cols) + [close_col]
    for c in pred_state_cols:
        if c in out.columns:
            _num_cols.append(c)
    for c in _num_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out.dropna().reset_index(drop=True)

    continuous_for_scale = _continuous_rl_feature_cols(
        feature_cols,
        hmm_pred_state_col=hmm_pred_state_col,
        lstm_pred_state_col=lstm_pred_state_col,
        onehot_pred_states=onehot_pred_states,
        pred_state_source=_src,
    )

    need_fit_mask = (winsorize_features or normalize_features) and bool(continuous_for_scale)
    fit_mask: np.ndarray | None = None
    fit_split_mode = ""
    if need_fit_mask:
        fit_mask, fit_split_mode = _robust_scaler_fit_mask(
            out,
            timestamp_col=timestamp_col,
            scaler_train_ratio=scaler_train_ratio,
            split_config_path=normalization_split_config_path,
        )

    if winsorize_features:
        if not continuous_for_scale:
            warnings.warn("Winsorize: no continuous feature columns; skipping.", UserWarning, stacklevel=2)
        elif fit_mask is None or not bool(fit_mask.any()):
            warnings.warn("Winsorize: empty RL train mask; skipping.", UserWarning, stacklevel=2)
        else:
            wb = _train_winsor_bounds(
                out,
                continuous_for_scale,
                fit_mask,
                low_q=winsor_low_q,
                high_q=winsor_high_q,
            )
            if not wb:
                warnings.warn("Winsorize: no per-column bounds computed; skipping.", UserWarning, stacklevel=2)
            else:
                _apply_winsor_clip(out, wb)
                wmeta_path = winsor_meta_path or (os.path.splitext(output_csv)[0] + "_winsor_meta.json")
                os.makedirs(os.path.dirname(wmeta_path) or ".", exist_ok=True)
                serial = {
                    "fit_mask_rule": fit_split_mode,
                    "n_rows_fit": int(fit_mask.sum()),
                    "n_rows_total": int(len(out)),
                    "winsor_low_q": float(winsor_low_q),
                    "winsor_high_q": float(winsor_high_q),
                    "columns": {
                        c: {"clip_low": lo, "clip_high": hi} for c, (lo, hi) in sorted(wb.items())
                    },
                }
                with open(wmeta_path, "w", encoding="utf-8") as fp:
                    json.dump(serial, fp, indent=2, ensure_ascii=False)

    if normalize_features:
        rsp = robust_scaler_path or (os.path.splitext(output_csv)[0] + "_robust_scaler.joblib")
        _apply_rl_train_robust_scaler(
            out,
            continuous_cols=continuous_for_scale,
            timestamp_col=timestamp_col,
            scaler_train_ratio=scaler_train_ratio,
            split_config_path=normalization_split_config_path,
            robust_scaler_path=rsp,
            fit_mask=fit_mask if need_fit_mask else None,
            fit_split_mode=fit_split_mode if need_fit_mask else None,
        )

    # --- save ---
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    out.to_csv(output_csv, index=False)

    if write_split_meta:
        try:
            from split_bounds import (
                discover_split_config_path,
                effective_rl_test_start,
                load_split_config,
                parse_split_bounds,
            )

            p = discover_split_config_path(Path(__file__).resolve().parent)
            if p is not None:
                meta = load_split_config(p)
                meta_path = os.path.splitext(output_csv)[0] + "_split_meta.json"
                os.makedirs(os.path.dirname(meta_path) or ".", exist_ok=True)
                try:
                    bounds = parse_split_bounds(meta)
                    t_eff = effective_rl_test_start(bounds)
                    meta = {
                        **meta,
                        "rl_alignment": {
                            "bounds_utc_iso": {k: v.isoformat() for k, v in bounds.items()},
                            "effective_test_start_utc_iso": t_eff.isoformat(),
                            "effective_test_start_rule": "max(test_start, rl_train_end) — RL 回测不早于 rl_train_end",
                            "rl_train_mask_rule": (
                                "timestamp > lstm_train_end "
                                "AND timestamp <= rl_train_end "
                                "AND timestamp < effective_test_start"
                            ),
                            "rl_test_mask_rule": "timestamp >= effective_test_start",
                            "moe_data_config_note": (
                                "MoEDataConfig.rl_train_start / rl_train_end（可选，ISO UTC）在存在 split_config 时"
                                "可进一步收紧或覆盖 RL 训练窗；见 RLModel/moe_hard.moe_train_test_split"
                            ),
                        },
                    }
                except (KeyError, ValueError, TypeError) as e:
                    warnings.warn(
                        f"data_engineering: could not parse split bounds for rl_alignment in sidecar JSON ({e!r}); "
                        f"writing raw split_config only to {meta_path}.",
                        UserWarning,
                        stacklevel=2,
                    )
                with open(meta_path, "w", encoding="utf-8") as fp:
                    json.dump(meta, fp, indent=2, ensure_ascii=False)
        except Exception as e:
            warnings.warn(
                f"data_engineering: failed to write split meta sidecar ({e!r}).",
                UserWarning,
                stacklevel=2,
            )

    return out

def resample_bars(
    df: pd.DataFrame,
    minutes: int = 15,
    timestamp_col: str = "timestamp",
    close_col: str = "close_5m",
) -> pd.DataFrame:
    """
    Resample a 5-minute ``data_e`` DataFrame to a coarser bar frequency.

    Aggregation rules per column type:
      - close        → last (end-of-window price)
      - log_ret_1    → sum  (log returns are additive)
      - volume_zscore→ mean (average intensity over window)
      - hl_range / log_hl → max (worst-case intra-window volatility)
      - everything else (smoothed indicators, regime one-hots) → last
    """
    if minutes <= 5:
        return df.copy()

    tmp = df.copy()
    tmp[timestamp_col] = pd.to_datetime(tmp[timestamp_col], utc=True)
    tmp = tmp.sort_values(timestamp_col).reset_index(drop=True)

    sum_cols = [c for c in tmp.columns if c.startswith("log_ret")]
    mean_cols = [c for c in tmp.columns if "volume_zscore" in c]
    max_cols = [c for c in tmp.columns if c.startswith("hl_range") or c.startswith("log_hl")]
    special = set(sum_cols + mean_cols + max_cols + [timestamp_col])
    last_cols = [c for c in tmp.columns if c not in special]

    agg: dict[str, str] = {}
    agg[timestamp_col] = "last"
    for c in sum_cols:
        agg[c] = "sum"
    for c in mean_cols:
        agg[c] = "mean"
    for c in max_cols:
        agg[c] = "max"
    for c in last_cols:
        agg[c] = "last"

    orig_cols = [c for c in tmp.columns]
    tmp = tmp.set_index(pd.DatetimeIndex(tmp[timestamp_col]))
    resampled = tmp.resample(f"{minutes}min").agg(agg)
    resampled = resampled.dropna(subset=[close_col]).reset_index(drop=True)
    resampled = resampled[[c for c in orig_cols if c in resampled.columns]]
    return resampled


if __name__ == "__main__":
    df_e = data_engineering(
        input_csv="data/BTCUSDT_combined_klines_20210201_20260201_hmm_states4_labeled_lstm_nextregime_ts32_predictions.csv",
        output_csv="data/data_e.csv",
        close_col="close_5m",
        state_col="state",
        hmm_pred_state_col="hmm_predicted_state",
        lstm_pred_state_col="lstm_predicted_state",
        pred_state_source="both",
        timestamp_col="timestamp",
        n_states=4,
        hmm_label_source_col="target",
        lstm_label_source_col="lstm_predicted_regime",
        # Winsorize (train 1%/99% → clip all rows) runs before RobustScaler when both True.
        winsorize_features=True,
        # winsorize_features=True → writes data/data_e_winsor_meta.json (default path)
        # RobustScaler: fit on RL train window (split_config rl_train_mask, else row-order train_ratio)
        normalize_features=True,
        # normalize_features=True → data/data_e_robust_scaler.joblib + _robust_scaler_meta.json
        multi_horizon_features=True
        # multi_horizon_features=True → roll_vol_{5,20,60}_5m, cum_ret_1h/4h_5m, adx_slope_5m (if adx_5m exists)
    )
    print("Saved:", "data/data_e.csv", "shape:", df_e.shape)
    print("Columns:", df_e.columns.tolist())