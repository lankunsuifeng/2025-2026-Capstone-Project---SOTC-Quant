import json
import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


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
    regime_col: str = "regime",
    timestamp_col: str = "timestamp",
    n_states: int = 4,
    onehot_pred_states: bool = True,
    write_split_meta: bool = True,
    # HMMLSTM export: ``target`` = HMM next-bar class; ``lstm_predicted_regime`` = LSTM output
    hmm_label_source_col: str | None = "target",
    lstm_label_source_col: str | None = "lstm_predicted_regime",
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

    # Drop raw HMM latent ``state``; RL features use ``hmm_predicted_state_*`` / ``lstm_predicted_state_*`` only.
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
    for c in [timestamp_col, regime_col, close_col]:
        if c in df.columns:
            keep.append(c)
    # keep raw predicted_state columns too (useful as RL targets / debugging)
    for c in pred_state_cols:
        keep.append(c)
    keep += feature_cols

    out = df[keep].copy()

    # ensure numeric features are float32 where possible
    for c in feature_cols + [close_col] + pred_state_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out.dropna().reset_index(drop=True)

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

if __name__ == "__main__":
    df_e = data_engineering(
        input_csv="data/BTCUSDT_combined_klines_20210201_20260201_hmm_states4_labeled_lstm_nextregime_ts32_predictions.csv",
        output_csv="data/data_e.csv",
        close_col="close_5m",
        state_col="state",
        hmm_pred_state_col="hmm_predicted_state",
        lstm_pred_state_col="lstm_predicted_state",
        pred_state_source="both",
        regime_col="regime",
        timestamp_col="timestamp",
        n_states=4,
        hmm_label_source_col="target",
        lstm_label_source_col="lstm_predicted_regime",
    )
    print("Saved:", "data/data_e.csv", "shape:", df_e.shape)
    print("Columns:", df_e.columns.tolist())