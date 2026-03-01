import pandas as pd
import numpy as np
import os

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
    input_csv: str = "data/BTCUSDT_combined_klines_20210201_20260201_states4_labeled.csv",
    output_csv: str = "data/data_e.csv",
    close_col: str = "close_5m",
    state_col: str = "state",
    regime_col: str = "regime",
    timestamp_col: str = "timestamp",
    n_states: int = 4,
) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    # --- build relative price features (avoid price level) ---
    # Use existing open/high/low if present; otherwise skip these features.
    if "open_5m" in df.columns:
        df["co_ret_5m"] = (df[close_col] / (df["open_5m"] + 1e-12)) - 1.0
    if "high_5m" in df.columns and "low_5m" in df.columns:
        df["hl_range_norm_5m"] = (df["high_5m"] - df["low_5m"]) / (df[close_col] + 1e-12)
        df["log_hl_5m"] = np.log((df["high_5m"] + 1e-12) / (df["low_5m"] + 1e-12))

    # --- one-hot for state ---
    if state_col in df.columns:
        cats = list(range(n_states))
        state_oh = pd.get_dummies(pd.Categorical(df[state_col], categories=cats), prefix=state_col)
        df = pd.concat([df.drop(columns=[state_col]), state_oh], axis=1)

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
    # add one-hot cols if created
    preferred_feats += [c for c in df.columns if c.startswith(f"{state_col}_")]

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
    keep += feature_cols

    out = df[keep].copy()

    # ensure numeric features are float32 where possible
    for c in feature_cols + [close_col]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out.dropna().reset_index(drop=True)

    # --- save ---
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    out.to_csv(output_csv, index=False)
    return out

if __name__ == "__main__":
    df_e = data_engineering(
        input_csv="data/BTCUSDT_combined_klines_20210201_20260201_states4_labeled.csv",
        output_csv="data/data_e.csv",
        close_col="close_5m",
        state_col="state",
        regime_col="regime",
        timestamp_col="timestamp",
        n_states=4,
    )
    print("Saved:", "data/data_e.csv", "shape:", df_e.shape)
    print("Columns:", df_e.columns.tolist())