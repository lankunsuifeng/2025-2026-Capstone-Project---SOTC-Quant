# src/regime_label.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from hmmlearn import hmm


def fit_hmm_features(df, feature_list, n_components=None, scale=True,
                     use_pca=True, use_autoencoder=False, autoencoder_epochs=100,
                     progress_callback=None):
    """
    Fit StandardScaler and/or PCA on df only (train period). Drops rows with NA in feature_list.

    Returns
    -------
    X : np.ndarray
        Transformed feature matrix (rows aligned with df_aligned).
    scaler : StandardScaler or None
    reduction_model : PCA or None
    df_aligned : pd.DataFrame
        Subset of df with complete features; same row count as X (for regime naming on train only).
    """
    _ = (use_autoencoder, autoencoder_epochs, progress_callback)  # backward-compat unused

    features = df[feature_list].copy()
    mask = features.notna().all(axis=1)
    if not mask.any():
        raise ValueError("No rows with complete HMM features after NA filter (fit_hmm_features).")
    df_aligned = df.loc[mask].copy()
    feats = features.loc[mask]
    X = feats.values.astype(np.float64)

    scaler = None
    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    reduction_model = None
    if use_pca:
        if n_components is None:
            n_components = min(X.shape[0], X.shape[1])
        reduction_model = PCA(n_components=n_components, random_state=42)
        X = reduction_model.fit_transform(X)

    return X, scaler, reduction_model, df_aligned


def transform_hmm_features(df, feature_list, scaler, reduction_model, scale=True, use_pca=True):
    """
    Apply a previously fitted scaler/PCA to all rows of df. Requires no NA in feature_list.
    """
    features = df[feature_list]
    if not features.notna().all(axis=1).all():
        raise ValueError(
            "transform_hmm_features: missing values in HMM feature columns; "
            "filter to complete rows before calling."
        )
    X = features.values.astype(np.float64)
    if scale:
        if scaler is None:
            raise ValueError("scale=True but scaler is None")
        X = scaler.transform(X)
    if use_pca:
        if reduction_model is None:
            raise ValueError("use_pca=True but reduction_model is None")
        X = reduction_model.transform(X)
    return X


def get_hmm_features(df, feature_list, n_components=None, scale=True,
                     use_pca=True, use_autoencoder=False, autoencoder_epochs=100,
                     progress_callback=None):
    """
    Backward-compatible: fit scaler/PCA on the entire df (same as old behavior — may leak if df includes future test period).

    Returns (X, scaler, reduction_model) only — no df_aligned.
    """
    X, scaler, reduction_model, _ = fit_hmm_features(
        df,
        feature_list,
        n_components=n_components,
        scale=scale,
        use_pca=use_pca,
        use_autoencoder=use_autoencoder,
        autoencoder_epochs=autoencoder_epochs,
        progress_callback=progress_callback,
    )
    return X, scaler, reduction_model

def train_hmm(features, n_states=3, n_iter=150, random_state=42):
    """
    Trains a diagonal-covariance Gaussian HMM.
    """
    model = hmm.GaussianHMM(
        n_components=n_states,
        covariance_type="diag",
        n_iter=n_iter,
        random_state=random_state,
        verbose=False
    )
    model.fit(features)
    return model

def map_states_to_regimes(df, labels, main_tf='5m'):
    """
    Map HMM states -> interpretable regimes for the 6-state case:
      Squeeze, Range, Weak Trend, Strong Trend, Choppy High-Vol, Volatility Spike
    Uses ATR-normalized (volatility) and ADX (trend strength). Fully deterministic.

    For leak-free naming, pass only the train-period dataframe and train-period state labels
    (same length, row-aligned), e.g. df_fit_clean and hmm_model.predict(X_train).
    """
    df_labeled = df.copy()
    if len(labels) != len(df_labeled):
        raise ValueError(
            f"map_states_to_regimes: len(labels)={len(labels)} != len(df)={len(df_labeled)}"
        )
    df_labeled['state'] = labels
    n_states = int(len(np.unique(labels)))

    vol_col = f'atr_norm_{main_tf}'
    bb_col  = f'bb_width_{main_tf}'
    trend_col = f'adx_{main_tf}'

    # ---- fallbacks if required columns missing
    if trend_col not in df_labeled.columns:
        # no ADX -> purely ordinal by volatility
        base = vol_col if vol_col in df_labeled.columns else (bb_col if bb_col in df_labeled.columns else None)
        if base is None:
            return {s: f"State_{i}" for i, s in enumerate(sorted(df_labeled['state'].unique()))}
        order = df_labeled.groupby('state')[base].mean().sort_values().index.tolist()
        names = ["Squeeze","Range","Weak Trend","Strong Trend","Choppy High-Vol","Volatility Spike"]
        return {s: (names[i] if i < len(names) else f"State_{i}") for i, s in enumerate(order)}

    if vol_col not in df_labeled.columns and bb_col not in df_labeled.columns:
        # no volatility proxy at all -> rank by trend only
        order = df_labeled.groupby('state')[trend_col].mean().sort_values().index.tolist()
        return {s: f"State_{i}" for i, s in enumerate(order)}

    # ---- choose volatility proxy (prefer ATR, else BB width)
    vol_proxy = vol_col if vol_col in df_labeled.columns else bb_col

    # ---- compute per-state means
    stats = df_labeled.groupby('state')[[vol_proxy, trend_col]].mean()

    if n_states != 6:
        # generic mapping for other counts (keep existing behavior simple)
        # low→high vol, then label generically
        vol_order = stats.sort_values(by=vol_proxy).index.tolist()
        return {s: f"State_{i}" for i, s in enumerate(vol_order)}

    # =========================
    # 6-STATE DETERMINISTIC MAP
    # =========================

    # sort states by volatility (ascending)
    vol_order = stats.sort_values(by=vol_proxy).index.tolist()

    # extremes
    squeeze  = vol_order[0]
    vol_spike = vol_order[-1]

    # four middle states
    mids = vol_order[1:-1]
    # split into low-vol band (2) and high-vol band (2) by volatility rank
    low_band  = mids[:2]
    high_band = mids[2:]

    # within each band, use ADX to split
    # low band: lower ADX -> Range ; higher ADX -> Weak Trend
    low_band_sorted = stats.loc[low_band].sort_values(by=trend_col).index.tolist()
    rng, weak_trend = low_band_sorted[0], low_band_sorted[1]

    # high band: lower ADX -> Choppy High-Vol ; higher ADX -> Strong Trend
    high_band_sorted = stats.loc[high_band].sort_values(by=trend_col).index.tolist()
    choppy, strong_trend = high_band_sorted[0], high_band_sorted[1]

    mapping = {
        squeeze:        'Squeeze',
        rng:            'Range',
        weak_trend:     'Weak Trend',
        strong_trend:   'Strong Trend',
        choppy:         'Choppy High-Vol',
        vol_spike:      'Volatility Spike',
    }
    return mapping
