# src/regime_label.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from hmmlearn import hmm

# Import autoencoder (optional, will be imported only if needed)
try:
    from .autoencoder import CNNAutoencoder
    AUTOENCODER_AVAILABLE = True
except ImportError:
    try:
        from autoencoder import CNNAutoencoder
        AUTOENCODER_AVAILABLE = True
    except ImportError:
        AUTOENCODER_AVAILABLE = False
        CNNAutoencoder = None

def get_hmm_features(df, feature_list, n_components=None, scale=True, 
                     use_pca=True, use_autoencoder=False, autoencoder_epochs=100,
                     progress_callback=None):
    """
    Selects features, drops NA rows, scales, and (optionally) applies PCA or Autoencoder.
    
    Parameters:
    -----------
    df : pd.DataFrame
        输入数据框
    feature_list : list
        特征列名列表
    n_components : int, optional
        降维后的维度（PCA components 或 Autoencoder latent_dim）
    scale : bool
        是否标准化
    use_pca : bool
        是否使用PCA降维
    use_autoencoder : bool
        是否使用Autoencoder降维（如果True，会覆盖use_pca）
    autoencoder_epochs : int
        Autoencoder训练轮数
    progress_callback : callable, optional
        进度回调函数，用于Autoencoder训练时显示进度
    
    Returns:
    --------
    X : np.array
        降维后的特征矩阵
    scaler : StandardScaler or None
        标准化器
    reduction_model : PCA or DenseAutoencoder or None
        降维模型（PCA或Autoencoder）
    """
    features = df[feature_list].copy()
    features = features.dropna(axis=0, how='any')

    scaler = None
    X = features.values

    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(features.values)

    reduction_model = None
    
    # 优先使用Autoencoder（如果可用且启用）
    if use_autoencoder and AUTOENCODER_AVAILABLE:
        input_dim = X.shape[1]
        latent_dim = n_components if n_components is not None else 12
        
        reduction_model = CNNAutoencoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            filters=[64, 32, 16],
            kernel_size=3,
            dropout_rate=0.2,
            random_state=42
        )
        
        # 训练并降维
        X = reduction_model.fit_transform(X, epochs=autoencoder_epochs, verbose=0, progress_callback=progress_callback)
        
    elif use_pca:
        if n_components is None:
            # keep full rank if not specified
            n_components = min(features.shape[0], features.shape[1])
        reduction_model = PCA(n_components=n_components, random_state=42)
        X = reduction_model.fit_transform(X)
        # Optionally log variance explained (caller can print if needed)
        # print(f"PCA explained variance ratio: {np.sum(reduction_model.explained_variance_ratio_):.4f}")

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
    """
    import numpy as np
    import pandas as pd

    df_labeled = df.copy()
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
