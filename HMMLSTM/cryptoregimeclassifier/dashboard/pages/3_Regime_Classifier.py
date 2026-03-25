# dashboard/pages/3_HMM_Labeling_simple_nographs.py
import streamlit as st
import sys
import os
import pandas as pd
import numpy as np
import joblib

# add src path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.regime_label import (
    fit_hmm_features,
    transform_hmm_features,
    train_hmm,
    map_states_to_regimes,
)
from src.split_bounds import try_load_split_config

DATA_FOLDER = "data/"
MODELS_FOLDER = "models/"
os.makedirs(MODELS_FOLDER, exist_ok=True)

st.set_page_config(page_title="HMM Labeling", layout="wide")
st.title("HMM Regime Labeling")

# --- File selection ---
if not os.path.exists(DATA_FOLDER):
    st.error(f"Data folder '{DATA_FOLDER}' not found.")
    st.stop()

feature_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith('_features.csv')]
if not feature_files:
    st.warning("No *_features.csv files found in data/. Run feature engineering first.")
    st.stop()

selected_file = st.selectbox("Feature file", sorted(feature_files))
df = pd.read_csv(os.path.join(DATA_FOLDER, selected_file), parse_dates=['timestamp'])
st.markdown(f"Rows loaded: **{len(df):,}**")

# --- Feature & param selection ---
# (rsi may be rsi_14_5m in your pipeline; we guard for missing)
default_features = ['atr_norm_5m', 'bb_width_5m', 'adx_5m', 'volume_zscore_50_5m', 'log_ret_1_5m']

# [atr_norm_5m, bb_width_5m, adx_5m, volume_zscore_50_5m, taker_buy_ratio_5m, vwap_skew_5m, trade_imbalance_5m]


available_features = [
    c for c in df.columns
    if c != 'timestamp' and not c.startswith(('open_', 'high_', 'low_', 'close_'))
]
hmm_features = st.multiselect(
    "Select features for HMM",
    options=sorted(available_features),
    default=[f for f in default_features if f in available_features]
)

n_states = st.slider("Number of HMM states", min_value=2, max_value=8, value=4, step=1)

# --- HMM fit cutoff (no leakage on later rows) ---
st.subheader("HMM fit window (anti-leakage)")
_ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
_min_d, _max_d = _ts.min().date(), _ts.max().date()
st.caption(
    "Scaler, PCA (if any), and HMM are **fit only** on rows with `timestamp.date` ≤ cutoff. "
    "Labels are then predicted for **all** loaded rows. Use the latest date only if you accept full-sample fit."
)
_default_cutoff = _max_d
_scfg = try_load_split_config()
if _scfg and _scfg.get("hmm_fit_end"):
    try:
        _d = pd.Timestamp(_scfg["hmm_fit_end"])
        if _d.tzinfo is None:
            _d = _d.tz_localize("UTC")
        else:
            _d = _d.tz_convert("UTC")
        _d_date = _d.date()
        if _min_d <= _d_date <= _max_d:
            _default_cutoff = _d_date
    except (ValueError, TypeError):
        pass
train_cutoff = st.date_input(
    "HMM fit end date (inclusive, by calendar day)",
    value=_default_cutoff,
    min_value=_min_d,
    max_value=_max_d,
    key="hmm_train_cutoff_date",
)
if _scfg and _scfg.get("hmm_fit_end"):
    st.caption(f"Default from repo `split_config.json` **hmm_fit_end** (editable above).")

# Dimensionality reduction options
st.sidebar.subheader("Dimensionality Reduction")
use_scaler = st.sidebar.checkbox("Scale (StandardScaler)", value=True)

# 仅支持 None / PCA，移除 CNN Autoencoder 和 GPU 相关逻辑
reduction_method = st.sidebar.radio(
    "Reduction method",
    options=["None", "PCA"],
    index=1,  # Default to PCA
    help="PCA: Fast linear reduction. Select 'None' to use raw (scaled) features."
)

n_components = None
if reduction_method == "PCA":
    max_dim = max(1, len(hmm_features))
    n_components = st.sidebar.slider(
        "PCA components",
        min_value=1,
        max_value=max_dim,
        value=min(4, max_dim)
    )

# --- Train action ---
if st.button("Train HMM and label"):
    if not hmm_features:
        st.error("Select at least one feature.")
        st.stop()

    # drop rows that have NA in selected features to keep alignment clean
    valid_mask = df[hmm_features].notna().all(axis=1)
    if not valid_mask.any():
        st.error("No rows with all selected features available (after NA filter).")
        st.stop()

    df_used = df.loc[valid_mask].copy()
    df_used["timestamp"] = pd.to_datetime(df_used["timestamp"], utc=True, errors="coerce")
    df_fit = df_used.loc[df_used["timestamp"].dt.date <= train_cutoff].copy()
    if len(df_fit) < n_states * 5:
        st.error(
            f"Too few rows for HMM fit up to {train_cutoff}: {len(df_fit)} rows "
            f"(need at least ~{n_states * 5} for n_states={n_states}). Choose a later cutoff."
        )
        st.stop()
    st.info(
        f"HMM fit rows: **{len(df_fit):,}** (≤ {train_cutoff}) · "
        f"Label rows (full): **{len(df_used):,}**"
    )

    # 创建进度显示容器
    progress_container = st.container()
    status_text = progress_container.empty()
    progress_bar = progress_container.progress(0)
    
    try:
        # Step 1: 准备特征
        status_text.text("📊 Step 1/5: Preparing features and filtering data...")
        progress_bar.progress(5)
        
        use_pca = (reduction_method == "PCA")

        # Step 2: 特征降维（仅 PCA 或不降维）
        if use_pca:
            status_text.text("📉 Step 2/5: Applying PCA reduction...")
            progress_bar.progress(30)
        else:
            status_text.text("📉 Step 2/5: Skipping dimensionality reduction (using raw features)...")
            progress_bar.progress(30)
        
        X_train, scaler, reduction_model, df_fit_clean = fit_hmm_features(
            df_fit,
            feature_list=hmm_features,
            n_components=n_components,
            scale=use_scaler,
            use_pca=use_pca,
        )
        X_all = transform_hmm_features(
            df_used,
            feature_list=hmm_features,
            scaler=scaler,
            reduction_model=reduction_model,
            scale=use_scaler,
            use_pca=use_pca,
        )

        # Step 3: 训练HMM
        status_text.text("🎯 Step 3/5: Training HMM model (this may take a minute)...")
        progress_bar.progress(60)

        hmm_model = train_hmm(X_train, n_states=n_states, n_iter=200, random_state=42)

        # Step 4: 生成标签
        status_text.text("🏷️ Step 4/5: Generating regime labels...")
        progress_bar.progress(80)

        states_all = hmm_model.predict(X_all)
        states_train = hmm_model.predict(X_train)
        labeled_df = df_used.copy()
        labeled_df["state"] = states_all

        # Regime names from train-period stats only (no future rows in groupby)
        state_mapping = map_states_to_regimes(df_fit_clean, states_train, main_tf="5m")
        labeled_df["regime"] = labeled_df["state"].map(state_mapping)

        # Step 5: 保存到session
        status_text.text("💾 Step 5/5: Saving results...")
        progress_bar.progress(95)

        # Store in session
        st.session_state['labeled_df'] = labeled_df
        st.session_state['hmm_model'] = hmm_model
        st.session_state['scaler'] = scaler
        st.session_state['reduction_model'] = reduction_model  # PCA or None
        st.session_state['reduction_method'] = reduction_method
        st.session_state['hmm_features'] = hmm_features
        st.session_state['state_mapping'] = state_mapping
        st.session_state['hmm_input_features'] = X_all
        st.session_state['hmm_train_features'] = X_train
        st.session_state['hmm_fit_cutoff_date'] = str(train_cutoff)
        st.session_state['hmm_fit_n_rows'] = len(df_fit_clean)

        # 完成
        progress_bar.progress(100)
        status_text.text("✅ Training completed successfully!")
        
        # 短暂延迟后清除进度显示
        import time
        time.sleep(0.5)
        progress_container.empty()
        
        st.success(f"✅ HMM trained and data labeled successfully! Processed {len(labeled_df):,} rows. See metrics below.")
        
    except Exception as e:
        progress_container.empty()
        st.error(f"❌ Error during training: {str(e)}")
        st.exception(e)  # 显示详细错误堆栈
        st.stop()

# --- If labeled, show metrics and tables (no charts) ---
if 'labeled_df' in st.session_state:
    labeled_df = st.session_state['labeled_df']
    hmm_model = st.session_state['hmm_model']
    state_mapping = st.session_state['state_mapping']
    hmm_features = st.session_state['hmm_features']
    X_for_score = st.session_state.get('hmm_train_features', None)
    if X_for_score is None:
        X_for_score = st.session_state.get('hmm_input_features', None)

    st.header("Model & Label Summary")
    if st.session_state.get("hmm_fit_cutoff_date"):
        st.markdown(
            f"- HMM fit cutoff (inclusive date): **{st.session_state['hmm_fit_cutoff_date']}**  "
            f"· rows used for fit (after NA drop): **{st.session_state.get('hmm_fit_n_rows', 'N/A')}**"
        )

    # log-likelihood, AIC, BIC (diag-cov GaussianHMM) on **train** matrix when available
    try:
        ll = float(hmm_model.score(X_for_score)) if X_for_score is not None else float('nan')
    except Exception:
        ll = float('nan')

    n_obs = int(X_for_score.shape[0]) if X_for_score is not None else len(labeled_df)
    n_labeled_total = len(labeled_df)
    n_components = int(hmm_model.n_components)
    n_features = int(X_for_score.shape[1]) if X_for_score is not None else len(hmm_features)

    # parameter count (approx): startprob (n-1) + trans (n*(n-1)) + means (n*d) + diag covars (n*d)
    k_params = (n_components - 1) + (n_components * (n_components - 1)) + (n_components * n_features) + (n_components * n_features)

    aic = (None if np.isnan(ll) else (-2.0 * ll + 2 * k_params))
    bic = (None if np.isnan(ll) else (-2.0 * ll + k_params * np.log(max(n_obs, 1))))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Log-Likelihood", f"{ll:.2f}" if not np.isnan(ll) else "N/A")
    c2.metric("AIC", f"{aic:.2f}" if aic is not None else "N/A")
    c3.metric("BIC", f"{bic:.2f}" if bic is not None else "N/A")
    c4.metric("Model params (k)", f"{k_params}")

    reduction_method = st.session_state.get('reduction_method', 'None')
    reduction_label = {
        'None': 'No reduction',
        'PCA': 'PCA'
    }.get(reduction_method, 'Unknown')
    
    st.markdown(
        f"- Observations for likelihood/AIC/BIC (fit set): **{n_obs}**  \n"
        f"- Rows labeled (full): **{n_labeled_total}**  \n"
        f"- HMM states: **{n_components}**  \n"
        f"- Reduction method: **{reduction_label}**  \n"
        f"- Feature dims (after reduction): **{n_features}**"
    )

    # state counts
    st.subheader("State counts and percentages")
    state_counts = labeled_df['state'].value_counts().sort_index()
    state_pct = (state_counts / state_counts.sum() * 100).round(2)
    freq_df = pd.DataFrame({
        'state': state_counts.index,
        'count': state_counts.values,
        'percent': state_pct.values,
        'regime_label': [state_mapping.get(s, f"State_{s}") for s in state_counts.index]
    }).reset_index(drop=True)
    st.dataframe(freq_df, use_container_width=True)

    # transition matrix
    st.subheader("Transition matrix (rows=from, cols=to)")
    trans = np.round(hmm_model.transmat_, 4)
    trans_df = pd.DataFrame(trans, columns=[f"to_{i}" for i in range(trans.shape[1])], index=[f"from_{i}" for i in range(trans.shape[0])])
    st.dataframe(trans_df, use_container_width=True)

    # per-state feature summary (means & std) – use whatever exists in DF
    st.subheader("Per-state feature summary (means & std)")
    interpret_feats = [f for f in [
        'log_ret_1_5m','atr_norm_5m','adx_5m','bb_width_5m','rsi_14_5m','volume_zscore_50_5m','macd_hist_5m'
    ] if f in labeled_df.columns]
    if interpret_feats:
        stats = labeled_df.groupby('state')[interpret_feats].agg(['mean', 'std']).round(6)
        stats.columns = ["_".join(col).strip() for col in stats.columns.values]
        stats = stats.reset_index()
        stats['regime_label'] = stats['state'].map(lambda s: state_mapping.get(s, f"State_{s}"))
        st.dataframe(stats, use_container_width=True)
    else:
        st.write("No common interpretability features present for summary.")

    # Save labeled data (HMM states + regime names)
    st.subheader("Save outputs")

    def _feature_stem_for_hmm(fname: str) -> str:
        if fname.endswith("_fe_features.csv"):
            return fname[: -len("_fe_features.csv")]
        if fname.endswith("_features.csv"):
            return fname[: -len("_features.csv")]
        return os.path.splitext(fname)[0]

    save_name = f"{_feature_stem_for_hmm(selected_file)}_hmm_states{n_components}_labeled.csv"
    save_path = os.path.join(DATA_FOLDER, save_name)
    if st.button(f"Save HMM-labeled CSV as `{save_name}`"):
        labeled_df.to_csv(save_path, index=False)
        st.success(f"Labeled data saved to `{save_path}`")

    # Save model artifacts
    reduction_method = st.session_state.get('reduction_method', 'None')
    artifact_name = f"Save model artifacts (HMM + scaler + {reduction_method})"
    
    if st.button(artifact_name):
        artifact = {
            'hmm_model': hmm_model,
            'scaler': st.session_state.get('scaler', None),
            'reduction_model': st.session_state.get('reduction_model', None),  # PCA or None
            'reduction_method': st.session_state.get('reduction_method', 'None'),
            'features': st.session_state.get('hmm_features', []),
            'state_mapping': st.session_state.get('state_mapping', {}),
            'n_states': n_components,
            'file_source': selected_file,
            'hmm_fit_cutoff_date': st.session_state.get('hmm_fit_cutoff_date'),
            'hmm_fit_n_rows': st.session_state.get('hmm_fit_n_rows'),
        }
        _hmm_stem = _feature_stem_for_hmm(selected_file)
        model_name = f"hmm_{_hmm_stem}_states{n_components}.joblib"
        model_path = os.path.join(MODELS_FOLDER, model_name)
        joblib.dump(artifact, model_path)
        st.success(f"Saved model artifacts to {model_path}")
