# dashboard/pages/5_LSTM_Training.py
import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.lstm_model import create_sequences, build_lstm_model
from src.split_bounds import (
    try_load_split_config,
    parse_split_bounds,
    lstm_train_mask,
    lstm_test_mask,
)

DATA_FOLDER = "data/"
MODEL_FOLDER = "models/"

# Repo-root relative export target for RL pipeline (Captstone Codes/RLModel/data)
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
RL_DATA_FOLDER = os.path.join(_REPO_ROOT, "RLModel", "data")


def build_lstm_prediction_export_df(
    df: pd.DataFrame,
    feature_cols: list,
    scaler,
    model,
    time_steps: int,
    regime_map: dict,
    inv_regime_map: dict,
    predict_batch_size: int = 4096,
) -> pd.DataFrame:
    """Sliding-window softmax predictions using **batched** inference.

    Rows are aligned to ``df.iloc[time_steps:]`` (matches training sequences).
    Instead of calling ``model.predict`` once per row, windows are stacked into
    batches of *predict_batch_size* and predicted together — typically 100-500x
    faster than the naive per-row loop.
    """
    X = df[list(feature_cols)].values.astype(np.float64)
    Xs = scaler.transform(X)
    n = len(df)
    total = n - time_steps

    all_probs_parts: list[np.ndarray] = []
    progress = st.progress(0, text="Batched prediction: starting…")

    for batch_start in range(0, total, predict_batch_size):
        batch_end = min(batch_start + predict_batch_size, total)
        batch = np.stack(
            [Xs[j : j + time_steps] for j in range(batch_start, batch_end)]
        )
        probs = model.predict(batch, verbose=0)
        all_probs_parts.append(probs)
        pct = batch_end / total
        progress.progress(pct, text=f"Predicted {batch_end:,} / {total:,} rows ({pct:.0%})")

    progress.empty()
    all_probs = np.concatenate(all_probs_parts, axis=0)  # (total, num_classes)

    pred_classes = np.argmax(all_probs, axis=1)
    predictions = [inv_regime_map.get(int(c), f"unknown_{c}") for c in pred_classes]
    confidences = np.max(all_probs, axis=1).tolist()

    pred_df = df.iloc[time_steps:].copy().reset_index(drop=True)
    pred_df["lstm_predicted_regime"] = predictions
    pred_df["lstm_prediction_confidence"] = confidences
    for regime_name, class_id in regime_map.items():
        cid = int(class_id)
        pred_df[f"lstm_prob_{regime_name}"] = all_probs[:, cid].tolist()
    return pred_df


st.set_page_config(page_title="LSTM Model Training", layout="wide")
st.title("Train Regime Prediction Model (LSTM)")
st.markdown("""
This page trains an LSTM network to predict the HMM-generated market regime for the next bar.
""")

# ----- 1. Load Data -----
try:
    labeled_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith("_labeled.csv")]
except FileNotFoundError:
    st.error(f"Data folder '{DATA_FOLDER}' not found.")
    labeled_files = []

if not labeled_files:
    st.warning("No HMM-labeled files found. Please run 'HMM Labeling' first.")
    st.stop()

selected_file = st.selectbox("Select HMM-labeled dataset", labeled_files)
df = pd.read_csv(os.path.join(DATA_FOLDER, selected_file))


# ----- 2. Target and Feature Preparation -----
unique_regimes = sorted(df['regime'].dropna().unique())
regime_map = {regime: idx for idx, regime in enumerate(unique_regimes)}
inv_regime_map = {v: k for k, v in regime_map.items()}
num_classes = len(regime_map)

df['target'] = df['regime'].map(regime_map)
# IMPORTANT (label alignment):
# We do NOT shift(-1) here. With create_sequences(), each window X[i:i+time_steps]
# is paired with y[i+time_steps], which corresponds to the *next bar* after the
# window end (i+time_steps-1 -> i+time_steps). Shifting here would make it 2 bars ahead.

feature_cols = [col for col in df.columns if not (
    col.startswith(('open_', 'high_', 'low_', 'close_', 'volume_')) or
    col in ['timestamp', 'state', 'regime', 'target']
)]

df.dropna(inplace=True)
X_raw = df[feature_cols]
y_raw = df['target'].astype(int)

# ----- 3. Train/Test Split -----
st.sidebar.header("Data & Model Configuration")
_split_cfg = try_load_split_config()
_use_time_split = st.sidebar.checkbox(
    "Use repo split_config.json (time-based)",
    value=_split_cfg is not None,
    disabled=_split_cfg is None,
    help="Train: timestamp <= lstm_train_end. Test: timestamp >= test_start.",
)
if _split_cfg is None:
    _use_time_split = False
if _use_time_split and "timestamp" not in df.columns:
    st.sidebar.warning("No `timestamp` column: cannot use split_config; using ratio split.")
    _use_time_split = False
test_size = st.sidebar.slider(
    "Test set size (only if not using split_config)",
    0.1, 0.5, 0.2, 0.05,
    disabled=_use_time_split,
)

if _use_time_split:
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    _bounds = parse_split_bounds(_split_cfg)
    _m_tr = lstm_train_mask(df, _bounds)
    _m_te = lstm_test_mask(df, _bounds)
    _df_tr = df.loc[_m_tr]
    _df_te = df.loc[_m_te]
    st.sidebar.caption(f"Train rows: **{len(_df_tr):,}** | Test rows: **{len(_df_te):,}**")
    if len(_df_tr) == 0 or len(_df_te) == 0:
        st.error("split_config produced empty train or test set. Adjust dates in split_config.json or disable this option.")
        st.stop()
    X_train_raw = _df_tr[feature_cols]
    X_test_raw = _df_te[feature_cols]
    y_train_raw = _df_tr["target"].astype(int)
    y_test_raw = _df_te["target"].astype(int)
else:
    n = len(df)
    train_n = int(n * (1 - test_size))
    X_train_raw, X_test_raw = X_raw.iloc[:train_n], X_raw.iloc[train_n:]
    y_train_raw, y_test_raw = y_raw.iloc[:train_n], y_raw.iloc[train_n:]

# ----- 4. Scaling -----
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)

# ----- 5. Create Sequences -----
time_steps = st.sidebar.slider("LSTM Lookback Window (Time Steps)", 16, 128, 32, 4)

X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_raw.values, time_steps)
X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_raw.values, time_steps)

# One-hot encode the labels
y_train_cat = to_categorical(y_train_seq, num_classes=num_classes)
y_test_cat = to_categorical(y_test_seq, num_classes=num_classes)

st.write(f"**Training sequences shape:** {X_train_seq.shape}")
st.write(f"**Test sequences shape:** {X_test_seq.shape}")

# ----- 6. Model Training -----
st.sidebar.header("LSTM Architecture")
lstm_units = st.sidebar.slider("LSTM Units", 32, 256, 64, 8)
dense_units = st.sidebar.slider("Dense Units", 16, 128, 32, 8)
dropout_rate = st.sidebar.slider("Dropout Rate", 0.1, 0.5, 0.1, 0.05)

st.sidebar.header("Training Parameters")
epochs = st.sidebar.number_input("Epochs", 10, 200, 10, 5)
batch_size = st.sidebar.select_slider("Batch Size", options=[32, 64, 128, 256], value=256)

if st.button("Train LSTM Model", type="primary"):
    model = build_lstm_model(
        input_shape=(X_train_seq.shape[1], X_train_seq.shape[2]),
        num_classes=num_classes,
        lstm_units=lstm_units,
        dense_units=dense_units,
        dropout_rate=dropout_rate
    )
    st.write(model.summary())
    
    from sklearn.utils import class_weight
    
    class_weights = class_weight.compute_class_weight(
        'balanced',
        classes=np.unique(y_train_seq),
        y=y_train_seq
    )
    # Convert the array of weights into a dictionary that Keras expects
    class_weight_dict = dict(enumerate(class_weights))
    
    st.write("**Applied Class Weights:**")
    st.json({inv_regime_map[k]: f"{v:.2f}x" for k, v in class_weight_dict.items()})

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
    ]

    with st.spinner("Training LSTM model..."):
        history = model.fit(
            X_train_seq, y_train_cat,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.15, # Use last 15% of training data for validation
            callbacks=callbacks,
            verbose=1,
            class_weight=class_weight_dict,
            shuffle=False
        )
        st.session_state['lstm_model'] = model
        st.session_state['lstm_history'] = history.history
        st.session_state['scaler'] = scaler
        st.session_state['lstm_train_time_steps'] = int(time_steps)
        st.session_state['lstm_train_feature_cols'] = list(feature_cols)
        st.session_state['lstm_train_regime_map'] = dict(regime_map)
        st.session_state['lstm_train_inv_regime_map'] = dict(inv_regime_map)
        st.session_state['lstm_training_file'] = selected_file

# ----- 7. Evaluation & Saving -----
if 'lstm_model' in st.session_state:
    st.success("Model training complete!")
    model = st.session_state['lstm_model']
    history = st.session_state['lstm_history']
    scaler = st.session_state['scaler']

    ts_model = int(st.session_state.get('lstm_train_time_steps', time_steps))
    if st.session_state.get('lstm_train_time_steps') is not None and ts_model != time_steps:
        st.info(
            f"Sidebar time_steps is **{time_steps}**, but this model was trained with **{ts_model}**. "
            "Test metrics and CSV export use the trained window length."
        )
    X_test_seq_eval, y_test_seq_eval = create_sequences(
        X_test_scaled, y_test_raw.values, ts_model
    )

    # Plot training history
    st.subheader("Training & Validation Loss")
    fig, ax = plt.subplots()
    ax.plot(history['loss'], label='Training Loss')
    ax.plot(history['val_loss'], label='Validation Loss')
    ax.legend()
    st.pyplot(fig)

    # Evaluate on test set
    y_pred_probs = model.predict(X_test_seq_eval, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    unique_labels = sorted(list(np.unique(np.concatenate([y_test_seq_eval, y_pred]))))
    target_names = [inv_regime_map.get(i, f"Unknown_{i}") for i in unique_labels]

    st.subheader("Classification Report (Test Set)")
    report = classification_report(y_test_seq_eval, y_pred, labels=unique_labels, target_names=target_names, output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).transpose())

    st.subheader("Confusion Matrix (Test Set)")
    cm = confusion_matrix(y_test_seq_eval, y_pred, labels=unique_labels)
    fig2, ax2 = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=target_names, yticklabels=target_names, ax=ax2)
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("True")
    st.pyplot(fig2)

    # --- Save Model, Scaler, and Metadata ---
    if st.button("Save LSTM Model and Scaler"):
        os.makedirs(MODEL_FOLDER, exist_ok=True)
        
        model_path = os.path.join(MODEL_FOLDER, "lstm_regime_model.keras")
        model.save(model_path)
        
        scaler_path = os.path.join(MODEL_FOLDER, "scaler.joblib")
        joblib.dump(scaler, scaler_path)
        
        meta = {
            "model_type": "LSTM",
            "target": "regime",
            "regime_map": regime_map,
            "features": feature_cols,
            "time_steps": int(st.session_state.get("lstm_train_time_steps", time_steps)),
            "training_file": selected_file,
        }
        meta_path = os.path.join(MODEL_FOLDER, "lstm_model_metadata.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=4)
            
        st.success(f"Model saved to: `{model_path}`")
        st.success(f"Scaler saved to: `{scaler_path}`")

        st.success(f"Metadata saved to: `{meta_path}`")

    # --- Export full-series LSTM predictions (same CSV as training distribution, no extra leakage beyond training choices) ---
    st.subheader("Export LSTM predictions (full labeled file)")
    st.caption(
        "Uses the **current** labeled file in the dropdown, the **saved** scaler from this session, "
        f"and lookback **{ts_model}**. Re-train if you change the file or features. "
        f"Saves to **RLModel/data**: `{RL_DATA_FOLDER}`"
    )
    fc_export = st.session_state.get('lstm_train_feature_cols', feature_cols)
    rm_export = st.session_state.get('lstm_train_regime_map', regime_map)
    irm_export = st.session_state.get('lstm_train_inv_regime_map', inv_regime_map)
    _missing_fc = [c for c in fc_export if c not in df.columns]
    if _missing_fc:
        st.warning(f"Cannot export: missing columns in dataframe: {_missing_fc[:8]}…")
    elif st.button("Save LSTM predictions CSV to data/", key="save_lstm_pred_csv"):
        with st.spinner("Running sliding-window predictions…"):
            pred_full = build_lstm_prediction_export_df(
                df, fc_export, scaler, model, ts_model, rm_export, irm_export
            )
        _stem = os.path.splitext(selected_file)[0]
        pred_fname = f"{_stem}_lstm_nextregime_ts{ts_model}_predictions.csv"
        os.makedirs(RL_DATA_FOLDER, exist_ok=True)
        pred_path = os.path.join(RL_DATA_FOLDER, pred_fname)
        pred_full.to_csv(pred_path, index=False)
        st.success(f"Saved **{len(pred_full):,}** rows to `{pred_path}`")
