# dashboard/pages/8_Historical_Prediction_Visualization.py
import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tensorflow.keras.models import load_model

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

DATA_FOLDER = "data/"
MODEL_FOLDER = "models/"

st.set_page_config(page_title="Historical Prediction Visualization", layout="wide")
st.title("Historical Prediction Visualization")
st.markdown("""
This page visualizes model predictions on historical data. 
Load a trained model and feature dataset to see how the model classifies different market regimes over time.
""")

# -------------------------------------------------------------------
# File Selection
# -------------------------------------------------------------------
st.sidebar.header("Configuration")

# Data file selection
try:
    feature_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith('_features.csv') or f.endswith('_labeled.csv')]
except FileNotFoundError:
    st.error(f"Data folder '{DATA_FOLDER}' not found.")
    feature_files = []

if not feature_files:
    st.warning("No feature files found. Please run feature engineering first.")
    st.stop()

selected_data_file = st.sidebar.selectbox("Select data file", sorted(feature_files))

# Model folder selection
model_folders = []
if os.path.exists(MODEL_FOLDER):
    model_folders = [f for f in os.listdir(MODEL_FOLDER) 
                     if os.path.isdir(os.path.join(MODEL_FOLDER, f))] or [MODEL_FOLDER]

selected_model_folder = st.sidebar.text_input(
    "Model folder path", 
    value=MODEL_FOLDER,
    help="Path to folder containing lstm_regime_model.keras, scaler.joblib, and lstm_model_metadata.json"
)

# Visualization options
st.sidebar.header("Visualization Options")
show_probabilities = st.sidebar.checkbox("Show probability distribution", value=False)
max_data_points = st.sidebar.number_input(
    "Max data points to visualize", 
    min_value=100, 
    max_value=10000, 
    value=1000, 
    step=100,
    help="Limit the number of points for performance"
)
start_date = st.sidebar.date_input("Start date (optional)", value=None)
end_date = st.sidebar.date_input("End date (optional)", value=None)

# -------------------------------------------------------------------
# Load Data and Model
# -------------------------------------------------------------------
@st.cache_resource
def load_model_artifacts(model_folder_path: str):
    """Load model, scaler, and metadata"""
    model_path = os.path.join(model_folder_path, "lstm_regime_model.keras")
    scaler_path = os.path.join(model_folder_path, "scaler.joblib")
    metadata_path = os.path.join(model_folder_path, "lstm_model_metadata.json")
    
    if not all(os.path.exists(p) for p in [model_path, scaler_path, metadata_path]):
        raise FileNotFoundError(f"Missing model files in {model_folder_path}")
    
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    return model, scaler, metadata

# Load data
df = pd.read_csv(os.path.join(DATA_FOLDER, selected_data_file))
if 'timestamp' in df.columns:
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

# Filter by date if specified
if start_date:
    df = df[df['timestamp'] >= pd.Timestamp(start_date)]
if end_date:
    df = df[df['timestamp'] <= pd.Timestamp(end_date)]

# Limit data points
if len(df) > max_data_points:
    df = df.tail(max_data_points).reset_index(drop=True)

st.info(f"Loaded {len(df):,} rows from {selected_data_file}")

# Try to load model
try:
    model, scaler, metadata = load_model_artifacts(selected_model_folder)
    st.success(f"Model loaded from {selected_model_folder}")
    
    # Display model info
    with st.expander("Model Information"):
        st.json(metadata)
        
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# -------------------------------------------------------------------
# Prepare Features and Run Predictions
# -------------------------------------------------------------------
feature_cols = metadata.get('features', [])
time_steps = metadata.get('time_steps', 64)
regime_map = metadata.get('regime_map', {})
inv_regime_map = {int(v): k for k, v in regime_map.items()}

# Check if required features exist
missing_features = [f for f in feature_cols if f not in df.columns]
if missing_features:
    st.error(f"Missing required features: {missing_features[:10]}...")
    st.stop()

# Ensure we have enough data
if len(df) < time_steps:
    st.error(f"Not enough data points. Need at least {time_steps}, got {len(df)}")
    st.stop()

# Prepare features
df_features = df[feature_cols].copy()
df_features = df_features.ffill().fillna(0)  # Forward fill then fill remaining with 0

# Create sequences and predict
st.subheader("Running Predictions...")
progress_bar = st.progress(0)

predictions = []
probabilities_list = []

# Use sliding window to predict for each time point
for i in range(time_steps, len(df_features)):
    if i % 100 == 0:
        progress_bar.progress((i - time_steps) / (len(df_features) - time_steps))
    
    # Get sequence window
    window = df_features.iloc[i - time_steps:i].values
    
    # Scale and reshape
    window_scaled = scaler.transform(window)
    window_batched = np.expand_dims(window_scaled, axis=0)
    
    # Predict
    probs = model.predict(window_batched, verbose=0)[0]
    pred_class = int(np.argmax(probs))
    pred_regime = inv_regime_map.get(pred_class, f"Unknown_{pred_class}")
    
    predictions.append(pred_regime)
    probabilities_list.append(probs.tolist())

progress_bar.progress(1.0)

# Create prediction dataframe
pred_df = df.iloc[time_steps:].copy().reset_index(drop=True)
pred_df['predicted_regime'] = predictions
pred_df['prediction_confidence'] = [max(probs) for probs in probabilities_list]

# Add probability columns for each regime
for regime_name, class_id in regime_map.items():
    pred_df[f'prob_{regime_name}'] = [probs[int(class_id)] for probs in probabilities_list]

st.success(f"Generated {len(predictions):,} predictions")

# -------------------------------------------------------------------
# Visualization
# -------------------------------------------------------------------
st.subheader("Price and Regime Predictions")

# Find price column
price_cols = [c for c in df.columns if 'close' in c.lower() or 'price' in c.lower()]
if not price_cols:
    st.error("No price column found. Expected 'close' or 'price' in column name.")
    st.stop()

price_col = price_cols[0]  # Use first available price column

# Create color map for regimes
unique_regimes = sorted(pred_df['predicted_regime'].unique())
colors = plt.cm.tab10(np.linspace(0, 1, len(unique_regimes)))
regime_colors = {regime: colors[i] for i, regime in enumerate(unique_regimes)}

# Main plot: Price with regime background
fig, ax = plt.subplots(figsize=(16, 8))

# Sort by timestamp to ensure correct order
pred_df_sorted = pred_df.sort_values('timestamp').reset_index(drop=True)

# Find continuous regime segments (where regime changes)
regime_changes = (pred_df_sorted['predicted_regime'] != pred_df_sorted['predicted_regime'].shift()).cumsum()

# Get y-axis limits first (will be set by price plot)
y_min = pred_df_sorted[price_col].min() * 0.98
y_max = pred_df_sorted[price_col].max() * 1.02

# Plot regime backgrounds first (so price line appears on top)
for segment_id in regime_changes.unique():
    segment_mask = regime_changes == segment_id
    segment_df = pred_df_sorted[segment_mask]
    
    if segment_df.empty:
        continue
    
    regime = segment_df['predicted_regime'].iloc[0]
    color = regime_colors.get(regime, 'gray')
    
    # Get the time range for this continuous segment
    start_time = segment_df['timestamp'].iloc[0]
    end_time = segment_df['timestamp'].iloc[-1]
    
    # Fill background for this regime segment (full height)
    ax.axvspan(start_time, end_time, 
              alpha=0.2, color=color, zorder=0)

# Plot price line on top
ax.plot(pred_df_sorted['timestamp'], pred_df_sorted[price_col], 
        color='black', linewidth=2, label='Price', alpha=0.9, zorder=3)

# Create legend
from matplotlib.patches import Patch
legend_elements = [plt.Line2D([0], [0], color='black', linewidth=2, label='Price')]
for regime in unique_regimes:
    legend_elements.append(
        Patch(facecolor=regime_colors[regime], alpha=0.3, label=f'Regime: {regime}')
    )

ax.set_xlabel('Time', fontsize=12)
ax.set_ylabel('Price', fontsize=12)
ax.set_title('Price with Predicted Regime Background', fontsize=14, fontweight='bold')
ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
ax.grid(True, alpha=0.3, zorder=1)

# Format x-axis dates
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
plt.xticks(rotation=45)

plt.tight_layout()
st.pyplot(fig)

# -------------------------------------------------------------------
# Regime Distribution
# -------------------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Regime Distribution")
    regime_counts = pred_df['predicted_regime'].value_counts()
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    colors_list = [regime_colors.get(regime, 'gray') for regime in regime_counts.index]
    ax2.pie(regime_counts.values, labels=regime_counts.index, autopct='%1.1f%%', 
            colors=colors_list, startangle=90)
    ax2.set_title('Distribution of Predicted Regimes', fontweight='bold')
    st.pyplot(fig2)

with col2:
    st.subheader("Average Confidence by Regime")
    avg_confidence = pred_df.groupby('predicted_regime')['prediction_confidence'].mean().sort_values(ascending=False)
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    colors_list = [regime_colors.get(regime, 'gray') for regime in avg_confidence.index]
    ax3.barh(avg_confidence.index, avg_confidence.values, color=colors_list)
    ax3.set_xlabel('Average Confidence', fontsize=11)
    ax3.set_title('Average Prediction Confidence by Regime', fontweight='bold')
    ax3.set_xlim(0, 1)
    st.pyplot(fig3)

# -------------------------------------------------------------------
# Probability Visualization (if enabled)
# -------------------------------------------------------------------
if show_probabilities:
    st.subheader("Regime Probability Over Time")
    fig4, ax4 = plt.subplots(figsize=(16, 6))
    
    for regime_name in regime_map.keys():
        if f'prob_{regime_name}' in pred_df.columns:
            ax4.plot(pred_df['timestamp'], pred_df[f'prob_{regime_name}'], 
                    label=regime_name, alpha=0.7, linewidth=1.5)
    
    ax4.set_xlabel('Time', fontsize=12)
    ax4.set_ylabel('Probability', fontsize=12)
    ax4.set_title('Regime Probability Distribution Over Time', fontsize=14, fontweight='bold')
    ax4.legend(loc='upper left', fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 1)
    
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax4.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    st.pyplot(fig4)

# -------------------------------------------------------------------
# Data Table
# -------------------------------------------------------------------
with st.expander("View Prediction Data"):
    display_cols = ['timestamp', price_col, 'predicted_regime', 'prediction_confidence']
    if show_probabilities:
        prob_cols = [f'prob_{regime}' for regime in regime_map.keys() if f'prob_{regime}' in pred_df.columns]
        display_cols.extend(prob_cols)
    
    st.dataframe(pred_df[display_cols].tail(100), use_container_width=True)
    
    # Download button
    csv = pred_df[display_cols].to_csv(index=False)
    st.download_button(
        label="Download predictions as CSV",
        data=csv,
        file_name=f"predictions_{selected_data_file}",
        mime="text/csv"
    )
