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
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, r2_score,
    mean_squared_error, mean_absolute_error
)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import seaborn as sns

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
# Model Evaluation Metrics
# -------------------------------------------------------------------
st.subheader("Model Evaluation Metrics")

# Check if ground truth labels exist in original df or pred_df
# Try multiple possible column names
has_ground_truth = False
ground_truth_col = None
possible_cols = ['regime', 'state', 'target', 'true_regime', 'label', 'regime_label']

# First check in pred_df (which contains data from time_steps onwards)
for col in possible_cols:
    if col in pred_df.columns:
        # Check if column has non-null values
        if pred_df[col].notna().sum() > 0:
            ground_truth_col = col
            has_ground_truth = True
            break

# If not found in pred_df, check in original df
# Model predicts next bar's regime, so pred_df[i] corresponds to df[time_steps+i]
if not has_ground_truth:
    for col in possible_cols:
        if col in df.columns:
            if df[col].notna().sum() > 0:
                ground_truth_col = col
                has_ground_truth = True
                # Align: pred_df rows correspond to df[time_steps:] rows
                # The prediction at pred_df[i] is for the regime at df[time_steps+i]
                # Ensure length matches
                df_slice = df[col].iloc[time_steps:].reset_index(drop=True)
                if len(df_slice) == len(pred_df):
                    pred_df[col] = df_slice.values
                else:
                    # If lengths don't match, try to align by index
                    pred_df[col] = None
                    # Try to match by original index
                    for idx in pred_df.index:
                        orig_idx = time_steps + idx
                        if orig_idx < len(df):
                            pred_df.loc[idx, col] = df.loc[orig_idx, col]
                break

# Debug: Show available columns
with st.expander("🔍 Debug: Available Columns", expanded=False):
    st.write("**Original df columns:**", list(df.columns))
    st.write("**Pred_df columns:**", list(pred_df.columns))
    st.write("**Columns checked for ground truth:**", possible_cols)
    if ground_truth_col:
        st.success(f"✅ Found ground truth column: '{ground_truth_col}'")
        non_null_count = pred_df[ground_truth_col].notna().sum()
        st.write(f"Non-null values: {non_null_count} / {len(pred_df)}")
        if non_null_count > 0:
            st.write(f"**Sample values:** {pred_df[ground_truth_col].dropna().head(10).tolist()}")
            st.write(f"**Unique values:** {pred_df[ground_truth_col].dropna().unique()[:10]}")
    else:
        st.warning("⚠️ No ground truth column found.")
        st.info("💡 **To see R² and other metrics, ensure your data file contains one of these columns:**")
        st.code("'regime', 'state', 'target', 'true_regime', 'label', 'regime_label'")
        # Show columns that might be relevant
        relevant_cols = [c for c in df.columns if any(keyword in c.lower() for keyword in ['regime', 'state', 'label', 'target'])]
        if relevant_cols:
            st.write("**Potentially relevant columns found:**", relevant_cols)

if has_ground_truth:
    # Prepare ground truth and predictions for evaluation
    y_true = pred_df[ground_truth_col].dropna()
    y_pred = pred_df['predicted_regime']
    
    # Align indices
    common_idx = y_true.index.intersection(y_pred.index)
    y_true_aligned = y_true.loc[common_idx]
    y_pred_aligned = y_pred.loc[common_idx]
    
    if len(y_true_aligned) > 0:
        # Encode labels to numeric for metrics calculation
        le = LabelEncoder()
        all_labels = pd.concat([y_true_aligned, y_pred_aligned]).unique()
        le.fit(all_labels)
        
        y_true_encoded = le.transform(y_true_aligned)
        y_pred_encoded = le.transform(y_pred_aligned)
        
        # Classification Metrics
        accuracy = accuracy_score(y_true_encoded, y_pred_encoded)
        precision = precision_score(y_true_encoded, y_pred_encoded, average='weighted', zero_division=0)
        recall = recall_score(y_true_encoded, y_pred_encoded, average='weighted', zero_division=0)
        f1 = f1_score(y_true_encoded, y_pred_encoded, average='weighted', zero_division=0)
        
        # Calculate R² (coefficient of determination) for classification
        # Using one-hot encoding approach
        ohe = OneHotEncoder(sparse_output=False)
        y_true_onehot = ohe.fit_transform(y_true_encoded.reshape(-1, 1))
        
        # Convert predictions to one-hot
        y_pred_onehot = np.zeros_like(y_true_onehot)
        y_pred_onehot[np.arange(len(y_pred_encoded)), y_pred_encoded] = 1
        
        # Calculate R² for each class and average
        r2_scores = []
        for i in range(y_true_onehot.shape[1]):
            r2 = r2_score(y_true_onehot[:, i], y_pred_onehot[:, i])
            r2_scores.append(r2)
        r2_mean = np.mean(r2_scores)
        
        # Calculate MSE and MAE (using encoded labels)
        mse = mean_squared_error(y_true_encoded, y_pred_encoded)
        mae = mean_absolute_error(y_true_encoded, y_pred_encoded)
        
        # Brier Score (for probability calibration)
        # Get true labels as one-hot
        true_probs = y_true_onehot
        
        # Get predicted probabilities aligned with common_idx
        # pred_df was created from df.iloc[time_steps:], so indices are offset
        # We need to map common_idx to the position in pred_df
        pred_probs_list = []
        for idx in common_idx:
            if idx in pred_df.index:
                # Find the position in pred_df
                pos_in_pred_df = list(pred_df.index).index(idx)
                if pos_in_pred_df < len(probabilities_list):
                    pred_probs_list.append(probabilities_list[pos_in_pred_df])
        
        if len(pred_probs_list) == len(true_probs):
            pred_probs = np.array(pred_probs_list)
            
            # Align regime order with LabelEncoder classes
            # le.classes_ contains the regime names in the order they were encoded
            # regime_map maps regime names to model class indices
            pred_probs_aligned = np.zeros((len(pred_probs), len(le.classes_)))
            for i, regime_name in enumerate(le.classes_):
                # Find corresponding class index in regime_map
                if regime_name in regime_map:
                    class_idx = int(regime_map[regime_name])
                    if class_idx < pred_probs.shape[1]:
                        pred_probs_aligned[:, i] = pred_probs[:, class_idx]
            
            # Calculate Brier Score
            brier_score = np.mean(np.sum((pred_probs_aligned - true_probs) ** 2, axis=1))
        else:
            # Fallback: skip Brier Score if alignment fails
            brier_score = np.nan
        
        # Display metrics in columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Accuracy", f"{accuracy:.4f}")
            st.metric("Precision (Weighted)", f"{precision:.4f}")
        
        with col2:
            st.metric("Recall (Weighted)", f"{recall:.4f}")
            st.metric("F1 Score (Weighted)", f"{f1:.4f}")
        
        with col3:
            st.metric("R² Score (Mean)", f"{r2_mean:.4f}")
            if not np.isnan(brier_score):
                st.metric("Brier Score", f"{brier_score:.4f}")
            else:
                st.metric("Brier Score", "N/A")
        
        # Additional metrics
        st.markdown("---")
        col4, col5 = st.columns(2)
        
        with col4:
            st.metric("Mean Squared Error (MSE)", f"{mse:.4f}")
        
        with col5:
            st.metric("Mean Absolute Error (MAE)", f"{mae:.4f}")
        
        # Confusion Matrix
        st.markdown("---")
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_true_encoded, y_pred_encoded, labels=range(len(le.classes_)))
        
        fig_cm, ax_cm = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                   xticklabels=le.classes_, yticklabels=le.classes_, ax=ax_cm)
        ax_cm.set_xlabel("Predicted Regime", fontsize=12)
        ax_cm.set_ylabel("True Regime", fontsize=12)
        ax_cm.set_title("Confusion Matrix", fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        st.pyplot(fig_cm)
        
        # Classification Report
        st.markdown("---")
        st.subheader("Detailed Classification Report")
        report = classification_report(y_true_encoded, y_pred_encoded, 
                                      target_names=le.classes_, 
                                      output_dict=True, 
                                      zero_division=0)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df, use_container_width=True)
        
    else:
        st.warning(f"Ground truth column '{ground_truth_col}' found but no overlapping data with predictions.")
else:
    st.warning("⚠️ No ground truth labels found in data. Cannot calculate R², Accuracy, etc.")
    st.info("💡 **Tip:** To see evaluation metrics, use a data file with 'regime', 'state', 'target', or 'label' column.")
    
    # Calculate prediction statistics even without ground truth
    st.markdown("---")
    st.subheader("Prediction Statistics (No Ground Truth Available)")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Predictions", f"{len(predictions):,}")
        st.metric("Unique Regimes Predicted", f"{pred_df['predicted_regime'].nunique()}")
        # Prediction stability: how often regime changes
        regime_changes = (pred_df['predicted_regime'] != pred_df['predicted_regime'].shift()).sum()
        stability_ratio = 1 - (regime_changes / len(pred_df))
        st.metric("Prediction Stability", f"{stability_ratio:.4f}")
    
    with col2:
        avg_confidence = pred_df['prediction_confidence'].mean()
        st.metric("Average Confidence", f"{avg_confidence:.4f}")
        median_confidence = pred_df['prediction_confidence'].median()
        st.metric("Median Confidence", f"{median_confidence:.4f}")
        std_confidence = pred_df['prediction_confidence'].std()
        st.metric("Confidence Std Dev", f"{std_confidence:.4f}")
    
    with col3:
        min_confidence = pred_df['prediction_confidence'].min()
        st.metric("Min Confidence", f"{min_confidence:.4f}")
        max_confidence = pred_df['prediction_confidence'].max()
        st.metric("Max Confidence", f"{max_confidence:.4f}")
        # Calculate entropy of predictions (uncertainty measure)
        regime_counts = pred_df['predicted_regime'].value_counts()
        probs = regime_counts / len(pred_df)
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        normalized_entropy = entropy / np.log(len(regime_counts)) if len(regime_counts) > 1 else 0
        st.metric("Normalized Entropy", f"{normalized_entropy:.4f}")
    
    # Additional statistics
    st.markdown("---")
    st.subheader("Probability Distribution Statistics")
    
    # Calculate statistics for each regime's probability
    prob_stats = {}
    for regime_name in regime_map.keys():
        prob_col = f'prob_{regime_name}'
        if prob_col in pred_df.columns:
            prob_stats[regime_name] = {
                'mean': pred_df[prob_col].mean(),
                'std': pred_df[prob_col].std(),
                'min': pred_df[prob_col].min(),
                'max': pred_df[prob_col].max()
            }
    
    if prob_stats:
        prob_stats_df = pd.DataFrame(prob_stats).T
        st.dataframe(prob_stats_df, use_container_width=True)

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
