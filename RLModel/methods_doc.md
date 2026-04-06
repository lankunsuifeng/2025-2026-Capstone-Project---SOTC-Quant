# Pipeline Methods: HMM-LSTM Regime Classification + RL Trading

## Overview

This project builds a cryptocurrency trading system for BTC/USDT on 5-minute bars (Feb 2021 -- Feb 2026). The pipeline has three stages: (1) Hidden Markov Model (HMM) for market regime identification, (2) LSTM for next-bar regime prediction, and (3) Reinforcement Learning (RL) agents that trade conditioned on regime signals. A calendar-based split ensures no data leakage across stages.

```
Raw 5m OHLCV + Indicators
        │
        ▼
  ┌────────────┐
  │  Gaussian   │  Unsupervised regime labeling
  │  HMM (K=4) │  on technical features
  └─────┬──────┘
        │ regime labels
        ▼
  ┌────────────┐
  │  LSTM       │  Supervised: predict next-bar
  │  Classifier │  regime from feature sequences
  └─────┬──────┘
        │ predicted regime (one-hot)
        ▼
  ┌────────────┐
  │  RL Agents  │  PPO / Hard Mixture-of-Experts
  │  (Trading)  │  conditioned on regime signals
  └────────────┘
```

---

## Stage 1: HMM Regime Identification

**Model**: Gaussian HMM with diagonal covariance (`hmmlearn`), K = 4 states.

**Input features**: A subset of technical indicators computed on 5-minute bars, including ATR (normalized), Bollinger Band width, ADX, volume z-score, and log-return. Raw OHLCV prices are excluded from HMM fitting.

**Preprocessing**: Features are standardized with StandardScaler, optionally followed by PCA (default: min(4, n_features) components). Both scaler and PCA are fit only on the training window (data up to `hmm_fit_end`) and applied to the full series to prevent look-ahead bias.

**State assignment**: The fitted HMM assigns a discrete state ID (0--3) to every bar via the Viterbi algorithm (`model.predict`). States are mapped to interpretable regime labels based on volatility and trend characteristics computed on the training window only.

**Code**: `HMMLSTM/cryptoregimeclassifier/src/regime_label.py`

---

## Stage 2: LSTM Next-Bar Regime Prediction

**Task**: Multi-class classification -- predict the HMM regime of the *next* bar given a sliding window of past features.

**Architecture**: Keras Sequential model:
- Input shape: `(time_steps, n_features)` with `time_steps = 32`
- LSTM layer (single layer, `return_sequences=False`) → Dropout → Dense (ReLU) → Dropout → Dense (softmax, K=4 classes)

**Input features**: All engineered technical indicators (same as HMM input plus additional ones); raw OHLCV columns are excluded. Features are standardized with StandardScaler fit on the LSTM training set only.

**Label**: The HMM-assigned regime at bar `t + time_steps` serves as the supervised target for the sequence ending at bar `t + time_steps - 1`. This means the LSTM learns to forecast one bar ahead of the observation window.

**Training**: Categorical cross-entropy loss; class weights balanced by inverse frequency; EarlyStopping and ReduceLROnPlateau callbacks; 15% validation split within the training period.

**Output**: For each bar, the LSTM produces a softmax probability vector over K regimes. The argmax class is taken as the predicted regime. Both the HMM-assigned regime and the LSTM-predicted regime are exported as one-hot encoded columns (e.g., `hmm_predicted_state_0..3`, `lstm_predicted_state_0..3`).

**Code**: `HMMLSTM/cryptoregimeclassifier/src/lstm_model.py`

---

## Stage 3: Feature Engineering for RL

The combined output from Stage 1--2 is further processed into a flat feature matrix for RL (`data_process.data_engineering`).

**Observation features** (per bar):

| Category | Features |
|----------|----------|
| Returns | `log_ret_1` (1-bar log return) |
| Trend | `ema_ratio_9_21`, `macd_hist`, `adx` |
| Volatility | `atr_norm`, `bb_width` |
| Momentum | `rsi_14` |
| Volume | `volume_zscore_50` |
| Intrabar | `co_ret` (close/open - 1), `hl_range_norm`, `log_hl` |
| Regime (optional) | `hmm_predicted_state_0..3`, `lstm_predicted_state_0..3` (one-hot) |

All features are relative or normalized -- no raw price levels enter the observation space. An optional RobustScaler (fit on RL training rows only) and Winsorization (1st/99th percentile on training rows) can be applied.

**Position augmentation**: The agent's current position is appended to the observation as a 3-dimensional one-hot vector `[is_short, is_cash, is_long]`, making the environment fully observable (position ∈ {-1, 0, +1}).

**Bar resampling**: 5-minute bars can be aggregated to 15-minute bars before RL training. Log-returns are summed; volume z-scores are averaged; high-low range features take the max; all other indicators take the last value in each 15-minute window. This improves the signal-to-noise ratio per decision step.

---

## Stage 4: RL Trading Agents

### Environment

**`TradingEnv`**: A discrete-action, single-asset trading environment.

- **State**: Feature vector (technical indicators + optional regime one-hots + position one-hot)
- **Action space**: {-1 (short), 0 (cash), +1 (long)} -- the agent selects a target position each step
- **Reward** (per step):

`r_t = reward_scale * ( a_t * R_t  -  fee * |delta_a_t|  -  lambda_turnover * |delta_a_t| )`

where `a_t` is the position chosen, `R_t` is the 1-bar simple return, fee = 5 bps per unit turnover, `lambda_turnover` is an optional turnover penalty coefficient, and reward_scale = 1000 (amplifies the ~1e-4 magnitude of raw returns for stable gradient signals). `|delta_a_t|` denotes the absolute change in position (0, 1, or 2).

- **Episode**: Random-start segments of 10,000 steps during training; full test set during evaluation

### PPO Agent

**Algorithm**: Proximal Policy Optimization (PPO) with clipped surrogate objective.

**Network** (`ActorCritic`): Shared backbone of 2 fully-connected layers (128 units each) with LayerNorm and ReLU, followed by separate linear heads for the policy (3-way softmax) and value function (scalar).

**PPO loss**:

`L = L_clip + 0.5 * L_value - beta * H(pi)`

where `L_clip` is the clipped surrogate objective (clip ratio epsilon = 0.2), `L_value` is the squared value error, and `H(pi)` is the entropy bonus with coefficient beta (default 0.1).

**Hyperparameters**: gamma = 0.99, lambda_GAE = 0.95, learning rate 3e-4 (Adam), rollout length 2048 steps, minibatch size 256, 10 epochs per update, gradient clipping at 0.5.

### Experiment Variants (PPO)

| Variant | Observation includes |
|---------|---------------------|
| `ppo_base` | Technical features only |
| `ppo_hmm` | Technical + HMM predicted regime (one-hot) |
| `ppo_hmm_lstm` | Technical + HMM + LSTM predicted regime (one-hot) |

### Hard Mixture-of-Experts (MoE) Agent

**Architecture**: K = 4 independent `ActorCritic` networks (same architecture as PPO), one per HMM regime.

**Routing**: At each step, the environment provides the current regime ID via the HMM (or LSTM) one-hot columns. The corresponding expert is selected deterministically -- only that expert's policy generates the action and only that expert receives the gradient update.

**Training**: Each expert is trained with the same PPO objective, but only on transitions belonging to its regime. During each minibatch, transitions are partitioned by `regime_id` and dispatched to the respective expert. GAE advantages and returns are computed over the full rollout (shared value targets), then sliced per regime for the policy update.

| Variant | Routing signal |
|---------|---------------|
| `moe_hmm` | HMM predicted regime (argmax of one-hot) |
| `moe_hmm_lstm` | LSTM predicted regime (argmax of one-hot) |

In the LSTM-routed variant (`moe_hmm_lstm`), HMM one-hot features are still included in the observation; only the routing decision uses the LSTM prediction.

---

## Calendar Split

A single `split_config.json` governs all stages to avoid data leakage:

| Boundary | Date | Purpose |
|----------|------|---------|
| `hmm_fit_end` | 2023-02-01 | HMM + PCA/Scaler fit only on data before this |
| `lstm_train_end` | 2023-02-01 | LSTM supervised training ends here |
| `rl_train_end` | 2025-02-01 | RL agents train on data after `lstm_train_end` up to here |
| `test_start` | 2025-02-01 | Out-of-sample evaluation begins (Feb 2025 -- Feb 2026) |

The RL training window strictly begins after the LSTM training cutoff, ensuring the regime predictions used during RL training are out-of-sample with respect to the LSTM model.

---

## Evaluation

**Backtest**: The trained agent is evaluated on the held-out test set (Feb 2025 -- Feb 2026) with deterministic action selection (argmax policy). Transaction fees of 5 bps per unit turnover are applied in the backtest environment.

**Metrics**:

- **Return**: Total return over the test period
- **Annualized Sharpe Ratio**: `(mean_bar_return / std_bar_return) * sqrt(M)` where M = bars per year (35,064 for 15-min bars, 7x24 crypto)
- **Calmar Ratio**: Return / |Max Drawdown|
- **Max Drawdown (MDD)**: Largest peak-to-trough decline in equity
- **Turnover Rate**: Average position changes per bar

**Baseline**: Buy-and-hold over the same test period.
