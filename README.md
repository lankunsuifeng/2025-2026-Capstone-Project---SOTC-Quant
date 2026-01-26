# Capstone Project for SOTC-Quant

## Introduction

## Main Work

## Feature Engineering

### MVP Feature Group: Classic Technical Indicators

本文档说明 MVP 特征工程中的经典技术指标特征集。所有特征都是严格向后看的、连续值的，适合强化学习使用。

#### 特征列表

**1. Simple Moving Averages (SMA)**
- **SMA20**: 20期简单移动平均
- **SMA50**: 50期简单移动平均
- **price_deviation_sma20**: `(close - SMA20) / SMA20` - 价格相对SMA20的偏离
- **price_deviation_sma50**: `(close - SMA50) / SMA50` - 价格相对SMA50的偏离
- **说明**: 移动平均线用于识别趋势方向，价格偏离可用于判断超买超卖。

**2. Exponential Moving Averages (EMA)**
- **EMA10**: 10期指数移动平均
- **EMA20**: 20期指数移动平均
- **EMA50**: 50期指数移动平均
- **ema_slope_10**: `EMA10_t - EMA10_{t-1}` - EMA10的变化量
- **ema_slope_20**: `EMA20_t - EMA20_{t-1}` - EMA20的变化量
- **ema_slope_50**: `EMA50_t - EMA50_{t-1}` - EMA50的变化量
- **说明**: EMA对价格变化更敏感，斜率可用于衡量趋势加速度。

**3. MACD (Moving Average Convergence Divergence)**
- **macd_line**: `EMA12 - EMA26` - MACD线
- **macd_signal**: `EMA9 of MACD` - 信号线（MACD的9期EMA）
- **macd_histogram**: `MACD - Signal` - MACD柱状图
- **说明**: MACD用于识别趋势变化和动量，柱状图显示MACD与信号线的差异。

**4. RSI (Relative Strength Index)**
- **rsi_14**: 14期相对强弱指标（使用Wilder平滑方法）
- **公式**: 
  ```
  delta = close.diff()
  gain = delta.clip(lower=0.0)
  loss = (-delta).clip(lower=0.0)
  avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
  avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
  rs = avg_gain / (avg_loss + eps)
  rsi = 100 - (100 / (1 + rs))
  ```
- **说明**: RSI范围0-100，>70表示超买，<30表示超卖。

**5. Rolling Volatility**
- **rolling_volatility_20**: 20期滚动波动率
- **公式**: `std(log(close_t / close_{t-1}))` over 20-period window
- **说明**: 对数收益率的滚动标准差，衡量价格波动程度。

**6. Bollinger Bands**
- **bb_width**: 布林带带宽 `(upper_band - lower_band) / middle_band`
- **bb_price_position**: 价格在带内的位置 `(close - lower_band) / (upper_band - lower_band)`
- **参数**: window=20, k=2（2倍标准差）
- **说明**: 带宽衡量波动性，价格位置（0-1）表示价格在带内的相对位置。

**7. Volume Moving Average**
- **volume_ma_20**: 20期成交量移动平均
- **volume_ratio**: `volume / volume_ma_20` - 成交量相对均值的比例
- **说明**: 用于识别异常成交量，volume_ratio > 1表示成交量高于平均水平。

**8. On-Balance Volume (OBV)**
- **obv**: 能量潮指标
- **计算逻辑**: 
  - 如果 close_t > close_{t-1}: OBV_t = OBV_{t-1} + volume_t
  - 如果 close_t < close_{t-1}: OBV_t = OBV_{t-1} - volume_t
  - 如果 close_t = close_{t-1}: OBV_t = OBV_{t-1}
- **说明**: OBV用于识别资金流向，OBV上升表示资金流入。

**9. VWAP (Volume Weighted Average Price)**
- **vwap**: 成交量加权平均价格（20期滚动窗口）
- **vwap_deviation**: `(close - VWAP) / VWAP` - 价格相对VWAP的偏离
- **计算**: `VWAP = sum(typical_price * volume) / sum(volume)` where `typical_price = (high + low + close) / 3`
- **说明**: VWAP是机构常用的参考价格，价格偏离VWAP可能表示市场情绪变化。

**10. Regime-oriented Features (市场状态分类特征)**
- **trend_sign_fast**: 短期趋势方向 `sign(EMA_10 - EMA_20)`，值在 {-1, 0, +1}
- **trend_sign_medium**: 中期趋势方向 `sign(EMA_20 - EMA_50)`，值在 {-1, 0, +1}
- **trend_strength**: 趋势强度（无标度）`|EMA_20 - EMA_50| / close_price`，连续非负值
- **vol_zscore**: 波动率相对水平 `(vol - rolling_mean(vol)) / rolling_std(vol)`，使用100期窗口计算波动率的滚动z-score
- **delta_vol**: 波动率变化 `rolling_vol_t - rolling_vol_{t-1}`，捕获市场风险的变化
- **说明**: 这些特征专门设计用于市场状态分类器，强调稳定性、可解释性和状态感知能力。不包含硬交易规则，仅作为状态输入。

#### 技术细节

**数据要求**
- 输入数据必须包含以下列：`open`, `high`, `low`, `close`, `volume`
- 数据应按时间顺序排列
- 时间索引应为 datetime 类型

**窗口参数**
- SMA: 20, 50
- EMA: 10, 20, 50
- MACD: fast=12, slow=26, signal=9
- RSI: window=14
- Rolling Volatility: window=20
- Bollinger Bands: window=20, k=2
- Volume MA: window=20
- VWAP: window=20
- Volatility Z-score: window=100 (用于计算波动率的滚动均值和标准差)

**注意事项**
1. 所有特征计算都是**严格向后看**的，避免前瞻偏差
2. 使用 `min_periods` 参数确保在窗口不足时返回 NaN
3. 在除法运算中添加小常数（epsilon=1e-10）避免除零错误
4. RSI使用Wilder平滑方法（RMA），alpha=1/window
5. OBV使用累积计算，初始值为第一个成交量

**特征用途**
这些特征可用于：
- 强化学习环境的状态空间
- 机器学习模型的输入特征
- 技术分析和量化策略的信号生成
- 市场状态分类和模式识别

#### 使用方法

```python
from mvp_feature import load_data, create_technical_indicators, get_feature_columns

# 加载数据
df = load_data('./data/BTCUSDT/5m/klines_2025_01.csv')

# 创建技术指标特征
df = create_technical_indicators(df)

# 获取所有特征列
feature_cols = get_feature_columns(df)

# 访问特征
features = df[feature_cols]
```

## How to use

## Requirements
