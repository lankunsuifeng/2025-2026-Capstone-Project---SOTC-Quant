# Capstone Project for SOTC-Quant

## Introduction

## Main Work

## Feature Engineering

### MVP Feature Group 1: Price Dynamics

本文档说明 MVP 特征工程中的价格动态特征组（Feature Group 1）。

#### 特征列表

**1. Log Return (1-bar)**
- **公式**: `log_return_1 = log(close_t / close_{t-1})`
- **说明**: 单期对数收益率，衡量相邻两个时间点的价格变化。

**2. Log Return (5-bar)**
- **公式**: `log_return_5 = log(close_t / close_{t-5})`
- **说明**: 5期对数收益率，衡量短期价格趋势。

**3. Log Return (20-bar)**
- **公式**: `log_return_20 = log(close_t / close_{t-20})`
- **说明**: 20期对数收益率，衡量中期价格趋势。

**4. Rolling Volatility**
- **公式**: 
  ```
  returns_t = log(close_t / close_{t-1})
  rolling_volatility = sqrt(mean(returns_{t-i}^2 for i = 0 to W-1))
  where W = 20
  ```
- **说明**: 滚动波动率，使用过去20期的收益率平方均值开方计算，衡量价格波动程度。

**5. High-Low Range**
- **公式**: `high_low_range = (high_t - low_t) / close_t`
- **说明**: 高低价范围相对于收盘价的比例，衡量单期内的价格波动幅度。

**6. Price vs EMA Deviation**
- **公式**: 
  ```
  ema_20 = exponential_moving_average(close, window = 20)
  price_ema_deviation = (close_t - ema_20) / ema_20
  ```
- **说明**: 收盘价相对于20期指数移动平均线的偏离程度，衡量价格是否偏离趋势。

**7. EMA Slope**
- **公式**: 
  ```
  ema_20 = exponential_moving_average(close, window = 20)
  ema_slope = ema_20_t - ema_20_{t-1}
  ```
- **说明**: EMA的变化量，衡量趋势的加速度或方向变化。

**8. VWAP Deviation**
- **公式**: `vwap_deviation = (close_t - vwap_t) / vwap_t`
- **说明**: 收盘价相对于成交量加权平均价格（VWAP）的偏离程度。VWAP使用过去20期的典型价格（(high + low + close) / 3）和成交量计算。

**9. Price Position in Recent Range**
- **公式**: 
  ```
  recent_low = min(low_{t-W} to low_t)
  recent_high = max(high_{t-W} to high_t)
  price_range_position = (close_t - recent_low) / (recent_high - recent_low)
  where W = 20
  ```
- **说明**: 当前收盘价在过去20期价格区间中的相对位置，范围在0到1之间。0表示接近最低价，1表示接近最高价。

**10. Trend vs Chop Indicator**
- **公式**: 
  ```
  ema_short = exponential_moving_average(close, window = 10)
  ema_long = exponential_moving_average(close, window = 30)
  returns_t = log(close_t / close_{t-1})
  volatility = sqrt(mean(returns_{t-i}^2 for i = 0 to W-1))
  where W = 20
  trend_strength = abs(ema_short - ema_long) / volatility
  ```
- **说明**: 趋势强度指标，衡量短期和长期EMA的差异相对于波动率的比例。值越大表示趋势越强，值越小表示市场处于震荡（chop）状态。

#### 技术细节

**数据要求**
- 输入数据必须包含以下列：`open`, `high`, `low`, `close`, `volume`
- 数据应按时间顺序排列
- 时间索引应为 datetime 类型

**窗口参数**
- Rolling Volatility: W = 20
- Price Position in Recent Range: W = 20
- EMA (短期): span = 10
- EMA (长期): span = 30
- VWAP: window = 20

**注意事项**
1. 所有特征计算都是**严格向后看**的，避免前瞻偏差
2. 使用 `min_periods` 参数确保在窗口不足时返回 NaN
3. 在除法运算中添加小常数（1e-10）避免除零错误
4. 对数收益率使用 `np.log()` 计算，比百分比收益率更适合金融建模

**特征用途**
这些特征可用于：
- 强化学习环境的状态空间
- 机器学习模型的输入特征
- 技术分析和量化策略的信号生成

#### 使用方法

```python
from mvp_feature import load_data, create_price_dynamics_features

# 加载数据
df = load_data('./data/BTCUSDT/5m/klines_2025_01.csv')

# 创建特征
df = create_price_dynamics_features(df)

# 访问特征
features = df[['log_return_1', 'log_return_5', 'log_return_20', 
               'rolling_volatility', 'high_low_range', 'price_ema_deviation',
               'ema_slope', 'vwap_deviation', 'price_range_position', 
               'trend_strength']]
```

## How to use

## Requirements
