# 加密货币交易环境

用于 stable-baselines3 的 PPO agent 训练的 Gymnasium 环境。

## 文件结构

```
trading_env/
├── __init__.py              # 模块初始化
├── config.py                # 所有超参数配置
├── reward.py                # 奖励函数
├── trading_env.py           # 主环境类
├── test_env.py              # 测试脚本
├── requirements.txt         # 依赖包
└── README.md               # 本文件
```

## 安装依赖

```bash
pip install -r trading_env/requirements.txt
```

## 快速开始

### 1. 准备数据

数据格式要求：
- DataFrame，包含列：`['open', 'high', 'low', 'close', 'volume', 'regime_probs']`
- `regime_probs` 列存储 list 或 numpy array，长度为 `NUM_REGIMES`（默认 3）

示例：
```python
import pandas as pd
import numpy as np

# 创建示例数据
data = pd.DataFrame({
    'open': [...],
    'high': [...],
    'low': [...],
    'close': [...],
    'volume': [...],
    'regime_probs': [
        [0.3, 0.5, 0.2],  # 每个时间步的 regime 概率
        [0.4, 0.4, 0.2],
        ...
    ]
})
```

### 2. 创建环境

```python
from trading_env import CryptoTradingEnv

env = CryptoTradingEnv(data)
```

### 3. 使用环境

```python
# 重置环境
obs, info = env.reset()

# 执行动作
action = env.action_space.sample()  # 0=Hold, 1=Buy, 2=Sell
obs, reward, terminated, truncated, info = env.step(action)

# 渲染（可选）
env.render()
```

### 4. 与 stable-baselines3 集成

```python
from stable_baselines3 import PPO
from trading_env import CryptoTradingEnv

# 创建环境
env = CryptoTradingEnv(data)

# 训练 PPO agent
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

# 使用训练好的模型
obs, info = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated:
        break
```

## 配置参数

在 `config.py` 中可以修改以下参数：

- `INITIAL_CAPITAL`: 初始资金（默认 10000）
- `TRANSACTION_FEE_RATE`: 手续费率（默认 0.001，即 0.1%）
- `MAX_POSITION_RATIO`: 单次最多用多少比例的资金买入（默认 0.5）
- `LOOKBACK_WINDOW`: 状态里看回多少根 K 线（默认 20）
- `NUM_REGIMES`: regime 数量（默认 3）
- `REWARD_LAMBDA`: reward 里 drawdown 惩罚的权重（默认 0.5）

## 状态空间

状态是一个 1D numpy array，由以下部分拼接：

1. **价格特征**（`LOOKBACK_WINDOW × 5`）：
   - 过去 `LOOKBACK_WINDOW` 根 K 线的 open, high, low, close, volume
   - 经过 min-max normalization 归一化到 [0, 1]

2. **当前仓位信息**（2 维）：
   - 持仓比例（0~1）
   - 当前回报率（相对于初始资金）

3. **Regime 概率向量**（`NUM_REGIMES` 维）：
   - classifier 输出的 regime 概率

总维度：`LOOKBACK_WINDOW × 5 + 2 + NUM_REGIMES`（默认 105）

## 动作空间

- `0`: Hold（持仓不动）
- `1`: Buy（买入，金额 = 可用资金 × MAX_POSITION_RATIO）
- `2`: Sell（卖出全部仓位）

## 奖励函数

```
reward = portfolio_return - REWARD_LAMBDA × max_drawdown_in_step
```

- `portfolio_return`: 这一步相对于上一步的资组回报率
- `max_drawdown_in_step`: 这一步内从峰值到当前的回撤

## 测试

运行测试脚本验证环境：

```bash
python trading_env/test_env.py
```

## 注意事项

1. 买入和卖出时都会扣除手续费
2. 买入时如果现金不够买一个最小单位，默认转为 Hold
3. 所有 float 精度使用 float64
4. 环境按时间顺序遍历数据，不做 shuffle
5. 到最后一根 K 线时 `terminated = True`
