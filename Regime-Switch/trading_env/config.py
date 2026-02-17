"""
交易环境配置文件
集中管理所有超参数，方便后续调优
"""

# 资金相关
INITIAL_CAPITAL = 10000.0  # 初始资金（美元）
TRANSACTION_FEE_RATE = 0.001  # 手续费率（0.1%）
MAX_POSITION_RATIO = 0.5  # 单次最多用多少比例的资金买入

# 状态空间相关
LOOKBACK_WINDOW = 20  # 状态里看回多少根 K 线

# Regime 相关
NUM_REGIMES = 3  # regime 数量

# Reward 相关
REWARD_LAMBDA = 0.5  # reward 里 drawdown 惩罚的权重

# 数据类型
DTYPE = 'float64'  # 所有 float 精度用 float64
