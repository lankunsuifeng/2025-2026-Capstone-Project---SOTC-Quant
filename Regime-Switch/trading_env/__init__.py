"""
加密货币交易环境模块
"""

from .trading_env import CryptoTradingEnv
from .config import (
    INITIAL_CAPITAL,
    TRANSACTION_FEE_RATE,
    MAX_POSITION_RATIO,
    LOOKBACK_WINDOW,
    NUM_REGIMES,
    REWARD_LAMBDA,
    DTYPE
)

__all__ = [
    'CryptoTradingEnv',
    'INITIAL_CAPITAL',
    'TRANSACTION_FEE_RATE',
    'MAX_POSITION_RATIO',
    'LOOKBACK_WINDOW',
    'NUM_REGIMES',
    'REWARD_LAMBDA',
    'DTYPE'
]
