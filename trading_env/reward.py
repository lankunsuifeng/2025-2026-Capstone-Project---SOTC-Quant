"""
奖励函数模块
实现 portfolio_return - REWARD_LAMBDA × max_drawdown_in_step
"""

import numpy as np
from .config import REWARD_LAMBDA, DTYPE


def calculate_reward(
    portfolio_value_current: float,
    portfolio_value_previous: float,
    portfolio_value_peak: float
) -> float:
    """
    计算单步奖励
    
    Args:
        portfolio_value_current: 当前资产总值
        portfolio_value_previous: 上一步资产总值
        portfolio_value_peak: 这一步内从开始到当前的峰值资产
        
    Returns:
        reward: portfolio_return - REWARD_LAMBDA × max_drawdown_in_step
    """
    # 计算这一步相对于上一步的资组回报率
    if portfolio_value_previous <= 0:
        portfolio_return = 0.0
    else:
        portfolio_return = (portfolio_value_current - portfolio_value_previous) / portfolio_value_previous
    
    # 计算这一步内从峰值到当前的回撤
    if portfolio_value_peak <= 0:
        max_drawdown_in_step = 0.0
    else:
        drawdown = (portfolio_value_peak - portfolio_value_current) / portfolio_value_peak
        max_drawdown_in_step = max(0.0, drawdown)  # 确保非负
    
    # 计算最终奖励
    reward = portfolio_return - REWARD_LAMBDA * max_drawdown_in_step
    
    return np.float64(reward)
