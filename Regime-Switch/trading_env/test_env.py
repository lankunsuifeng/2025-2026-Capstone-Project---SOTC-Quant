"""
测试脚本：用随机动作验证环境能正常运行
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from trading_env.trading_env import CryptoTradingEnv
from trading_env.config import NUM_REGIMES, LOOKBACK_WINDOW


def generate_dummy_data(n_steps: int = 100) -> pd.DataFrame:
    """
    生成测试用的虚拟数据
    
    Args:
        n_steps: 数据步数
        
    Returns:
        data: DataFrame with required columns
    """
    np.random.seed(42)
    
    # 生成价格数据（随机游走）
    base_price = 50000.0
    returns = np.random.randn(n_steps) * 0.01
    prices = base_price * np.exp(np.cumsum(returns))
    
    # 生成 OHLCV 数据
    data = pd.DataFrame({
        'open': prices * (1 + np.random.randn(n_steps) * 0.001),
        'high': prices * (1 + np.abs(np.random.randn(n_steps)) * 0.002),
        'low': prices * (1 - np.abs(np.random.randn(n_steps)) * 0.002),
        'close': prices,
        'volume': np.random.uniform(100, 1000, n_steps),
    })
    
    # 确保 high >= close >= low, high >= open >= low
    data['high'] = np.maximum(data['high'], np.maximum(data['open'], data['close']))
    data['low'] = np.minimum(data['low'], np.minimum(data['open'], data['close']))
    
    # 生成 regime 概率向量（每个 regime 的概率和为 1）
    regime_probs = []
    for _ in range(n_steps):
        probs = np.random.dirichlet(np.ones(NUM_REGIMES))
        regime_probs.append(probs.tolist())
    
    data['regime_probs'] = regime_probs
    
    return data


def test_env():
    """测试环境"""
    print("=" * 60)
    print("测试加密货币交易环境")
    print("=" * 60)
    
    # 生成测试数据
    print("\n1. 生成测试数据...")
    data = generate_dummy_data(n_steps=200)
    print(f"   数据形状: {data.shape}")
    print(f"   数据列: {data.columns.tolist()}")
    print(data.head())
    
    # 创建环境
    print("\n2. 创建环境...")
    env = CryptoTradingEnv(data)
    print(f"   观察空间: {env.observation_space}")
    print(f"   动作空间: {env.action_space}")
    print(f"   观察维度: {env.observation_dim}")
    
    # 重置环境
    print("\n3. 重置环境...")
    obs, info = env.reset()
    print(f"   初始观察形状: {obs.shape}")
    print(f"   初始信息: {info}")
    
    # 用随机动作跑一遍
    print("\n4. 执行随机动作...")
    total_reward = 0.0
    step_count = 0
    max_steps = min(50, len(data) - LOOKBACK_WINDOW - 1)  # 限制步数以便快速测试
    
    for step in range(max_steps):
        # 随机动作
        action = env.action_space.sample()
        
        # 执行动作
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        step_count += 1
        
        # 每 10 步打印一次
        if (step + 1) % 10 == 0:
            print(f"   Step {step + 1}: action={action}, reward={reward:.6f}, "
                  f"portfolio_value={info['portfolio_value']:.2f}, "
                  f"total_return={info['total_return']*100:.2f}%")
        
        if terminated:
            print(f"   环境在第 {step + 1} 步终止")
            break
    
    print(f"\n5. 测试完成")
    print(f"   总步数: {step_count}")
    print(f"   累计奖励: {total_reward:.6f}")
    print(f"   最终资产: {info['portfolio_value']:.2f}")
    print(f"   总回报率: {info['total_return']*100:.2f}%")
    
    # 测试 render
    print("\n6. 测试 render...")
    env.render()
    
    print("\n" + "=" * 60)
    print("所有测试通过！环境运行正常。")
    print("=" * 60)


if __name__ == "__main__":
    test_env()
