#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MVP Feature Engineering: Price Dynamics Features
实现10个基础价格动态特征
"""

import pandas as pd
import numpy as np
import os

def load_data(file_path):
    """加载 CSV 数据文件"""
    print(f"正在加载数据: {file_path}")
    df = pd.read_csv(file_path)
    
    # 转换时间列为 datetime 类型
    df['open_time'] = pd.to_datetime(df['open_time'], utc=True)
    
    # 设置时间为索引
    df.set_index('open_time', inplace=True)
    
    print(f"数据加载完成，共 {len(df)} 条记录")
    print(f"时间范围: {df.index.min()} 至 {df.index.max()}")
    
    return df

def create_price_dynamics_features(df):
    """
    创建价格动态特征组（MVP）
    
    参数:
        df: pandas DataFrame，必须包含 'open', 'high', 'low', 'close', 'volume' 列
    
    返回:
        pandas DataFrame，添加了所有价格动态特征的 DataFrame
    """
    print("\n" + "="*60)
    print("创建价格动态特征（MVP）")
    print("="*60)
    
    # 检查必需的列
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"DataFrame 中缺少必需的列: {missing_cols}")
    
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    # 1. Log Return (1-bar)
    print("\n1. 计算 Log Return (1-bar)...")
    df['log_return_1'] = np.log(close / close.shift(1))
    
    # 2. Log Return (5-bar)
    print("2. 计算 Log Return (5-bar)...")
    df['log_return_5'] = np.log(close / close.shift(5))
    
    # 3. Log Return (20-bar)
    print("3. 计算 Log Return (20-bar)...")
    df['log_return_20'] = np.log(close / close.shift(20))
    
    # 4. Rolling Volatility
    print("4. 计算 Rolling Volatility...")
    returns = np.log(close / close.shift(1))
    df['rolling_volatility'] = returns.rolling(window=20, min_periods=20).apply(
        lambda x: np.sqrt(np.mean(x**2)), raw=True
    )
    
    # 5. High-Low Range
    print("5. 计算 High-Low Range...")
    df['high_low_range'] = (high - low) / close
    
    # 6. Price vs EMA Deviation
    print("6. 计算 Price vs EMA Deviation...")
    ema_20 = close.ewm(span=20, adjust=False).mean()
    df['price_ema_deviation'] = (close - ema_20) / ema_20
    
    # 7. EMA Slope
    print("7. 计算 EMA Slope...")
    df['ema_slope'] = ema_20 - ema_20.shift(1)
    
    # 8. VWAP Deviation
    print("8. 计算 VWAP Deviation...")
    # VWAP = sum(price * volume) / sum(volume) over rolling window
    # 使用典型价格 (high + low + close) / 3 作为价格
    typical_price = (high + low + close) / 3
    vwap = (typical_price * volume).rolling(window=20, min_periods=20).sum() / \
           volume.rolling(window=20, min_periods=20).sum()
    df['vwap_deviation'] = (close - vwap) / vwap
    
    # 9. Price Position in Recent Range
    print("9. 计算 Price Position in Recent Range...")
    W = 20
    recent_low = low.rolling(window=W, min_periods=W).min()
    recent_high = high.rolling(window=W, min_periods=W).max()
    df['price_range_position'] = (close - recent_low) / (recent_high - recent_low + 1e-10)  # 避免除零
    
    # 10. Trend vs Chop Indicator
    print("10. 计算 Trend vs Chop Indicator...")
    ema_short = close.ewm(span=10, adjust=False).mean()
    ema_long = close.ewm(span=30, adjust=False).mean()
    volatility = df['rolling_volatility']  # 使用之前计算的波动率
    df['trend_strength'] = np.abs(ema_short - ema_long) / (volatility + 1e-10)  # 避免除零
    
    # 打印特征统计信息
    feature_cols = [
        'log_return_1', 'log_return_5', 'log_return_20',
        'rolling_volatility', 'high_low_range', 'price_ema_deviation',
        'ema_slope', 'vwap_deviation', 'price_range_position', 'trend_strength'
    ]
    
    print("\n" + "="*60)
    print("特征统计摘要")
    print("="*60)
    for col in feature_cols:
        if col in df.columns:
            print(f"\n{col}:")
            print(df[col].describe())
            print(f"缺失值数量: {df[col].isna().sum()}")
    
    return df

def main():
    """主函数"""
    # 数据文件路径
    data_file = './data/BTCUSDT/5m/klines_2025_01.csv'
    
    # 检查文件是否存在
    if not os.path.exists(data_file):
        print(f"错误: 文件不存在 - {data_file}")
        return
    
    # 加载数据
    df = load_data(data_file)
    
    # 创建价格动态特征
    df = create_price_dynamics_features(df)
    
    print("\n特征工程完成！")
    print(f"\n最终 DataFrame 形状: {df.shape}")
    print(f"特征列: {[col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume', 'num_trades', 'taker_buy_volume', 'taker_buy_quote_volume']]}")

if __name__ == "__main__":
    main()
