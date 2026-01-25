#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
数据预处理和特征工程
使用 2025年1月的数据查看成交量和比特币走势
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
    print(f"\n数据预览:")
    print(df.head())
    print(f"\n数据统计:")
    print(df[['open', 'high', 'low', 'close', 'volume']].describe())
    
    return df

def log_transform_rolling_zscore(series, window=288, min_periods=1):
    """
    对序列进行 log(1+x) 变换，然后计算滚动 z-score
    
    参数:
        series: pandas Series，输入数据
        window: int，滚动窗口大小（默认288，即24小时，因为5分钟数据，288*5=1440分钟=24小时）
        min_periods: int，计算所需的最小观测数
    
    返回:
        pandas Series，滚动 z-score 结果
    """
    # Step 1: log(1+x) 变换
    log_transformed = np.log1p(series)
    
    # Step 2: 计算滚动均值和标准差
    rolling_mean = log_transformed.rolling(window=window, min_periods=min_periods).mean()
    rolling_std = log_transformed.rolling(window=window, min_periods=min_periods).std()
    
    # Step 3: 计算滚动 z-score
    # z-score = (x - mean) / std
    rolling_zscore = (log_transformed - rolling_mean) / rolling_std
    
    return rolling_zscore

def create_taker_buy_volume_feature(df, window=288, feature_name='taker_buy_quote_volume_log_zscore'):
    """
    创建 taker_buy_quote_volume 的特征：log(1+x) + 滚动 z-score
    
    参数:
        df: pandas DataFrame，包含 'taker_buy_quote_volume' 列
        window: int，滚动窗口大小（默认288，即24小时）
        feature_name: str，新特征列的名称
    
    返回:
        pandas DataFrame，添加了新特征的 DataFrame
    """
    if 'taker_buy_quote_volume' not in df.columns:
        raise ValueError("DataFrame 中必须包含 'taker_buy_quote_volume' 列")
    
    # 创建特征
    df[feature_name] = log_transform_rolling_zscore(
        df['taker_buy_quote_volume'], 
        window=window
    )
    
    print(f"\n特征 '{feature_name}' 已创建")
    print(f"窗口大小: {window} (约 {window * 5 / 60:.1f} 小时)")
    print(f"特征统计:")
    print(df[feature_name].describe())
    print(f"缺失值数量: {df[feature_name].isna().sum()}")
    
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
    
    # 特征工程：创建 taker_buy_quote_volume 的 log(1+x) + 滚动 z-score 特征
    print("\n" + "="*60)
    print("特征工程：taker_buy_quote_volume log(1+x) + 滚动 z-score")
    print("="*60)
    df = create_taker_buy_volume_feature(df, window=288)
    
    print("\n特征工程完成！")

if __name__ == "__main__":
    main()
