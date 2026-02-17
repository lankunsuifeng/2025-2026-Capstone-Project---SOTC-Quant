#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
绘制组合技术指标图
包含：Close + EMA20 + EMA50, Rolling Volatility, Bollinger Band Width
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
from mvp_feature import load_data, create_technical_indicators

def plot_combined_indicators(df, n_samples=1000):
    """
    绘制组合技术指标图
    
    参数:
        df: pandas DataFrame，包含特征和 'close' 列
        n_samples: int，显示的数据点数量（默认1000）
    """
    if 'close' not in df.columns:
        raise ValueError("DataFrame 中必须包含 'close' 列")
    
    # 检查必需的特征是否存在
    required_features = ['ema_20', 'ema_50', 'rolling_volatility_20', 'bb_width']
    missing_features = [f for f in required_features if f not in df.columns]
    if missing_features:
        raise ValueError(f"DataFrame 中缺少以下特征: {missing_features}")
    
    # 只取前 n_samples 个数据点
    df_plot = df.head(n_samples)
    
    # 创建3个子图
    fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    
    # 第一个子图：Close + EMA20 + EMA50
    ax = axes[0]
    ax.plot(df_plot.index, df_plot['close'], linewidth=2, color='#2E86AB', label='Close Price', alpha=0.8)
    ax.plot(df_plot.index, df_plot['ema_20'], linewidth=1.5, color='#A23B72', label='EMA 20', alpha=0.7)
    ax.plot(df_plot.index, df_plot['ema_50'], linewidth=1.5, color='#F18F01', label='EMA 50', alpha=0.7)
    
    # 填充EMA之间的区域
    ax.fill_between(df_plot.index, df_plot['ema_20'], df_plot['ema_50'], 
                   where=(df_plot['ema_20'] >= df_plot['ema_50']),
                   color='green', alpha=0.1, label='EMA20 > EMA50')
    ax.fill_between(df_plot.index, df_plot['ema_20'], df_plot['ema_50'], 
                   where=(df_plot['ema_20'] < df_plot['ema_50']),
                   color='red', alpha=0.1, label='EMA20 < EMA50')
    
    ax.set_ylabel('Price (USDT)', fontsize=12, fontweight='bold')
    ax.set_title('Close Price vs EMA20 vs EMA50', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # 第二个子图：Rolling Volatility
    ax = axes[1]
    ax.plot(df_plot.index, df_plot['rolling_volatility_20'], linewidth=1.5, color='#C73E1D', 
           label='Rolling Volatility (20-period)', alpha=0.8)
    
    # 添加波动率的统计参考线
    vol_mean = df_plot['rolling_volatility_20'].mean()
    vol_std = df_plot['rolling_volatility_20'].std()
    ax.axhline(y=vol_mean, color='gray', linestyle='--', linewidth=1, alpha=0.7, 
               label=f'Mean ({vol_mean:.6f})')
    ax.axhline(y=vol_mean + vol_std, color='red', linestyle='--', linewidth=0.8, alpha=0.5, 
               label=f'Mean + 1σ')
    ax.axhline(y=vol_mean - vol_std, color='green', linestyle='--', linewidth=0.8, alpha=0.5, 
               label=f'Mean - 1σ')
    
    # 填充高波动区域
    ax.fill_between(df_plot.index, vol_mean + vol_std, df_plot['rolling_volatility_20'], 
                   where=(df_plot['rolling_volatility_20'] >= vol_mean + vol_std),
                   color='red', alpha=0.2, label='High Volatility')
    ax.fill_between(df_plot.index, vol_mean - vol_std, df_plot['rolling_volatility_20'], 
                   where=(df_plot['rolling_volatility_20'] <= vol_mean - vol_std),
                   color='green', alpha=0.2, label='Low Volatility')
    
    ax.set_ylabel('Volatility', fontsize=12, fontweight='bold')
    ax.set_title('Rolling Volatility (20-period, log returns std)', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # 第三个子图：Bollinger Band Width
    ax = axes[2]
    ax.plot(df_plot.index, df_plot['bb_width'], linewidth=1.5, color='#6A994E', 
           label='Bollinger Band Width', alpha=0.8)
    
    # 添加带宽的统计参考线
    bb_width_mean = df_plot['bb_width'].mean()
    bb_width_std = df_plot['bb_width'].std()
    ax.axhline(y=bb_width_mean, color='gray', linestyle='--', linewidth=1, alpha=0.7, 
               label=f'Mean ({bb_width_mean:.6f})')
    ax.axhline(y=bb_width_mean + bb_width_std, color='red', linestyle='--', linewidth=0.8, alpha=0.5, 
               label=f'Mean + 1σ')
    ax.axhline(y=bb_width_mean - bb_width_std, color='green', linestyle='--', linewidth=0.8, alpha=0.5, 
               label=f'Mean - 1σ')
    
    # 填充高带宽区域
    ax.fill_between(df_plot.index, bb_width_mean + bb_width_std, df_plot['bb_width'], 
                   where=(df_plot['bb_width'] >= bb_width_mean + bb_width_std),
                   color='red', alpha=0.2, label='Wide Bands (High Volatility)')
    ax.fill_between(df_plot.index, bb_width_mean - bb_width_std, df_plot['bb_width'], 
                   where=(df_plot['bb_width'] <= bb_width_mean - bb_width_std),
                   color='green', alpha=0.2, label='Narrow Bands (Low Volatility)')
    
    ax.set_ylabel('Band Width', fontsize=12, fontweight='bold')
    ax.set_xlabel('Time', fontsize=12, fontweight='bold')
    ax.set_title('Bollinger Band Width (20-period, k=2)', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # 格式化 x 轴日期
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    axes[-1].xaxis.set_major_locator(mdates.HourLocator(interval=6))
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.show()
    
    # 打印统计信息
    print(f"\n指标统计信息（前 {n_samples} 个样本）:")
    print(f"\nRolling Volatility:")
    print(f"  均值: {vol_mean:.6f}")
    print(f"  标准差: {vol_std:.6f}")
    print(f"  最小值: {df_plot['rolling_volatility_20'].min():.6f}")
    print(f"  最大值: {df_plot['rolling_volatility_20'].max():.6f}")
    
    print(f"\nBollinger Band Width:")
    print(f"  均值: {bb_width_mean:.6f}")
    print(f"  标准差: {bb_width_std:.6f}")
    print(f"  最小值: {df_plot['bb_width'].min():.6f}")
    print(f"  最大值: {df_plot['bb_width'].max():.6f}")
    
    return fig

def main():
    """主函数"""
    # 数据文件路径
    data_file = './data/BTCUSDT/5m/klines_2025_01.csv'
    
    # 检查文件是否存在
    if not os.path.exists(data_file):
        print(f"错误: 文件不存在 - {data_file}")
        return
    
    # 加载数据
    print("="*60)
    print("加载数据")
    print("="*60)
    df = load_data(data_file)
    
    # 创建技术指标特征
    df = create_technical_indicators(df)
    
    # 绘制组合指标图
    print("\n" + "="*60)
    print("生成组合指标对比图（前1000个数据点）")
    print("="*60)
    plot_combined_indicators(df, n_samples=1000)
    
    print("\n完成！")

if __name__ == "__main__":
    main()
