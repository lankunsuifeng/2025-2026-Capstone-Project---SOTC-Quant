#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import rankdata

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

def rank(series):
    """
    计算序列的排名（0到1之间）
    """
    return pd.Series(rankdata(series, method='average'), index=series.index) / len(series)

def delay(series, periods=1):
    """
    延迟函数：返回前 periods 期的值
    """
    return series.shift(periods)

def delta(series, periods=1):
    """
    差分函数：当前值减去前 periods 期的值
    """
    return series.diff(periods)

def ts_min(series, window):
    """
    滚动最小值
    """
    return series.rolling(window=window, min_periods=1).min()

def ts_max(series, window):
    """
    滚动最大值
    """
    return series.rolling(window=window, min_periods=1).max()

def ts_argmax(series, window):
    """
    滚动窗口内最大值的索引位置（从窗口开始算起）
    
    严格向后看：只使用当前时刻及之前的数据
    返回的是从窗口开始到最大值出现的位置（0-based）
    """
    def argmax_in_window(x):
        """
        在窗口内找到最大值的索引位置
        x 是窗口内的值数组，从最早到最新
        """
        if len(x) == 0:
            return np.nan
        # 找到最大值的索引（如果有多个相同最大值，返回最后一个）
        # 使用 [::-1] 反转，然后 argmax，这样会找到最后一个最大值
        reversed_x = x[::-1]
        max_idx_reversed = np.argmax(reversed_x)
        # 转换回正向索引：从窗口开始到最大值的位置
        max_idx = len(x) - 1 - max_idx_reversed
        return float(max_idx)
    
    # 使用 rolling 确保严格向后看
    return series.rolling(window=window, min_periods=1).apply(
        argmax_in_window,
        raw=True  # 使用 raw=True 提高性能，x 是 numpy array
    )

def signed_power(series, power):
    """
    符号幂函数：保持符号的幂运算
    """
    return np.sign(series) * (np.abs(series) ** power)

def sign(series):
    """
    符号函数：返回 -1, 0, 或 1
    """
    return np.sign(series)

def ts_rank(series, window):
    """
    时间序列滚动排名（等同于 rolling_rank，但使用 Alpha101 命名约定）
    
    严格向后看，避免前瞻偏差
    """
    return rolling_rank(series, window)

def alpha1(df, window=20, argmax_window=5, rank_window=288):
    """
    Alpha1 (Bitcoin-adapted): 
    rolling_rank(Ts_ArgMax(SignedPower(((returns < 0) ? std(returns, window) : close), 2.), argmax_window), rank_window) - 0.5
    
    Adapted for single-asset Bitcoin setting:
    - Uses time-series rolling rank instead of cross-sectional rank
    - All operations are strictly backward-looking (no look-ahead bias)
    - Output is continuous-valued feature suitable for RL
    
    参数:
        df: DataFrame with 'close' column
        window: 计算 returns std 的窗口大小（默认20）
        argmax_window: Ts_ArgMax 的窗口大小（默认5）
        rank_window: 滚动排名的窗口大小（默认20）
    
    返回:
        Series: alpha1 特征值（连续值，范围约在 -0.5 到 0.5 之间）
    """
    # 计算收益率（向后看，无前瞻偏差）
    returns = df['close'].pct_change()
    
    # 计算滚动标准差（向后看，使用过去 window 期的数据）
    returns_std = returns.rolling(window=window, min_periods=window).std()
    
    # 条件：如果 returns < 0，使用 std(returns, window)，否则使用 close
    # 保持原始 Alpha101 的条件结构
    condition = (returns < 0)
    conditional_value = np.where(condition, returns_std, df['close'])
    
    # SignedPower(..., 2) - 保持符号的幂运算
    powered = signed_power(pd.Series(conditional_value, index=df.index), 2.0)
    
    # Ts_ArgMax(..., argmax_window) - 向后看的滚动窗口内最大值位置
    argmax_result = ts_argmax(powered, argmax_window)
    
    # rolling_rank(..., rank_window) - 0.5
    # 使用时间序列滚动排名（只向后看，避免前瞻偏差）
    alpha1_result = rolling_rank(argmax_result, rank_window) - 0.5
    
    return alpha1_result

def rolling_rank(series, window):
    """
    时间序列滚动窗口排名（归一化到 0-1）
    
    严格向后看，避免前瞻偏差：
    - 只使用当前时刻及之前的数据
    - 每个时间点的排名基于该点及其之前的 window 个值
    - 适合单资产时间序列特征工程
    
    参数:
        series: pandas Series，输入序列
        window: int，滚动窗口大小
    
    返回:
        Series: 归一化排名值（0-1之间）
    """
    def rank_in_window(x):
        """
        在窗口内计算最后一个值的排名（归一化）
        x 是窗口内的值数组，最后一个值是要排名的值
        """
        if len(x) == 0:
            return np.nan
        if len(x) == 1:
            return 0.5  # 只有一个值，排名为中间值
        
        # 计算窗口内所有值的排名
        ranks = rankdata(x, method='average')
        # 取最后一个值的排名（当前时刻的值）
        current_rank = ranks[-1]
        # 归一化到 0-1：rank / (len - 1)
        normalized_rank = (current_rank - 1) / (len(x) - 1)
        
        return normalized_rank
    
    # 使用 rolling 确保严格向后看
    return series.rolling(window=window, min_periods=1).apply(
        rank_in_window,
        raw=True  # 使用 raw=True 提高性能，x 是 numpy array
    )

def alpha7(df, adv_window=20, delta_periods=7, rank_window=60):
    """
    Alpha7 (Bitcoin-adapted):
    (adv20 < volume)
        ? ((-1 * ts_rank(abs(delta(close, 7)), 60)) * sign(delta(close, 7)))
        : (-1)
    
    Adapted for single-asset Bitcoin setting:
    - Volume-gated momentum exhaustion indicator
    - Uses time-series rolling rank (ts_rank) instead of cross-sectional rank
    - All operations are strictly backward-looking (no look-ahead bias)
    - Output is continuous-valued feature suitable for RL
    
    参数:
        df: DataFrame with 'close' and 'volume' columns
        adv_window: int，计算 adv20 的滚动窗口大小（默认20）
        delta_periods: int，delta(close, ...) 的周期数（默认7）
        rank_window: int，ts_rank 的滚动窗口大小（默认60）
    
    返回:
        Series: alpha7 特征值（连续值，适合强化学习）
    """
    volume = df['volume']
    close = df['close']
    
    # adv20: 滚动平均成交量（20个bar的平均值）
    adv20 = volume.rolling(window=adv_window, min_periods=adv_window).mean()
    
    # 条件：adv20 < volume (即 volume > adv20)
    volume_gate = volume > adv20
    
    # delta(close, 7): 向后看的7期价格变化
    delta_close_7 = delta(close, delta_periods)
    
    # abs(delta(close, 7)): 价格变化的绝对值
    abs_delta_close_7 = np.abs(delta_close_7)
    
    # ts_rank(abs(delta(close, 7)), 60): 滚动排名（严格向后看）
    ts_rank_abs_delta = ts_rank(abs_delta_close_7, rank_window)
    
    # sign(delta(close, 7)): 价格变化的方向（-1, 0, 或 1）
    sign_delta_close_7 = sign(delta_close_7)
    
    # 计算条件分支的值
    # 如果 volume > adv20: (-1 * ts_rank(...)) * sign(...)
    # 否则: -1
    conditional_value = (-1 * ts_rank_abs_delta) * sign_delta_close_7
    else_value = pd.Series(-1.0, index=df.index)
    
    # 应用条件
    alpha7_result = np.where(volume_gate, conditional_value, else_value)
    
    return pd.Series(alpha7_result, index=df.index)

def alpha9(df, window=5):
    """
    Alpha9: rank(((0 < ts_min(delta(close, 1), 5)) ? delta(close, 1) : 
                  ((ts_max(delta(close, 1), 5) < 0) ? delta(close, 1) : (-1 * delta(close, 1)))))
    
    参数:
        df: DataFrame with 'close' column
        window: ts_min 和 ts_max 的窗口大小
    
    返回:
        Series: alpha9 特征值
    """
    close = df['close']
    
    # delta(close, 1)
    delta_close = delta(close, 1)
    
    # ts_min(delta(close, 1), 5)
    min_delta = ts_min(delta_close, window)
    
    # ts_max(delta(close, 1), 5)
    max_delta = ts_max(delta_close, window)
    
    # 条件逻辑
    # 如果 0 < ts_min(...)，使用 delta(close, 1)
    # 否则如果 ts_max(...) < 0，使用 delta(close, 1)
    # 否则使用 -1 * delta(close, 1)
    condition1 = (0 < min_delta)
    condition2 = (max_delta < 0)
    
    result = np.where(
        condition1,
        delta_close,
        np.where(
            condition2,
            delta_close,
            -1 * delta_close
        )
    )
    
    # rank
    alpha9_result = rank(pd.Series(result, index=df.index))
    
    return alpha9_result

def create_alpha_features(df):
    """
    创建 Alpha1, Alpha7, Alpha9 特征
    
    参数:
        df: pandas DataFrame，包含 'close', 'volume' 列
    
    返回:
        pandas DataFrame，添加了 alpha 特征的 DataFrame
    """
    print("\n" + "="*60)
    print("创建 Alpha 特征")
    print("="*60)
    
    # 检查必需的列
    required_cols = ['close', 'volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"DataFrame 中缺少必需的列: {missing_cols}")
    
    # Alpha1
    print("\n计算 Alpha1...")
    df['alpha1'] = alpha1(df)
    print(f"Alpha1 统计:")
    print(df['alpha1'].describe())
    print(f"缺失值数量: {df['alpha1'].isna().sum()}")
    
    # Alpha7
    print("\n计算 Alpha7...")
    df['alpha7'] = alpha7(df)
    print(f"Alpha7 统计:")
    print(df['alpha7'].describe())
    print(f"缺失值数量: {df['alpha7'].isna().sum()}")
    
    # Alpha9
    print("\n计算 Alpha9...")
    df['alpha9'] = alpha9(df)
    print(f"Alpha9 统计:")
    print(df['alpha9'].describe())
    print(f"缺失值数量: {df['alpha9'].isna().sum()}")
    
    return df

def plot_alpha7_and_close(df, title="Alpha7 and Close Price", n_samples=1000):
    """
    绘制 Alpha7 特征和收盘价的对比图（两个子图）
    
    参数:
        df: pandas DataFrame，必须包含 'alpha7' 和 'close' 列
        title: str，图表标题
        n_samples: int，显示的数据点数量（默认1000，只显示前n_samples个）
    """
    if 'alpha7' not in df.columns:
        raise ValueError("DataFrame 中必须包含 'alpha7' 列")
    if 'close' not in df.columns:
        raise ValueError("DataFrame 中必须包含 'close' 列")
    
    # 只取前 n_samples 个数据点
    df_plot = df.head(n_samples)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    
    # 上图：收盘价
    ax1.plot(df_plot.index, df_plot['close'], linewidth=1.5, color='#2E86AB', label='Close Price')
    ax1.set_ylabel('Close Price (USDT)', fontsize=12, fontweight='bold')
    ax1.set_title(f'{title} - Close Price (First {n_samples} samples)', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 下图：Alpha7 特征
    ax2.plot(df_plot.index, df_plot['alpha7'], linewidth=1.5, color='#A23B72', label='Alpha7')
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax2.axhline(y=-1, color='red', linestyle='--', linewidth=0.8, alpha=0.3, label='-1 (else branch)')
    ax2.set_ylabel('Alpha7 Feature Value', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Time', fontsize=12, fontweight='bold')
    ax2.set_title(f'Alpha7 Feature (First {n_samples} samples)', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # 格式化 x 轴日期
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    ax2.xaxis.set_major_locator(mdates.HourLocator(interval=4))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.show()
    
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
    df = load_data(data_file)
    
    # 特征工程：创建 taker_buy_quote_volume 的 log(1+x) + 滚动 z-score 特征
    print("\n" + "="*60)
    print("特征工程：taker_buy_quote_volume log(1+x) + 滚动 z-score")
    print("="*60)
    df = create_taker_buy_volume_feature(df, window=288)
    
    # 创建 Alpha 特征
    df = create_alpha_features(df)
    
    # 绘制 Alpha7 和收盘价（只显示前1000个数据点）
    print("\n正在生成 Alpha7 和收盘价对比图...")
    plot_alpha7_and_close(df, title="Alpha7 and Close Price (2025-01)", n_samples=1000)
    
    print("\n特征工程完成！")

if __name__ == "__main__":
    main()
