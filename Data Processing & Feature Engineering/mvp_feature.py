#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Classical Feature Engineering: Classic Technical Indicators
实现经典技术指标特征集，适合强化学习
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

def calculate_sma(series, window):
    """计算简单移动平均"""
    return series.rolling(window=window, min_periods=window).mean()

def calculate_ema(series, span):
    """计算指数移动平均"""
    return series.ewm(span=span, adjust=False).mean()

def calculate_macd(close, fast=12, slow=26, signal=9):
    """
    计算 MACD 指标
    
    参数:
        close: 收盘价序列
        fast: 快速EMA周期（默认12）
        slow: 慢速EMA周期（默认26）
        signal: 信号线EMA周期（默认9）
    
    返回:
        macd_line: MACD线 (EMA12 - EMA26)
        signal_line: 信号线 (EMA9 of MACD)
        histogram: MACD柱状图 (MACD - Signal)
    """
    ema_fast = calculate_ema(close, fast)
    ema_slow = calculate_ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal)
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram

def calculate_rsi(close, window=14, eps=1e-10):
    """
    计算相对强弱指标 (RSI) - 使用 Wilder 平滑方法
    
    参数:
        close: 收盘价序列
        window: RSI窗口大小（默认14）
        eps: 避免除零的小常数（默认1e-10）
    
    返回:
        RSI值（0-100之间）
    """
    delta = close.diff()
    
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    
    # Wilder smoothing (RMA): alpha = 1/window
    avg_gain = gain.ewm(alpha=1/window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1/window, adjust=False, min_periods=window).mean()
    
    rs = avg_gain / (avg_loss + eps)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_rolling_volatility(close, window=20):
    """
    计算滚动波动率（对数收益率的滚动标准差）
    
    参数:
        close: 收盘价序列
        window: 滚动窗口大小（默认20）
    
    返回:
        滚动波动率
    """
    log_returns = np.log(close / close.shift(1))
    volatility = log_returns.rolling(window=window, min_periods=window).std()
    
    return volatility

def calculate_bollinger_bands(close, window=20, k=2):
    """
    计算布林带
    
    参数:
        close: 收盘价序列
        window: 滚动窗口大小（默认20）
        k: 标准差倍数（默认2）
    
    返回:
        upper_band: 上轨
        middle_band: 中轨（SMA）
        lower_band: 下轨
        band_width: 带宽
        price_position: 价格在带内的位置（0-1之间）
    """
    middle_band = calculate_sma(close, window)
    std = close.rolling(window=window, min_periods=window).std()
    
    upper_band = middle_band + (k * std)
    lower_band = middle_band - (k * std)
    
    # 带宽
    band_width = (upper_band - lower_band) / (middle_band + 1e-10)
    
    # 价格在带内的位置（0=下轨，1=上轨）
    price_position = (close - lower_band) / (upper_band - lower_band + 1e-10)
    
    return upper_band, middle_band, lower_band, band_width, price_position

def calculate_obv(close, volume):
    """
    计算能量潮指标 (On-Balance Volume)
    
    参数:
        close: 收盘价序列
        volume: 成交量序列
    
    返回:
        OBV值
    """
    obv = pd.Series(index=close.index, dtype=float)
    obv.iloc[0] = volume.iloc[0]
    
    for i in range(1, len(close)):
        if close.iloc[i] > close.iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
        elif close.iloc[i] < close.iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
        else:
            obv.iloc[i] = obv.iloc[i-1]
    
    return obv

def calculate_vwap(high, low, close, volume, window=20):
    """
    计算成交量加权平均价格 (VWAP)
    
    参数:
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        volume: 成交量序列
        window: 滚动窗口大小（默认20）
    
    返回:
        VWAP值
    """
    typical_price = (high + low + close) / 3
    vwap = (typical_price * volume).rolling(window=window, min_periods=window).sum() / \
           volume.rolling(window=window, min_periods=window).sum()
    
    return vwap

def create_technical_indicators(df):
    """
    创建经典技术指标特征集
    
    参数:
        df: pandas DataFrame，必须包含 'open', 'high', 'low', 'close', 'volume' 列
    
    返回:
        pandas DataFrame，添加了所有技术指标特征的 DataFrame
    """
    print("\n" + "="*60)
    print("创建经典技术指标特征（MVP）")
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
    epsilon = 1e-10  # 避免除零的小常数
    
    # 1. Simple Moving Averages
    print("\n1. 计算 Simple Moving Averages...")
    df['sma_20'] = calculate_sma(close, 20)
    df['sma_50'] = calculate_sma(close, 50)
    
    # Price deviation from SMA
    df['price_deviation_sma20'] = (close - df['sma_20']) / (df['sma_20'] + epsilon)
    df['price_deviation_sma50'] = (close - df['sma_50']) / (df['sma_50'] + epsilon)
    
    # 2. Exponential Moving Averages
    print("2. 计算 Exponential Moving Averages...")
    df['ema_10'] = calculate_ema(close, 10)
    df['ema_20'] = calculate_ema(close, 20)
    df['ema_50'] = calculate_ema(close, 50)
    
    # EMA slopes
    df['ema_slope_10'] = df['ema_10'] - df['ema_10'].shift(1)
    df['ema_slope_20'] = df['ema_20'] - df['ema_20'].shift(1)
    df['ema_slope_50'] = df['ema_50'] - df['ema_50'].shift(1)
    
    # 3. MACD
    print("3. 计算 MACD...")
    macd_line, signal_line, histogram = calculate_macd(close, fast=12, slow=26, signal=9)
    df['macd_line'] = macd_line
    df['macd_signal'] = signal_line
    df['macd_histogram'] = histogram
    
    # 4. RSI
    print("4. 计算 RSI...")
    df['rsi_14'] = calculate_rsi(close, window=14)
    
    # 5. Rolling Volatility
    print("5. 计算 Rolling Volatility...")
    df['rolling_volatility_20'] = calculate_rolling_volatility(close, window=20)
    
    # 6. Bollinger Bands
    print("6. 计算 Bollinger Bands...")
    _, _, _, band_width, price_position = calculate_bollinger_bands(close, window=20, k=2)
    df['bb_width'] = band_width
    df['bb_price_position'] = price_position
    
    # 7. Volume Moving Average
    print("7. 计算 Volume Moving Average...")
    df['volume_ma_20'] = calculate_sma(volume, 20)
    df['volume_ratio'] = volume / (df['volume_ma_20'] + epsilon)
    
    # 8. On-Balance Volume (OBV)
    print("8. 计算 On-Balance Volume...")
    df['obv'] = calculate_obv(close, volume)
    
    # 9. VWAP
    print("9. 计算 VWAP...")
    vwap = calculate_vwap(high, low, close, volume, window=20)
    df['vwap'] = vwap
    df['vwap_deviation'] = (close - vwap) / (vwap + epsilon)
    
    # 10. Regime-oriented Features (for market regime classification)
    print("10. 计算 Regime-oriented Features...")
    
    # 1. Short-term trend direction
    df['trend_sign_fast'] = np.sign(df['ema_10'] - df['ema_20'])
    
    # 2. Medium-term trend direction
    df['trend_sign_medium'] = np.sign(df['ema_20'] - df['ema_50'])
    
    # 3. Trend strength (scale-free)
    df['trend_strength'] = np.abs(df['ema_20'] - df['ema_50']) / (close + epsilon)
    
    # 4. Volatility level (relative) - rolling z-score of volatility
    rolling_vol = df['rolling_volatility_20']
    vol_mean = rolling_vol.rolling(window=100, min_periods=100).mean()
    vol_std = rolling_vol.rolling(window=100, min_periods=100).std()
    df['vol_zscore'] = (rolling_vol - vol_mean) / (vol_std + epsilon)
    
    # 5. Volatility change (regime transition signal)
    df['delta_vol'] = rolling_vol - rolling_vol.shift(1)
    
    # 打印特征统计信息
    feature_cols = [
        'sma_20', 'sma_50', 'price_deviation_sma20', 'price_deviation_sma50',
        'ema_10', 'ema_20', 'ema_50', 'ema_slope_10', 'ema_slope_20', 'ema_slope_50',
        'macd_line', 'macd_signal', 'macd_histogram',
        'rsi_14',
        'rolling_volatility_20',
        'bb_width', 'bb_price_position',
        'volume_ma_20', 'volume_ratio',
        'obv',
        'vwap', 'vwap_deviation',
        'trend_sign_fast', 'trend_sign_medium', 'trend_strength', 'vol_zscore', 'delta_vol'
    ]
    
    print("\n" + "="*60)
    print("特征统计摘要")
    print("="*60)
    for col in feature_cols:
        if col in df.columns:
            print(f"\n{col}:")
            print(f"  缺失值数量: {df[col].isna().sum()}")
            print(f"  统计: mean={df[col].mean():.6f}, std={df[col].std():.6f}, min={df[col].min():.6f}, max={df[col].max():.6f}")
    
    return df

def get_feature_columns(df):
    """
    获取所有特征列名（排除原始OHLCV数据）
    
    参数:
        df: pandas DataFrame
    
    返回:
        list: 特征列名列表
    """
    original_cols = ['open', 'high', 'low', 'close', 'volume', 'close_time', 
                     'quote_volume', 'num_trades', 'taker_buy_volume', 'taker_buy_quote_volume']
    feature_cols = [col for col in df.columns if col not in original_cols]
    return feature_cols

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
    
    # 创建技术指标特征
    df = create_technical_indicators(df)
    
    # 获取特征列
    feature_cols = get_feature_columns(df)
    
    print("\n" + "="*60)
    print("特征工程完成！")
    print("="*60)
    print(f"\n最终 DataFrame 形状: {df.shape}")
    print(f"特征数量: {len(feature_cols)}")
    print(f"\n特征列表:")
    for i, col in enumerate(feature_cols, 1):
        print(f"  {i}. {col}")

if __name__ == "__main__":
    main()
