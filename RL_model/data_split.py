# data_split.py
"""
数据分割工具：将数据分割为训练集和测试集
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple

def train_test_split(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    time_col: str = None,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    将数据分割为训练集和测试集
    
    Parameters:
    -----------
    df : pd.DataFrame
        原始数据
    train_ratio : float
        训练集比例（默认0.8，即80%）
    time_col : str, optional
        如果有时间列，按时间顺序分割；否则随机分割
    seed : int
        随机种子（仅用于随机分割）
    
    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame]
        (训练集, 测试集)
    """
    df = df.copy()
    
    # 如果有时间列，按时间顺序分割（更符合实际场景）
    if time_col and time_col in df.columns:
        df = df.sort_values(time_col).reset_index(drop=True)
        split_idx = int(len(df) * train_ratio)
        df_train = df.iloc[:split_idx].copy()
        df_test = df.iloc[split_idx:].copy()
        print(f"按时间顺序分割: 训练集 {len(df_train)} 条 ({len(df_train)/len(df)*100:.1f}%), "
              f"测试集 {len(df_test)} 条 ({len(df_test)/len(df)*100:.1f}%)")
    else:
        # 随机分割
        rng = np.random.default_rng(seed)
        indices = np.arange(len(df))
        rng.shuffle(indices)
        split_idx = int(len(df) * train_ratio)
        train_indices = indices[:split_idx]
        test_indices = indices[split_idx:]
        df_train = df.iloc[train_indices].copy().reset_index(drop=True)
        df_test = df.iloc[test_indices].copy().reset_index(drop=True)
        print(f"随机分割: 训练集 {len(df_train)} 条 ({len(df_train)/len(df)*100:.1f}%), "
              f"测试集 {len(df_test)} 条 ({len(df_test)/len(df)*100:.1f}%)")
    
    return df_train, df_test


def save_split_data(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    train_path: str = "train_data.csv",
    test_path: str = "test_data.csv"
):
    """
    保存分割后的数据
    
    Parameters:
    -----------
    df_train : pd.DataFrame
        训练集
    df_test : pd.DataFrame
        测试集
    train_path : str
        训练集保存路径
    test_path : str
        测试集保存路径
    """
    df_train.to_csv(train_path, index=False)
    df_test.to_csv(test_path, index=False)
    print(f"训练集已保存至: {train_path}")
    print(f"测试集已保存至: {test_path}")


if __name__ == "__main__":
    # 示例用法
    DATA_PATH = "BTCUSDT_combined_klines_20210201_20260201_states4_labeled.csv"
    TRAIN_PATH = "train_data.csv"
    TEST_PATH = "test_data.csv"
    TRAIN_RATIO = 0.8
    
    print(f"正在读取数据: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"原始数据: {len(df)} 条")
    
    # 尝试按时间分割（如果有datetime列）
    time_col = None
    for col in ["datetime", "timestamp", "time", "date"]:
        if col in df.columns:
            time_col = col
            break
    
    df_train, df_test = train_test_split(
        df,
        train_ratio=TRAIN_RATIO,
        time_col=time_col,
        seed=42
    )
    
    save_split_data(df_train, df_test, TRAIN_PATH, TEST_PATH)
    print("\n数据分割完成！")
