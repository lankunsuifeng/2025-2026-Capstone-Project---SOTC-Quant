# run_experiment.py
"""
完整的实验流程：数据分割 -> 训练 -> 测试 -> 可视化
"""
from __future__ import annotations
import sys
from pathlib import Path
from data_split import train_test_split, save_split_data
from ppo_train import (
    df_to_arrays,
    TradingEnv,
    TradingEnvConfig,
    PPOAgent,
    PPOConfig,
)
from ppo_test import backtest, summarize
from plot_backtest import plot_backtest_results
import pandas as pd

def run_full_experiment(
    data_path: str,
    train_ratio: float = 0.8,
    train_updates: int = 200,
    policy_path: str = "ppo_policy.pt",
    train_csv: str = "train_data.csv",
    test_csv: str = "test_data.csv",
    backtest_csv: str = "backtest_steps.csv",
    plot_dir: str = "plots",
    feature_cols: list = None,
    close_col: str = "close_5m",
    time_col: str = None,
):
    """
    运行完整的实验流程
    
    Parameters:
    -----------
    data_path : str
        原始数据路径
    train_ratio : float
        训练集比例（默认0.8）
    train_updates : int
        训练更新次数
    policy_path : str
        模型保存路径
    train_csv : str
        训练数据保存路径
    test_csv : str
        测试数据保存路径
    backtest_csv : str
        回测结果保存路径
    plot_dir : str
        图表保存目录
    feature_cols : list
        特征列名列表
    close_col : str
        收盘价列名
    time_col : str
        时间列名（用于按时间分割）
    """
    # 默认特征列
    if feature_cols is None:
        feature_cols = [
            "volume_5m", "log_ret_1_5m", "ema_ratio_9_21_5m",
            "macd_hist_5m", "adx_5m", "atr_norm_5m", "bb_width_5m",
            "rsi_14_5m", "volume_zscore_50_5m", "state"
        ]
    
    print("=" * 60)
    print("开始完整实验流程")
    print("=" * 60)
    
    # 步骤1: 数据分割
    print("\n[步骤1] 数据分割")
    print("-" * 60)
    df = pd.read_csv(data_path)
    print(f"原始数据: {len(df)} 条")
    
    # 自动检测时间列
    if time_col is None:
        for col in ["datetime", "timestamp", "time", "date"]:
            if col in df.columns:
                time_col = col
                break
    
    df_train, df_test = train_test_split(
        df,
        train_ratio=train_ratio,
        time_col=time_col,
        seed=42
    )
    
    save_split_data(df_train, df_test, train_csv, test_csv)
    
    # 步骤2: 训练模型
    print("\n[步骤2] 训练模型")
    print("-" * 60)
    X_train, close_train = df_to_arrays(
        df_train, feature_cols, close_col=close_col, dropna=True
    )
    print(f"训练数据: {len(X_train)} 条")
    
    env_cfg = TradingEnvConfig(
        fee_bps=2.0,
        max_episode_steps=5000,
        random_start=True,
        seed=42
    )
    env_train = TradingEnv(X_train, close_train, env_cfg)
    
    cfg = PPOConfig()
    agent = PPOAgent(env_train, cfg)
    agent.train(total_updates=train_updates)
    agent.save(policy_path)
    print(f"模型已保存至: {policy_path}")
    
    # 步骤3: 测试模型
    print("\n[步骤3] 回测测试")
    print("-" * 60)
    X_test, close_test = df_to_arrays(
        df_test, feature_cols, close_col=close_col, dropna=True
    )
    print(f"测试数据: {len(X_test)} 条")
    
    env_test_cfg = TradingEnvConfig(
        fee_bps=2.0,
        hold_cost_bps=0.0,
        max_episode_steps=None,
        random_start=False,
        start_index=1,
        seed=42,
    )
    env_test = TradingEnv(X_test, close_test, env_test_cfg)
    
    # 重新创建agent并加载模型
    agent_test = PPOAgent(env_test, cfg)
    agent_test.load(policy_path)
    
    steps = backtest(agent_test, env_test, capital=1.0)
    steps.to_csv(backtest_csv, index=False)
    
    s = summarize(steps)
    print("\n==== 回测摘要 ====")
    for k, v in s.items():
        print(f"{k:>14s}: {v}")
    print(f"\n回测结果已保存至: {backtest_csv}")
    
    # 步骤4: 可视化
    print("\n[步骤4] 生成可视化图表")
    print("-" * 60)
    plot_backtest_results(backtest_csv, save_dir=plot_dir)
    
    print("\n" + "=" * 60)
    print("实验流程完成！")
    print("=" * 60)


if __name__ == "__main__":
    # 默认配置
    DATA_PATH = "BTCUSDT_combined_klines_20210201_20260201_states4_labeled.csv"
    TRAIN_RATIO = 0.8
    TRAIN_UPDATES = 200
    
    # 如果提供了命令行参数，使用参数
    if len(sys.argv) > 1:
        DATA_PATH = sys.argv[1]
    if len(sys.argv) > 2:
        TRAIN_RATIO = float(sys.argv[2])
    if len(sys.argv) > 3:
        TRAIN_UPDATES = int(sys.argv[3])
    
    run_full_experiment(
        data_path=DATA_PATH,
        train_ratio=TRAIN_RATIO,
        train_updates=TRAIN_UPDATES,
    )
