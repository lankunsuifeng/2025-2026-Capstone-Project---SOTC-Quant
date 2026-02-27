# plot_backtest.py
"""
可视化 PPO 回测结果
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

def plot_backtest_results(csv_path: str, save_dir: str = "plots", figsize=(15, 10)):
    """
    绘制回测结果的多面板图表
    
    Parameters:
    -----------
    csv_path : str
        回测结果 CSV 文件路径
    save_dir : str
        保存图片的目录
    figsize : tuple
        图片大小
    """
    # 读取数据
    df = pd.read_csv(csv_path)
    if len(df) == 0:
        print("警告: 数据为空，无法绘图")
        return
    
    # 创建保存目录
    Path(save_dir).mkdir(exist_ok=True)
    
    # 计算额外指标
    df['drawdown'] = calculate_drawdown(df['equity'].values)
    df['returns'] = df['equity'].pct_change().fillna(0)
    
    # 创建图表
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)
    
    # 1. Equity Curve (权益曲线)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(df['step'], df['equity'], linewidth=1.5, color='#2E86AB', label='Equity')
    ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Baseline (1.0)')
    ax1.set_xlabel('Step', fontsize=11)
    ax1.set_ylabel('Equity', fontsize=11)
    ax1.set_title('Equity Curve (权益曲线)', fontsize=12, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # 2. Drawdown (回撤)
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.fill_between(df['step'], df['drawdown'], 0, color='#A23B72', alpha=0.6, label='Drawdown')
    ax2.set_xlabel('Step', fontsize=11)
    ax2.set_ylabel('Drawdown', fontsize=11)
    ax2.set_title('Drawdown (回撤)', fontsize=12, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    # 3. Price and Positions (价格和持仓)
    ax3 = fig.add_subplot(gs[1, 1])
    ax3_twin = ax3.twinx()
    
    # 价格
    line1 = ax3.plot(df['step'], df['close'], linewidth=1, color='#1B998B', alpha=0.7, label='Price')
    ax3.set_xlabel('Step', fontsize=11)
    ax3.set_ylabel('Price', fontsize=11, color='#1B998B')
    ax3.tick_params(axis='y', labelcolor='#1B998B')
    
    # 持仓
    colors_pos = ['#E63946' if p == -1 else '#06A77D' if p == 1 else '#F1A208' for p in df['pos']]
    ax3_twin.scatter(df['step'], df['pos'], c=colors_pos, s=10, alpha=0.5, label='Position')
    ax3_twin.set_ylabel('Position (-1/0/1)', fontsize=11, color='#E63946')
    ax3_twin.tick_params(axis='y', labelcolor='#E63946')
    ax3_twin.set_ylim(-1.5, 1.5)
    
    ax3.set_title('Price & Positions (价格和持仓)', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. Actions Distribution (动作分布)
    ax4 = fig.add_subplot(gs[2, 0])
    action_counts = df['action'].value_counts().sort_index()
    colors_act = {'-1': '#E63946', '0': '#F1A208', '1': '#06A77D'}
    bars = ax4.bar([str(k) for k in action_counts.index], action_counts.values, 
                   color=[colors_act.get(str(k), '#808080') for k in action_counts.index],
                   alpha=0.7, edgecolor='black', linewidth=1.2)
    ax4.set_xlabel('Action', fontsize=11)
    ax4.set_ylabel('Count', fontsize=11)
    ax4.set_title('Actions Distribution (动作分布)', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=10)
    
    # 5. Returns Distribution (收益分布)
    ax5 = fig.add_subplot(gs[2, 1])
    returns = df['returns'].values
    returns = returns[returns != 0]  # 移除零收益
    if len(returns) > 0:
        ax5.hist(returns, bins=50, color='#7209B7', alpha=0.7, edgecolor='black', linewidth=0.5)
        ax5.axvline(x=0, color='red', linestyle='--', linewidth=1.5, label='Zero')
        ax5.axvline(x=np.mean(returns), color='green', linestyle='--', linewidth=1.5, label=f'Mean: {np.mean(returns):.6f}')
        ax5.set_xlabel('Returns', fontsize=11)
        ax5.set_ylabel('Frequency', fontsize=11)
        ax5.set_title('Returns Distribution (收益分布)', fontsize=12, fontweight='bold')
        ax5.legend(loc='best')
        ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Cumulative Metrics (累计指标)
    ax6 = fig.add_subplot(gs[3, :])
    ax6_twin = ax6.twinx()
    
    # 累计奖励
    line1 = ax6.plot(df['step'], df['cum_reward'], linewidth=1.5, color='#06A77D', label='Cumulative Reward')
    ax6.set_xlabel('Step', fontsize=11)
    ax6.set_ylabel('Cumulative Reward', fontsize=11, color='#06A77D')
    ax6.tick_params(axis='y', labelcolor='#06A77D')
    
    # 累计手续费
    df['cum_fee'] = df['fee'].cumsum()
    line2 = ax6_twin.plot(df['step'], df['cum_fee'], linewidth=1.5, color='#E63946', 
                         linestyle='--', label='Cumulative Fee')
    ax6_twin.set_ylabel('Cumulative Fee', fontsize=11, color='#E63946')
    ax6_twin.tick_params(axis='y', labelcolor='#E63946')
    
    ax6.set_title('Cumulative Reward & Fee (累计奖励和手续费)', fontsize=12, fontweight='bold')
    
    # 合并图例
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax6.legend(lines, labels, loc='best')
    ax6.grid(True, alpha=0.3)
    
    # 添加统计信息文本框
    total_return = (df['equity'].iloc[-1] / df['equity'].iloc[0] - 1) * 100
    max_dd = df['drawdown'].min() * 100
    sharpe = np.mean(df['returns']) / (np.std(df['returns']) + 1e-12) if len(df['returns']) > 0 else 0
    total_fee = df['fee'].sum()
    total_turnover = df['turnover'].sum()
    
    stats_text = (
        f"Total Return: {total_return:.2f}%\n"
        f"Max Drawdown: {max_dd:.2f}%\n"
        f"Sharpe Ratio: {sharpe:.4f}\n"
        f"Total Fee: {total_fee:.6f}\n"
        f"Total Turnover: {total_turnover:.2f}\n"
        f"Total Steps: {len(df)}"
    )
    
    fig.text(0.02, 0.02, stats_text, fontsize=9, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('PPO Backtest Results (PPO 回测结果)', fontsize=14, fontweight='bold', y=0.995)
    
    # 保存图片
    save_path = Path(save_dir) / "backtest_results.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"图表已保存至: {save_path}")
    
    plt.show()


def calculate_drawdown(equity: np.ndarray) -> np.ndarray:
    """
    计算回撤序列
    
    Parameters:
    -----------
    equity : np.ndarray
        权益序列
    
    Returns:
    --------
    np.ndarray
        回撤序列
    """
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak
    return drawdown


if __name__ == "__main__":
    import sys
    
    # 默认使用 backtest_steps.csv
    csv_path = "backtest_steps.csv"
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    
    if not Path(csv_path).exists():
        print(f"错误: 文件 {csv_path} 不存在")
        print("用法: python plot_backtest.py [csv_path]")
        sys.exit(1)
    
    print(f"正在读取: {csv_path}")
    plot_backtest_results(csv_path, save_dir="plots")
