# backtest_run.py
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt


from ppo import (
    df_to_arrays,
    TradingEnv,
    TradingEnvConfig,
    PPOAgent,
    PPOConfig,
)

def buy_and_hold(df_test: pd.DataFrame, close_col: str = "close_5m", capital: float = 1.0) -> pd.DataFrame:
    close = pd.to_numeric(df_test[close_col], errors="coerce").to_numpy(dtype=float)
    close = close[np.isfinite(close)]
    if len(close) < 2:
        return pd.DataFrame()

    step_ret = close[1:] / np.maximum(close[:-1], 1e-12) - 1.0  # simple return per step

    equity = np.empty(len(step_ret) + 1, dtype=float)
    equity[0] = 1.0
    for i in range(len(step_ret)):
        equity[i + 1] = equity[i] * (1.0 + step_ret[i])

    steps = pd.DataFrame({
        "step": np.arange(len(equity), dtype=int),
        "close": close[:len(equity)],
        "ret": np.r_[np.nan, step_ret],          
        "step_ret": np.r_[np.nan, step_ret],     
        "equity": equity,
        "nav": equity * float(capital),
    })
    return steps
def max_drawdown(equity: np.ndarray) -> float:
    peak = equity[0]
    mdd = 0.0
    for x in equity:
        if x > peak:
            peak = x
        dd = x / (peak + 1e-12) - 1.0
        if dd < mdd:
            mdd = dd
    return float(mdd)

@torch.no_grad()
def act_deterministic(agent: PPOAgent, obs_np: np.ndarray) -> int:
    obs = torch.tensor(obs_np, dtype=torch.float32, device=agent.device)
    logits, _ = agent.model(obs.unsqueeze(0))
    act_idx = int(torch.argmax(logits, dim=-1).item())     # 0/1/2
    return int(agent.IDX_TO_ACT[act_idx])                  # -1/0/1

def backtest(agent: PPOAgent, env: TradingEnv, capital: float = 1.0) -> pd.DataFrame:
    obs = env.reset()
    done = False
    rows = []
    cum_reward = 0.0
    step = 0

    while not done:
        action = act_deterministic(agent, obs)
        next_obs, reward, done, info = env.step(action)

        t = int(info["t"])
        simple_ret = (env.close[t] + 1e-12) / (env.close[t-1] + 1e-12) - 1.0
        pos = int(info["pos"])     # -1/0/1
        fee = float(info["fee"])
        hold = float(info["hold_cost"])
        step_ret = pos * float(simple_ret) - fee - hold
        equity = (rows[-1]["equity"] if rows else 1.0) * (1.0 + step_ret)   
        nav = equity * float(capital)

        rows.append({
            "step": step,
            "step_ret": float(step_ret),
            "t": t,
            "close": float(env.close[t]),
            "ret": float(info["ret"]),
            "action": int(action),                 
            "pos": int(info["pos"]),           
            "turnover": float(info["turnover"]),
            "fee": float(info["fee"]),
            "hold_cost": float(info["hold_cost"]),
            "reward": float(reward),
            "cum_reward": float(cum_reward),
            "equity": equity,
            "nav": nav,
        })

        obs = next_obs
        step += 1

    return pd.DataFrame(rows)

def summarize(steps: pd.DataFrame) -> dict:
    out = {"bars": int(len(steps))}
    if len(steps) == 0:
        return out
    # equity
    if "equity" not in steps.columns:
        raise ValueError("summarize() expects column 'equity' in steps DataFrame")

    equity = steps["equity"].to_numpy(dtype=float)
    out["equity_end"] = float(equity[-1])
    out["total_return"] = float(equity[-1] / (equity[0] + 1e-12) - 1.0)
    out["max_drawdown"] = max_drawdown(equity)

    # sharpe (prefer step_ret if available)
    if len(steps) >= 3:
        if "step_ret" in steps.columns:
            step_ret = steps["step_ret"].to_numpy(dtype=float)
            step_ret = step_ret[np.isfinite(step_ret)]
        else:
            step_ret = equity[1:] / np.maximum(equity[:-1], 1e-12) - 1.0
            step_ret = step_ret[np.isfinite(step_ret)]

        out["sharpe_step"] = float(np.mean(step_ret) / (np.std(step_ret) + 1e-12)) if len(step_ret) > 2 else 0.0
    else:
        out["sharpe_step"] = 0.0

    # optional columns (PPO has them; buy&hold may not)
    out["turnover_sum"] = float(steps["turnover"].sum()) if "turnover" in steps.columns else 0.0
    out["fee_sum"] = float(steps["fee"].sum()) if "fee" in steps.columns else 0.0
    out["hold_cost_sum"] = float(steps["hold_cost"].sum()) if "hold_cost" in steps.columns else 0.0
    out["reward_sum"] = float(steps["reward"].sum()) if "reward" in steps.columns else 0.0

    return out
def add_drawdown_column(steps: pd.DataFrame) -> pd.DataFrame:
    if len(steps) == 0:
        steps["drawdown"] = []
        return steps
    equity = steps["equity"].to_numpy(dtype=float)
    peak = np.maximum.accumulate(equity)
    dd = equity / np.maximum(peak, 1e-12) - 1.0
    steps = steps.copy()
    steps["drawdown"] = dd
    return steps
def plot_backtest(steps: pd.DataFrame, out_dir: str = ".", prefix: str = "bt", show: bool = False):
    """
    Save plots to out_dir with filenames like:
      bt_equity.png, bt_drawdown.png, bt_pos.png, bt_reward.png, bt_actions.png, bt_costs.png
    """
    os.makedirs(out_dir, exist_ok=True)

    if len(steps) == 0:
        print("[plot] steps is empty, skip plotting.")
        return

    steps = add_drawdown_column(steps)

    x = steps["step"].to_numpy()

    # 1) Equity / NAV
    plt.figure()
    plt.plot(x, steps["equity"].to_numpy(dtype=float), label="equity")
    plt.plot(x, steps["nav"].to_numpy(dtype=float), label="nav")
    plt.title("Equity / NAV")
    plt.xlabel("step")
    plt.ylabel("value")
    plt.legend()
    f1 = os.path.join(out_dir, f"{prefix}_equity.png")
    plt.savefig(f1, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()

    # 2) Drawdown
    plt.figure()
    plt.plot(x, steps["drawdown"].to_numpy(dtype=float))
    plt.title("Drawdown")
    plt.xlabel("step")
    plt.ylabel("drawdown")
    f2 = os.path.join(out_dir, f"{prefix}_drawdown.png")
    plt.savefig(f2, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()

    # 3) Position
    plt.figure()
    plt.plot(x, steps["pos"].to_numpy(dtype=int))
    plt.title("Position")
    plt.xlabel("step")
    plt.ylabel("pos (-1/0/1)")
    f3 = os.path.join(out_dir, f"{prefix}_pos.png")
    plt.savefig(f3, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()

    # 4) Reward (and rolling mean)
    plt.figure()
    r = steps["reward"].to_numpy(dtype=float)
    plt.plot(x, r, label="reward")
    # rolling mean for readability
    win = max(5, min(200, len(r)//20))
    roll = pd.Series(r).rolling(win).mean().to_numpy()
    plt.plot(x, roll, label=f"reward_roll{win}")
    plt.title("Reward per Step")
    plt.xlabel("step")
    plt.ylabel("reward")
    plt.legend()
    f4 = os.path.join(out_dir, f"{prefix}_reward.png")
    plt.savefig(f4, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()

    # 5) Action distribution
    plt.figure()
    acts = steps["action"].to_numpy(dtype=int)
    vals, cnts = np.unique(acts, return_counts=True)
    plt.bar([str(v) for v in vals], cnts)
    plt.title("Action Counts")
    plt.xlabel("action (-1/0/1)")
    plt.ylabel("count")
    f5 = os.path.join(out_dir, f"{prefix}_actions.png")
    plt.savefig(f5, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()

    # 6) Cumulative turnover / fee / hold_cost
    plt.figure()
    plt.plot(x, steps["turnover"].cumsum().to_numpy(dtype=float), label="cum_turnover")
    plt.plot(x, steps["fee"].cumsum().to_numpy(dtype=float), label="cum_fee")
    plt.plot(x, steps["hold_cost"].cumsum().to_numpy(dtype=float), label="cum_hold_cost")
    plt.title("Cumulative Turnover / Costs")
    plt.xlabel("step")
    plt.ylabel("cumulative")
    plt.legend()
    f6 = os.path.join(out_dir, f"{prefix}_costs.png")
    plt.savefig(f6, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()

    print("[plot] saved:")
    print(" ", f1)
    print(" ", f2)
    print(" ", f3)
    print(" ", f4)
    print(" ", f5)
    print(" ", f6)

if __name__ == "__main__":
    POLICY_PATH = "model/ppo_policy.pt"
    TEST_CSV = "data/data_e.csv"         
           

    ENV_CFG = TradingEnvConfig(
        fee_bps=0.1,
        hold_cost_bps=0.1,
        max_episode_steps=None,        
        random_start=False,
        start_index=1,
        seed=42,
    )
    CAPITAL = 10.0
    OUT_CSV = "result/backtest_steps.csv"

    df_test = pd.read_csv(TEST_CSV)
    split_idx = int(len(df_test)*0.8)
    df_test = df_test[split_idx:].copy()
    FEATURE_COLS = df_test.columns.drop(["regime","timestamp","close_5m"]).tolist()
    X_test, close_test = df_to_arrays(df_test, FEATURE_COLS, close_col="close_5m", dropna=True)

    env = TradingEnv(X_test, close_test, ENV_CFG)

    cfg = PPOConfig()
    agent = PPOAgent(env, cfg)
    agent.load(POLICY_PATH)

    steps = backtest(agent, env, capital=CAPITAL)
    steps.to_csv(OUT_CSV, index=False)
    PLOT_DIR = "plots"
    plot_backtest(steps, out_dir=PLOT_DIR, prefix="ppo_test", show=False)
    s = summarize(steps)

    print("==== Backtest Summary ====")
    for k, v in s.items():
        print(f"{k:>14s}: {v}")

    print(f"Saved to: {OUT_CSV}")


    print("==== Buy&Hold Summary ====")
    bh_steps = buy_and_hold(df_test, close_col="close_5m", capital=CAPITAL)
    s_bh = summarize(bh_steps.rename(columns={"equity": "equity"}))  # summarize expects 'equity'
    for k, v in s_bh.items():
        print(f"{k:>14s}: {v}")
    

    # Plot equity comparison
    os.makedirs(PLOT_DIR, exist_ok=True)
    plt.figure()
    plt.plot(steps["step"], steps["equity"], label="ppo_equity")
    if len(bh_steps) > 0:
        plt.plot(bh_steps["step"], bh_steps["equity"], label="buy_hold_equity")
    plt.title("Equity Comparison")
    plt.xlabel("step")
    plt.ylabel("equity")
    plt.legend()
    plt.savefig(os.path.join(PLOT_DIR, "ppo_vs_buyhold_equity.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved equity comparison to: {os.path.join(PLOT_DIR, 'ppo_vs_buyhold_equity.png')}")

