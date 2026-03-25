# backtest_run.py
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from typing import Any
from dataclasses import dataclass
from ppo import (
    df_to_arrays,
    TradingEnv,
    TradingEnvConfig,
    PPOAgent,
    PPOConfig,
)
from split_bounds import (
    try_load_split_config,
    parse_split_bounds,
    effective_rl_test_start,
    rl_test_mask,
)

@dataclass
class PPOTestConfig:
    agent_path:str
    test_csv:str = "data/data_e.csv"
    close_col: str = "close_5m"
    drop_cols: tuple[str,...] = ("regime","timestamp","close_5m")
    train_ratio: float = 0.8
    split_config_path: str | None = None  # optional explicit split_config.json path
    use_test_split:bool = True
    capital:float = 1.0
    out_csv: str | None = "result/backtest_steps.csv"
    plot_dir: str | None = None
    plot_prefix:str = "ppo_test"
    compare_buy_hold:bool = True
    buy_hold_plot_name: str = "agent_vs_buyhold.png"


def buy_and_hold(df_test: pd.DataFrame, close_col: str = "close_5m", capital: float = 1.0) -> pd.DataFrame:
    close = pd.to_numeric(df_test[close_col], errors="coerce").to_numpy(dtype=float)
    close = close[np.isfinite(close)]
    if len(close) < 2:
        return pd.DataFrame()

    step_ret = close[1:] / np.maximum(close[:-1], 1e-12) - 1.0 

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
    act_idx = int(torch.argmax(logits, dim=-1).item())     
    return int(agent.IDX_TO_ACT[act_idx])                  

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
        simple_ret = float(info["ret"])
        pos = int(info["pos"])     
        fee = float(info["fee"])
        hold = float(info["hold_cost"])
        step_ret = pos * simple_ret - fee - hold
        equity = (rows[-1]["equity"] if rows else 1.0) * (1.0 + step_ret)   
        nav = equity * float(capital)
        cum_reward += float(reward)

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

# Build Test Env:

def _timestamp_range_meta(df: pd.DataFrame, ts_col: str = "timestamp") -> dict[str, str | None]:
    if ts_col not in df.columns or len(df) == 0:
        return {"min": None, "max": None}
    ts = pd.to_datetime(df[ts_col], utc=True, errors="coerce").dropna()
    if ts.empty:
        return {"min": None, "max": None}
    return {"min": ts.min().isoformat(), "max": ts.max().isoformat()}


def _build_eval_split(df: pd.DataFrame, test_cfg: PPOTestConfig, ts_col: str = "timestamp") -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Prefer calendar split via split_config (same RL test window rule as MoE):
      test = ts >= max(test_start, rl_train_end)
    Fallback to ratio split when config/timestamp is unavailable.
    """
    if not test_cfg.use_test_split:
        return df.copy(), {"split_mode": "full_dataset", "test_rows": len(df)}

    raw = (
        try_load_split_config(test_cfg.split_config_path)
        if test_cfg.split_config_path
        else try_load_split_config()
    )
    if raw is not None and ts_col in df.columns:
        try:
            bounds = parse_split_bounds(raw)
            m_te = rl_test_mask(df, bounds, ts_col=ts_col)
            df_te = df.loc[m_te].copy()
            if len(df_te) > 0:
                meta = {
                    "split_mode": "split_config.json",
                    "test_start_effective": effective_rl_test_start(bounds).isoformat(),
                    "test_rows": len(df_te),
                    "test_timestamp_range": _timestamp_range_meta(df_te, ts_col),
                }
                return df_te, meta
        except (KeyError, ValueError, TypeError):
            pass

    split_idx = int(len(df) * test_cfg.train_ratio)
    df_te = df.iloc[split_idx:].copy()
    meta = {
        "split_mode": "train_ratio_fallback",
        "train_ratio": test_cfg.train_ratio,
        "test_rows": len(df_te),
        "test_timestamp_range": _timestamp_range_meta(df_te, ts_col),
    }
    return df_te, meta

def build_test_env(env_cfg,test_cfg):
    df = pd.read_csv(test_cfg.test_csv)
    df, split_meta = _build_eval_split(df, test_cfg, ts_col="timestamp")
    print(f"[ppo_test] eval split meta: {split_meta}")

    feature_cols = df.columns.drop(list(test_cfg.drop_cols)).to_list()
    X_test,close_test = df_to_arrays(df,feature_cols,close_col=test_cfg.close_col,dropna=True)
    eval_env_cfg = TradingEnvConfig(
        fee_bps=env_cfg.fee_bps,
        hold_cost_bps=env_cfg.hold_cost_bps,
        max_episode_steps=None,
        random_start=False,
        start_index=env_cfg.start_index,
        seed=env_cfg.seed,
    )
    env = TradingEnv(X_test,close_test,eval_env_cfg)
    return env,df


def run_ppo_backtest(
    agent_path: str,
    env_cfg: TradingEnvConfig,
    test_cfg: PPOTestConfig | None = None,
) -> dict[str, Any]:
    if test_cfg is None:
        test_cfg = PPOTestConfig(agent_path=agent_path)

    env, df_test = build_test_env(env_cfg, test_cfg)

    agent = PPOAgent(env, PPOConfig())
    agent.load(agent_path)

    steps = backtest(agent, env, capital=test_cfg.capital)
    summary = summarize(steps)

    bh_steps = pd.DataFrame()
    bh_summary: dict[str, float] = {}
    if test_cfg.compare_buy_hold:
        bh_steps = buy_and_hold(df_test, close_col=test_cfg.close_col, capital=test_cfg.capital)
        bh_summary = summarize(bh_steps)

    if test_cfg.out_csv:
        os.makedirs(os.path.dirname(test_cfg.out_csv) or ".", exist_ok=True)
        steps.to_csv(test_cfg.out_csv, index=False)

    if test_cfg.plot_dir:
        plot_backtest(steps, out_dir=test_cfg.plot_dir, prefix=test_cfg.plot_prefix, show=False)

        if test_cfg.compare_buy_hold and len(bh_steps) > 0:
            os.makedirs(test_cfg.plot_dir, exist_ok=True)
            plt.figure()
            plt.plot(steps["step"], steps["equity"], label="ppo_equity")
            plt.plot(bh_steps["step"], bh_steps["equity"], label="buy_hold_equity")
            plt.title("Equity Comparison")
            plt.xlabel("step")
            plt.ylabel("equity")
            plt.legend()
            compare_path = os.path.join(test_cfg.plot_dir, test_cfg.buy_hold_plot_name)
            plt.savefig(compare_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"Saved equity comparison to: {compare_path}")

    return {
        "steps": steps,
        "summary": summary,
        "buy_hold_steps": bh_steps,
        "buy_hold_summary": bh_summary,
    }


if __name__ == "__main__":
    policy_path = "model/ppo_policy.pt"

    env_cfg = TradingEnvConfig(
        fee_bps=2.5,
        hold_cost_bps=2.5,
        max_episode_steps=None,
        random_start=False,
        start_index=1,
        seed=42,
    )

    test_cfg = PPOTestConfig(
        agent_path="model/ppo_policy.pt",
        test_csv="data/data_e.csv",
        close_col="close_5m",
        drop_cols=("regime", "timestamp", "close_5m"),
        train_ratio=0.8,
        use_test_split=True,
        capital=10.0,
        out_csv="result/backtest_steps.csv",
        plot_dir="plots",
        plot_prefix="ppo_test",
        compare_buy_hold=True,
    )

    result = run_ppo_backtest(policy_path, env_cfg, test_cfg)

    print("==== Backtest Summary ====")
    for k, v in result["summary"].items():
        print(f"{k:>14s}: {v}")

    if result["buy_hold_summary"]:
        print("==== Buy&Hold Summary ====")
        for k, v in result["buy_hold_summary"].items():
            print(f"{k:>14s}: {v}")

