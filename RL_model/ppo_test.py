# backtest_run.py
from __future__ import annotations
import numpy as np
import pandas as pd
import torch

from ppo import (
    df_to_arrays,
    TradingEnv,
    TradingEnvConfig,
    PPOAgent,
    PPOConfig,
)

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
        cum_reward += float(reward)
        equity = float(np.exp(cum_reward))       
        nav = equity * float(capital)

        rows.append({
            "step": step,
            "t": t,
            "close": float(env.close[t]),
            "ret_log": float(info["ret"]),
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

    equity = steps["equity"].to_numpy(dtype=float)
    out["equity_end"] = float(equity[-1])
    out["total_return"] = float(equity[-1] / (equity[0] + 1e-12) - 1.0)
    out["max_drawdown"] = max_drawdown(equity)

    if len(equity) >= 3:
        step_ret = equity[1:] / np.maximum(equity[:-1], 1e-12) - 1.0
        out["sharpe_step"] = float(np.mean(step_ret) / (np.std(step_ret) + 1e-12))
    else:
        out["sharpe_step"] = 0.0

    out["turnover_sum"] = float(steps["turnover"].sum())
    out["fee_sum"] = float(steps["fee"].sum())
    out["hold_cost_sum"] = float(steps["hold_cost"].sum())
    out["reward_sum"] = float(steps["reward"].sum())
    return out


if __name__ == "__main__":
    POLICY_PATH = "ppo_policy.pt"
    TEST_CSV = "BTCUSDT_combined_klines_20210201_20260201_states4_labeled.csv"
    # 必须与训练时使用的特征列完全一致
    FEATURE_COLS = [
        "volume_5m", "log_ret_1_5m", "ema_ratio_9_21_5m",
        "macd_hist_5m", "adx_5m", "atr_norm_5m", "bb_width_5m",
        "rsi_14_5m", "volume_zscore_50_5m", "state"
    ]

    ENV_CFG = TradingEnvConfig(
        fee_bps=2.0,
        hold_cost_bps=0.0,
        max_episode_steps=None,        
        random_start=False,
        start_index=1,
        seed=42,
    )
    CAPITAL = 1.0
    OUT_CSV = "backtest_steps.csv"

    df_test = pd.read_csv(TEST_CSV)
    # 必须使用与训练时相同的 close_col
    X_test, close_test = df_to_arrays(df_test, FEATURE_COLS, close_col="close_5m", dropna=True)

    env = TradingEnv(X_test, close_test, ENV_CFG)

    cfg = PPOConfig()
    agent = PPOAgent(env, cfg)
    agent.load(POLICY_PATH)

    steps = backtest(agent, env, capital=CAPITAL)
    steps.to_csv(OUT_CSV, index=False)

    s = summarize(steps)
    print("==== Backtest Summary ====")
    for k, v in s.items():
        print(f"{k:>14s}: {v}")

    print(f"Saved to: {OUT_CSV}")