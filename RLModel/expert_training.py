from __future__ import annotations
import os
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import optuna
from ppo import (
    df_to_arrays,
    TradingEnv,
    TradingEnvConfig,
    PPOAgent,
    PPOConfig,
)


@dataclass
class DataConfig:
    csv_path: str = "data/data_e.csv"
    close_col: str = "close_5m"
    regime_bool_cols: Tuple[str, ...] = ("state_0", "state_1", "state_2", "state_3")
    drop_cols: Tuple[str, ...] = ("timestamp","regime","close_5m", "state_0", "state_1", "state_2", "state_3")
    train_ratio: float = 0.8

def train_test_split_df(df: pd.DataFrame, train_ratio: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    split_idx = int(len(df) * train_ratio)
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()
def build_feature_cols(df: pd.DataFrame, drop_cols: Tuple[str, ...]) -> List[str]:
    return df.columns.drop(list(drop_cols)).tolist()


def extract_regime_segments(df,regime_bool_cols,r:int):
    col = regime_bool_cols[r]
    sub = df[df[col].astype(bool)].copy()
    sub = sub.reset_index(drop=True)
    return sub


@torch.no_grad()

# Action
def act_deterministic(agent: PPOAgent, obs_np: np.ndarray) -> int:
    obs = torch.tensor(obs_np, dtype=torch.float32, device=agent.device)
    logits, _ = agent.model(obs.unsqueeze(0))
    act_idx = int(torch.argmax(logits, dim=-1).item()) # 0/1/2
    return int(agent.IDX_TO_ACT[act_idx])                  

# Backtest
def backtest_deterministic(agent:PPOAgent,env:TradingEnv,capital=1.0):
    obs = env.reset()
    done = False
    rows = []

    while not done:
        action = act_deterministic(agent,obs)
        next_obs,reward,done,info = env.step(action)

        t = int(info["t"])
        simple_ret = (env.close[t] + 1e-12)/(env.close[t-1] + 1e-12)-1.0
        pos = int(info["pos"])
        fee = float(info["fee"])
        hold = float(info["hold_cost"])
        step_ret = pos * float(simple_ret) - fee - hold
        equity_prev = rows[-1]["equity"] if rows else 1.0
        equity = equity_prev * (1.0 + step_ret)
        nav = equity*float(capital)

        rows.append(
            {
                "t": t,
                "step_ret": float(step_ret),
                "equity": float(equity),
                "nav": float(nav),
                "pos": int(pos),
                "action": int(action),
                "fee": float(fee),
                "hold_cost": float(hold),
                "turnover": float(info["turnover"]),
                "reward": float(reward),
            }
        )
        obs = next_obs

    return pd.DataFrame(rows)

# MDD
def max_drawdown(equity: np.ndarray) -> float:
    peak = equity[0]
    mdd = 0.0
    for x in equity:
        peak = max(peak, x)
        dd = x / (peak + 1e-12) - 1.0
        mdd = min(mdd, dd)
    return float(mdd)


# Step Summary

def summarize_steps(steps: pd.DataFrame) -> Dict[str, float]:
    if len(steps) == 0:
        return {"bars": 0, "equity_end": 1.0, "total_return": 0.0, "mdd": 0.0, "sharpe": 0.0}

    eq = steps["equity"].to_numpy(dtype=float)
    rets = steps["step_ret"].to_numpy(dtype=float)
    rets = rets[np.isfinite(rets)]

    sharpe = float(np.mean(rets) / (np.std(rets) + 1e-12)) if len(rets) > 10 else 0.0
    mdd = max_drawdown(eq)

    return {
        "bars": int(len(steps)),
        "equity_end": float(eq[-1]),
        "total_return": float(eq[-1] - 1.0),
        "mdd": float(mdd),
        "sharpe": sharpe,
        "turnover_sum": float(steps["turnover"].sum()),
        "fee_sum": float(steps["fee"].sum()),
        "hold_cost_sum": float(steps["hold_cost"].sum()),
    }

# Optuna Objective Score
def score_summary(s:Dict[str,float]):
    return float(1.0*s["total_return"] + 0.1*s["sharpe"] + 2.0*s["mdd"] - 0.0001 * s["turnover_sum"])
# Optuna Objective per regime expert
def make_ppo_cfg_from_trial(trial: optuna.Trial) -> PPOConfig:
    cfg = PPOConfig()
    cfg.lr = trial.suggest_float("lr", 1e-5, 3e-3, log=True)
    cfg.gamma = trial.suggest_float("gamma", 0.90, 0.999)
    cfg.gae_lambda = trial.suggest_float("gae_lambda", 0.80, 0.99)
    cfg.clip_eps = trial.suggest_float("clip_eps", 0.05, 0.3)
    cfg.ent_coef = trial.suggest_float("ent_coef", 0.0, 0.2)
    cfg.vf_coef = trial.suggest_float("vf_coef", 0.2, 1.0)
    cfg.max_grad_norm = trial.suggest_float("max_grad_norm", 0.2, 1.0)

    cfg.hidden = trial.suggest_categorical("hidden", [64, 128, 256])
    cfg.rollout_steps = trial.suggest_categorical("rollout_steps", [512, 1024, 2048])
    cfg.minibatch_size = trial.suggest_categorical("minibatch_size", [64, 128, 256])
    cfg.update_epochs = trial.suggest_int("update_epochs", 3, 10)

    return cfg
class _FrozenTrial:
    def __init__(self, params: Dict[str, object]):
        self._params = params

    def suggest_float(self, name, low, high, log=False):
        return float(self._params[name])

    def suggest_int(self, name, low, high, step=1):
        return int(self._params[name])

    def suggest_categorical(self, name, choices):
        return self._params[name]
def make_ppo_cfg_from_params(params: Dict[str, object]) -> PPOConfig:
    frozen = _FrozenTrial(params)
    return make_ppo_cfg_from_trial(frozen)

def train_one_expert(df_train,df_val,feature_cols:List[str],data_cfg,env_cfg,ppo_cfg,total_updates:int,seed:int = 42)-> Tuple[PPOAgent,Dict[str,float]]:
    X_train,close_train = df_to_arrays(df_train,feature_cols,close_col=data_cfg.close_col,dropna=True)
    env_train = TradingEnv(X_train, close_train, env_cfg)

    torch.manual_seed(seed)
    np.random.seed(42)
    agent = PPOAgent(env_train,ppo_cfg)
    agent.train(total_updates=total_updates,log_every=max(10,total_updates//10),plot_path=f"RLModel/train_plots/expert_tmp.png")
    X_val,close_val = df_to_arrays(df_val,feature_cols,close_col=data_cfg.close_col,dropna = True)
    env_val = TradingEnv(X_val,close_val,TradingEnvConfig(fee_bps=env_cfg.fee_bps,hold_cost_bps=env_cfg.hold_cost_bps,max_episode_steps=None,random_start=False,seed=seed,start_index=1))
    steps = backtest_deterministic(agent,env_val,capital=1.0)
    summ = summarize_steps(steps)
    return agent, summ

# Different Regime
def tune_expert_for_regime(
    r: int,
    df_train_all: pd.DataFrame,
    df_val_all: pd.DataFrame,
    feature_cols: List[str],
    data_cfg: DataConfig,
    fixed_env_cfg: TradingEnvConfig,
    n_trials: int = 25,
    total_updates: int = 120,
    study_dir: str = "result/optuna",
) -> Dict[str, object]:
    os.makedirs(study_dir, exist_ok=True)

    df_train = extract_regime_segments(df_train_all, data_cfg.regime_bool_cols, r)
    df_val = extract_regime_segments(df_val_all, data_cfg.regime_bool_cols, r)
    if len(df_train) < 2000 or len(df_val) < 500:
        print(f"[regime {r}] too few samples: train={len(df_train)} val={len(df_val)} (skip or reduce thresholds)")
        # still return something, but mark as skipped
        return {"regime": r, "skipped": True, "train_rows": len(df_train), "val_rows": len(df_val)}

    def objective(trial: optuna.Trial) -> float:
        ppo_cfg = make_ppo_cfg_from_trial(trial)
        # speed: fewer updates during search
        agent, summ = train_one_expert(
            df_train=df_train,
            df_val=df_val,
            feature_cols=feature_cols,
            data_cfg=data_cfg,
            env_cfg=fixed_env_cfg,
            ppo_cfg=ppo_cfg,
            total_updates=total_updates,
            seed=42,
        )
        sc = score_summary(summ)
        trial.set_user_attr("summary", summ)
        return sc

    study = optuna.create_study(direction="maximize", study_name=f"expert_regime_{r}")
    study.optimize(objective, n_trials=n_trials)

    best = study.best_trial
    best_params = dict(best.params)
    best_summary = best.user_attrs.get("summary", {})

    best_ppo_cfg = make_ppo_cfg_from_trial(_FrozenTrial(best_params))
    agent, val_summ = train_one_expert(
        df_train=df_train,
        df_val=df_val,
        feature_cols=feature_cols,
        data_cfg=data_cfg,
        ppo_cfg=best_ppo_cfg,
        total_updates=max(total_updates, 200),
        seed=42,
    )

    return {
        "regime": r,
        "skipped": False,
        "train_rows": len(df_train),
        "val_rows": len(df_val),
        "study_best_value": float(best.value),
        "best_params": best_params,
        "best_summary": best_summary,
        "final_val_summary": val_summ,
        "agent": agent,
        "env_cfg": fixed_env_cfg,
        "ppo_cfg": best_ppo_cfg,
    }


# Save the model
def save_expert_bundle(
    out_root: str,
    r: int,
    agent: PPOAgent,
    env_cfg: TradingEnvConfig,
    best_params: Dict[str, object],
    final_val_summary: Dict[str, float],
):
    d = os.path.join(out_root, f"regime_{r}")
    os.makedirs(d, exist_ok=True)

    model_path = os.path.join(d, "ppo_policy.pt")
    agent.save(model_path)

    meta = {
        "regime": r,
        "fixed_env_cfg": {
            "fee_bps": env_cfg.fee_bps,
            "hold_cost_bps": env_cfg.hold_cost_bps,
            "max_episode_steps": env_cfg.max_episode_steps,
            "random_start": env_cfg.random_start,
            "seed": env_cfg.seed,
            "start_index": env_cfg.start_index,
        },
        "best_params": best_params,
        "final_val_summary": final_val_summary,
    }

    with open(os.path.join(d, "best_params.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[save] regime {r}: {model_path}")
    return model_path



def main():
    data_cfg = DataConfig(
        csv_path="data/data_e.csv",
        close_col="close_5m",
        regime_bool_cols=("state_0", "state_1", "state_2", "state_3"),
        drop_cols=("timestamp", "regime", "close_5m", "state_0", "state_1", "state_2", "state_3"),
        train_ratio=0.8,
    )

    out_root = "model/experts"
    os.makedirs(out_root, exist_ok=True)

    n_trials = 25
    search_updates = 120

    fixed_env_cfg = TradingEnvConfig(
        fee_bps=0.5,
        hold_cost_bps=0.1,
        max_episode_steps=10000,
        random_start=True,
        seed=42,
        start_index=1,
    )

    df = pd.read_csv(data_cfg.csv_path)
    print("[check] state counts:", {c: int(df[c].astype(bool).sum()) for c in data_cfg.regime_bool_cols})

    feature_cols = build_feature_cols(df, data_cfg.drop_cols)

    df_train_all, df_test_all = train_test_split_df(df, data_cfg.train_ratio)
    df_val_all, df_holdout_all = train_test_split_df(df_test_all, 0.5)
    print(f"[data] train={len(df_train_all)} val={len(df_val_all)} holdout={len(df_holdout_all)}")

    all_results = {}

    for r in [0, 1, 2, 3]:
        print(f"\n========== TUNE expert regime={r} ==========")
        res = tune_expert_for_regime(
            r=r,
            df_train_all=df_train_all,
            df_val_all=df_val_all,
            feature_cols=feature_cols,
            data_cfg=data_cfg,
            fixed_env_cfg=fixed_env_cfg,
            n_trials=n_trials,
            total_updates=search_updates,
            study_dir="result/optuna",
        )
        all_results[r] = res

        if not res.get("skipped", False):
            save_expert_bundle(
                out_root=out_root,
                r=r,
                agent=res["agent"],
                env_cfg=fixed_env_cfg,
                best_params=res["best_params"],
                final_val_summary=res["final_val_summary"],
            )

    summary = {}
    for r, res in all_results.items():
        if res.get("skipped", False):
            summary[str(r)] = {
                "skipped": True,
                "train_rows": res["train_rows"],
                "val_rows": res["val_rows"],
            }
        else:
            summary[str(r)] = {
                "skipped": False,
                "train_rows": res["train_rows"],
                "val_rows": res["val_rows"],
                "fixed_env_cfg": {
                    "fee_bps": fixed_env_cfg.fee_bps,
                    "hold_cost_bps": fixed_env_cfg.hold_cost_bps,
                    "max_episode_steps": fixed_env_cfg.max_episode_steps,
                    "random_start": fixed_env_cfg.random_start,
                    "seed": fixed_env_cfg.seed,
                    "start_index": fixed_env_cfg.start_index,
                },
                "study_best_value": res["study_best_value"],
                "best_params": res["best_params"],
                "final_val_summary": res["final_val_summary"],
            }

    summary_path = os.path.join(out_root, "experts_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[done] saved summary: {summary_path}")


if __name__ == "__main__":
    main()




