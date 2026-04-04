"""
Unified runner for five RL experiments (same train/test calendar as MoE when split_config is used).

  1. PPO-Base          — tech features only (no HMM/LSTM regime columns)
  2. PPO+HMM           — tech + hmm_predicted_state_0..3 (HMM next-bar label / target)
  3. PPO+HMM+LSTM      — tech + hmm_predicted_state_0..3 + lstm_predicted_state_0..3
  4. MoE+HMM           — routed by HMM predicted one-hot; obs without those
  5. MoE+HMM+LSTM      — routed by LSTM one-hot; obs includes HMM predicted one-hot

Requires ``data_e.csv`` from ``data_process.data_engineering`` (``pred_state_source="both"``):
``timestamp``, ``close_5m``, ``hmm_predicted_state_0``..``_3``, ``lstm_predicted_state_0``..``_3``.

After ``run_experiments``, ``result_root`` also gets ``equity_overlay.png``,
``comparison_metrics.csv`` (Sharpe / MDD / Return / Turnover), and
``per_regime_sharpe_long.csv`` / ``per_regime_sharpe_wide.csv`` (HMM regime–wise Sharpe).

Usage (from ``RLModel/``)::

    python experiment_runner.py --csv data/data_e.csv --all
    python experiment_runner.py --csv data/data_e.csv --only ppo_base,mo_e_hmm_lstm
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_process import resample_bars
from ppo import PPOAgent, PPOConfig, TradingEnv, TradingEnvConfig, df_to_arrays
from ppo_test import PPOTestConfig, buy_and_hold, run_ppo_backtest, summarize
from seed_utils import set_training_seed
from moe_hard import (
    HardMoEAgent,
    MoEDataConfig,
    _prepare_moe_arrays,
    drop_cols_existing,
    moe_train_test_split,
    run_moe_backtest,
    save_moe_bundle,
    train_moe_experts,
)

N_REGIMES = 4
# Matches data_process one-hot prefix for ``target`` → ``hmm_predicted_state``
HMM_OH = tuple(f"hmm_predicted_state_{i}" for i in range(N_REGIMES))
LSTM_OH = tuple(f"lstm_predicted_state_{i}" for i in range(N_REGIMES))

ExperimentId = Literal[
    "ppo_base",
    "ppo_hmm",
    "ppo_hmm_lstm",
    "moe_hmm",
    "moe_hmm_lstm",
]

EXPERIMENT_ORDER: tuple[ExperimentId, ...] = (
    "ppo_base",
    "ppo_hmm",
    "ppo_hmm_lstm",
    "moe_hmm",
    "moe_hmm_lstm",
)


def _cols_starting(columns: pd.Index, prefix: str) -> list[str]:
    return [c for c in columns if str(c).startswith(prefix)]


def drop_cols_for_ppo_base(df: pd.DataFrame) -> tuple[str, ...]:
    """Exclude all regime / state signals from observations."""
    fixed = [
        "timestamp",
        "close_5m",
        "hmm_predicted_state",
        "lstm_predicted_state",
    ]
    extra = _cols_starting(df.columns, "hmm_predicted_state_") + _cols_starting(
        df.columns, "lstm_predicted_state_"
    )
    seen: set[str] = set()
    out: list[str] = []
    for c in fixed + extra:
        if c in df.columns and c not in seen:
            seen.add(c)
            out.append(c)
    return tuple(out)


def drop_cols_for_ppo_hmm(df: pd.DataFrame) -> tuple[str, ...]:
    """Tech + ``hmm_predicted_state_*``; drop LSTM columns."""
    fixed = [
        "timestamp",
        "close_5m",
        "hmm_predicted_state",
        "lstm_predicted_state",
    ]
    extra = _cols_starting(df.columns, "lstm_predicted_state_")
    seen: set[str] = set()
    out: list[str] = []
    for c in fixed + extra:
        if c in df.columns and c not in seen:
            seen.add(c)
            out.append(c)
    return tuple(out)


def drop_cols_for_ppo_hmm_lstm(df: pd.DataFrame) -> tuple[str, ...]:
    """Tech + ``hmm_predicted_state_*`` + ``lstm_predicted_state_*`` (raw scalars only dropped)."""
    fixed = [
        "timestamp",
        "close_5m",
        "hmm_predicted_state",
        "lstm_predicted_state",
    ]
    seen: set[str] = set()
    out: list[str] = []
    for c in fixed:
        if c in df.columns and c not in seen:
            seen.add(c)
            out.append(c)
    return tuple(out)


def moe_drop_cols_hmm() -> tuple[str, ...]:
    return ("timestamp", "close_5m", *HMM_OH)


def moe_drop_cols_hmm_lstm(df: pd.DataFrame) -> tuple[str, ...]:
    """Observations: tech + ``hmm_predicted_state_*``; routing: LSTM one-hot (excluded from X)."""
    fixed = [
        "timestamp",
        "close_5m",
        *LSTM_OH,
        "hmm_predicted_state",
        "lstm_predicted_state",
    ]
    seen: set[str] = set()
    out: list[str] = []
    for c in fixed:
        if c in df.columns and c not in seen:
            seen.add(c)
            out.append(c)
    return tuple(out)


def _split_cfg_stub(csv_path: str, train_ratio: float, split_config_path: str | None) -> MoEDataConfig:
    """Minimal config for ``moe_train_test_split`` (only path / ratio / split matter)."""
    return MoEDataConfig(
        csv_path=csv_path,
        train_ratio=train_ratio,
        split_config_path=split_config_path,
    )


def _feature_cols(df: pd.DataFrame, drop_cols: tuple[str, ...], close_col: str) -> list[str]:
    cols = [c for c in df.columns if c not in set(drop_cols) and c != close_col]
    return cols


def _validate(exp_id: ExperimentId, df: pd.DataFrame) -> None:
    for c in ("timestamp", "close_5m"):
        if c not in df.columns:
            raise ValueError(f"CSV missing required column {c!r}")
    if exp_id in ("ppo_hmm", "ppo_hmm_lstm", "moe_hmm", "moe_hmm_lstm"):
        missing = [c for c in HMM_OH if c not in df.columns]
        if missing:
            raise ValueError(f"{exp_id}: missing HMM predicted-state columns {missing}")
    if exp_id in ("ppo_hmm_lstm", "moe_hmm_lstm"):
        missing = [c for c in LSTM_OH if c not in df.columns]
        if missing:
            raise ValueError(f"{exp_id}: missing LSTM predicted-state columns {missing}")


def _ppo_drop_cols(exp_id: ExperimentId, df: pd.DataFrame) -> tuple[str, ...]:
    if exp_id == "ppo_base":
        return drop_cols_for_ppo_base(df)
    if exp_id == "ppo_hmm":
        return drop_cols_for_ppo_hmm(df)
    if exp_id == "ppo_hmm_lstm":
        return drop_cols_for_ppo_hmm_lstm(df)
    raise ValueError(exp_id)


def _moe_data_cfg(exp_id: ExperimentId, df: pd.DataFrame, csv_path: str, train_ratio: float, split_config_path: str | None) -> MoEDataConfig:
    if exp_id == "moe_hmm":
        return MoEDataConfig(
            csv_path=csv_path,
            close_col="close_5m",
            regime_cols=HMM_OH,
            drop_cols=moe_drop_cols_hmm(),
            train_ratio=train_ratio,
            split_config_path=split_config_path,
        )
    if exp_id == "moe_hmm_lstm":
        return MoEDataConfig(
            csv_path=csv_path,
            close_col="close_5m",
            regime_cols=LSTM_OH,
            drop_cols=moe_drop_cols_hmm_lstm(df),
            train_ratio=train_ratio,
            split_config_path=split_config_path,
        )
    raise ValueError(exp_id)


@dataclass
class RunnerConfig:
    csv_path: str = "data/data_e.csv"
    close_col: str = "close_5m"
    train_ratio: float = 0.8
    split_config_path: str | None = None
    resample_minutes: int = 0  # 0 = no resample; 15 = resample 5m bars to 15m
    total_updates: int = 200
    log_every: int = 10
    capital: float = 10.0
    model_root: str = "model/experiments"
    result_root: str = "result/experiments"
    meta_root: str = "result/experiment_meta"
    env_cfg: TradingEnvConfig = field(
        default_factory=lambda: TradingEnvConfig(
            fee_bps=5.0,
            hold_cost_bps=0.0,
            max_episode_steps=10000,
            random_start=True,
            start_index=1,
            seed=42,
        )
    )
    ppo_cfg: PPOConfig = field(default_factory=PPOConfig)
    eval_env_cfg: TradingEnvConfig = field(
        default_factory=lambda: TradingEnvConfig(
            fee_bps=5.0,
            hold_cost_bps=0.0,
            max_episode_steps=None,
            random_start=False,
            start_index=1,
            seed=42,
        )
    )


def _sharpe_from_step_returns(step_ret: np.ndarray) -> float:
    r = step_ret.astype(float, copy=False)
    r = r[np.isfinite(r)]
    if len(r) < 3:
        return float("nan")
    return float(np.mean(r) / (np.std(r) + 1e-12))


def _per_hmm_regime_sharpe_table(steps: pd.DataFrame) -> list[dict[str, Any]]:
    if steps.empty or "hmm_regime_id" not in steps.columns or "step_ret" not in steps.columns:
        return []
    rows: list[dict[str, Any]] = []
    for rid, g in steps.groupby("hmm_regime_id"):
        sr = g["step_ret"].to_numpy(dtype=float)
        rows.append(
            {
                "hmm_regime_id": int(rid),
                "sharpe": _sharpe_from_step_returns(sr),
                "n_bars": int(len(g)),
            }
        )
    rows.sort(key=lambda x: x["hmm_regime_id"])
    return rows


def write_aggregate_backtest_reports(
    cfg: RunnerConfig,
    experiment_ids: list[ExperimentId],
) -> None:
    """
    After experiments, write under ``cfg.result_root``:

    - ``equity_overlay.png`` — all models' test equity vs buy & hold
    - ``comparison_metrics.csv`` — Sharpe / MDD / Return / Turnover per experiment
    - ``per_regime_sharpe_long.csv`` — HMM regime–segmented Sharpe (long)
    - ``per_regime_sharpe_wide.csv`` — same Sharpe pivoted by regime column
    """
    os.makedirs(cfg.result_root, exist_ok=True)

    # --- equity overlay ---
    plt.figure(figsize=(11, 5.5))
    n_model_curves = 0
    for eid in experiment_ids:
        path = os.path.join(cfg.result_root, f"{eid}_backtest_steps.csv")
        if not os.path.isfile(path):
            continue
        st = pd.read_csv(path)
        if "equity" not in st.columns or "step" not in st.columns or st.empty:
            continue
        plt.plot(
            st["step"].to_numpy(),
            st["equity"].to_numpy(dtype=float),
            label=eid,
            alpha=0.9,
            linewidth=1.1,
        )
        n_model_curves += 1

    df_full = pd.read_csv(cfg.csv_path)
    split_stub = _split_cfg_stub(cfg.csv_path, cfg.train_ratio, cfg.split_config_path)
    _, df_test, _ = moe_train_test_split(df_full, split_stub)
    bh = buy_and_hold(df_test, close_col=cfg.close_col, capital=cfg.capital)
    has_bh = len(bh) > 0 and "equity" in bh.columns
    if has_bh:
        plt.plot(
            bh["step"].to_numpy(),
            bh["equity"].to_numpy(dtype=float),
            label="buy_hold",
            color="0.2",
            linestyle="--",
            linewidth=1.5,
        )

    overlay_path = os.path.join(cfg.result_root, "equity_overlay.png")
    if n_model_curves == 0 and not has_bh:
        plt.close()
        print(f"[aggregate] No step CSVs / buy-hold curve; skipped equity overlay ({overlay_path})")
    else:
        plt.title("Test-set equity overlay (aligned split)")
        plt.xlabel("step")
        plt.ylabel("equity (normalized start 1.0)")
        plt.legend(loc="best", fontsize=8)
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        plt.savefig(overlay_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[aggregate] Wrote equity overlay: {overlay_path}")

    # --- comparison metrics + per-regime ---
    metric_rows: list[dict[str, Any]] = []
    long_regime: list[dict[str, Any]] = []

    for eid in experiment_ids:
        path = os.path.join(cfg.result_root, f"{eid}_backtest_steps.csv")
        if not os.path.isfile(path):
            continue
        st = pd.read_csv(path)
        summ = summarize(st)
        metric_rows.append(
            {
                "experiment": eid,
                "Sharpe": summ.get("sharpe_step", float("nan")),
                "MDD": summ.get("max_drawdown", float("nan")),
                "Return": summ.get("total_return", float("nan")),
                "Turnover": summ.get("turnover_sum", float("nan")),
                "bars": summ.get("bars", 0),
            }
        )
        for seg in _per_hmm_regime_sharpe_table(st):
            long_regime.append({"experiment": eid, **seg})

    if metric_rows:
        mdf = pd.DataFrame(metric_rows)
        mpath = os.path.join(cfg.result_root, "comparison_metrics.csv")
        mdf.to_csv(mpath, index=False)
        print(f"[aggregate] Wrote comparison metrics: {mpath}")
    else:
        print("[aggregate] No backtest step CSVs; skipped comparison_metrics.csv")

    if long_regime:
        ldf = pd.DataFrame(long_regime)
        lpath = os.path.join(cfg.result_root, "per_regime_sharpe_long.csv")
        ldf.to_csv(lpath, index=False)
        print(f"[aggregate] Wrote per-HMM-regime Sharpe (long): {lpath}")
        wide = ldf.pivot_table(
            index="experiment",
            columns="hmm_regime_id",
            values="sharpe",
            aggfunc="first",
        )
        wide.columns = [f"sharpe_hmm_regime_{int(c)}" for c in wide.columns]
        wpath = os.path.join(cfg.result_root, "per_regime_sharpe_wide.csv")
        wide.reset_index().to_csv(wpath, index=False)
        print(f"[aggregate] Wrote per-HMM-regime Sharpe (wide): {wpath}")
    else:
        print(
            "[aggregate] No hmm_regime_id in step CSVs (re-run backtests); "
            "skipped per_regime_sharpe_*.csv"
        )


def train_and_backtest_ppo(
    exp_id: ExperimentId,
    df: pd.DataFrame,
    cfg: RunnerConfig,
) -> dict[str, Any]:
    _validate(exp_id, df)
    drop_cols = _ppo_drop_cols(exp_id, df)
    feature_cols = _feature_cols(df, drop_cols, cfg.close_col)
    if not feature_cols:
        raise ValueError(f"{exp_id}: no feature columns left after drop_cols")

    split_stub = _split_cfg_stub(cfg.csv_path, cfg.train_ratio, cfg.split_config_path)
    df_train, df_test, split_meta = moe_train_test_split(df, split_stub)

    X_train, close_train = df_to_arrays(
        df_train, feature_cols, close_col=cfg.close_col, dropna=True
    )
    train_env = TradingEnv(X_train, close_train, cfg.env_cfg)
    agent = PPOAgent(train_env, cfg.ppo_cfg)
    plot_dir = os.path.join(cfg.model_root, exp_id)
    os.makedirs(plot_dir, exist_ok=True)
    plot_base = os.path.join(plot_dir, "training_curves.png")
    agent.train(total_updates=cfg.total_updates, log_every=cfg.log_every, plot_path=plot_base)

    ckpt_path = os.path.join(cfg.model_root, f"{exp_id}_ppo_policy.pt")
    os.makedirs(os.path.dirname(ckpt_path) or ".", exist_ok=True)
    agent.save(ckpt_path)

    meta = {
        "experiment": exp_id,
        "type": "ppo",
        "csv_path": cfg.csv_path,
        "drop_cols": list(drop_cols),
        "feature_cols": feature_cols,
        "close_col": cfg.close_col,
        "train_rows": int(len(df_train)),
        "test_rows": int(len(df_test)),
        "split": split_meta,
        "checkpoint": ckpt_path,
    }
    os.makedirs(cfg.meta_root, exist_ok=True)
    with open(os.path.join(cfg.meta_root, f"{exp_id}.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    test_cfg = PPOTestConfig(
        agent_path=ckpt_path,
        test_csv=cfg.csv_path,
        close_col=cfg.close_col,
        drop_cols=tuple(drop_cols),
        train_ratio=cfg.train_ratio,
        split_config_path=cfg.split_config_path,
        use_test_split=True,
        capital=cfg.capital,
        out_csv=os.path.join(cfg.result_root, f"{exp_id}_backtest_steps.csv"),
        plot_dir=os.path.join(cfg.result_root, f"{exp_id}_plots"),
        plot_prefix=exp_id,
        compare_buy_hold=True,
    )
    bt = run_ppo_backtest(ckpt_path, cfg.eval_env_cfg, test_cfg)
    summary = {k: float(v) if isinstance(v, (int, float)) else v for k, v in bt["summary"].items()}
    bh = bt.get("buy_hold_summary") or {}
    summary_bh = {f"bh_{k}": float(v) if isinstance(v, (int, float)) else v for k, v in bh.items()}
    return {"experiment": exp_id, "kind": "ppo", "summary": summary, "summary_buy_hold": summary_bh, "meta": meta}


def train_and_backtest_moe(
    exp_id: ExperimentId,
    df: pd.DataFrame,
    cfg: RunnerConfig,
) -> dict[str, Any]:
    _validate(exp_id, df)
    data_cfg = _moe_data_cfg(exp_id, df, cfg.csv_path, cfg.train_ratio, cfg.split_config_path)

    df_train, df_test, split_meta = moe_train_test_split(df, data_cfg)
    feature_cols = df_train.columns.drop(drop_cols_existing(df_train, data_cfg.drop_cols)).tolist()
    if cfg.close_col in feature_cols:
        feature_cols = [c for c in feature_cols if c != cfg.close_col]

    X_train, close_train, regime_ids_train, _ = _prepare_moe_arrays(
        df=df_train,
        feature_cols=feature_cols,
        close_col=data_cfg.close_col,
        regime_cols=data_cfg.regime_cols,
    )

    train_env = TradingEnv(
        X_train,
        close_train,
        cfg.env_cfg,
        regime_ids=regime_ids_train,
    )
    agent = HardMoEAgent(
        obs_dims=train_env.obs_dim(),
        cfg=cfg.ppo_cfg,
        n_experts=len(data_cfg.regime_cols),
    )
    agent, history = train_moe_experts(
        agent=agent,
        env=train_env,
        total_updates=cfg.total_updates,
        log_every=cfg.log_every,
    )

    model_dir = os.path.join(cfg.model_root, exp_id)
    os.makedirs(model_dir, exist_ok=True)
    bundle_path = os.path.join(model_dir, "moe_experts.pt")
    save_moe_bundle(
        out_dir=model_dir,
        agent=agent,
        ppo_cfg=cfg.ppo_cfg,
        env_cfg=cfg.env_cfg,
        extra_meta={
            "experiment": exp_id,
            "data_cfg": {
                "csv_path": data_cfg.csv_path,
                "close_col": data_cfg.close_col,
                "regime_cols": list(data_cfg.regime_cols),
                "drop_cols": list(data_cfg.drop_cols),
                "train_ratio": data_cfg.train_ratio,
                "split_config_path": data_cfg.split_config_path,
            },
            "train_rows": int(len(df_train)),
            "test_rows": int(len(df_test)),
            "total_updates": cfg.total_updates,
            "split": split_meta,
            "feature_cols": feature_cols,
        },
    )

    meta = {
        "experiment": exp_id,
        "type": "moe",
        "bundle": bundle_path,
        "data_cfg": {
            "csv_path": data_cfg.csv_path,
            "close_col": data_cfg.close_col,
            "regime_cols": list(data_cfg.regime_cols),
            "drop_cols": list(data_cfg.drop_cols),
            "train_ratio": data_cfg.train_ratio,
            "split_config_path": data_cfg.split_config_path,
        },
        "split": split_meta,
    }
    os.makedirs(cfg.meta_root, exist_ok=True)
    with open(os.path.join(cfg.meta_root, f"{exp_id}.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)

    bt = run_moe_backtest(
        bundle_path=bundle_path,
        env_cfg=cfg.eval_env_cfg,
        data_cfg=data_cfg,
        capital=cfg.capital,
        out_csv=os.path.join(cfg.result_root, f"{exp_id}_backtest_steps.csv"),
        plot_dir=os.path.join(cfg.result_root, f"{exp_id}_plots"),
        plot_prefix=exp_id,
        compare_buy_hold=True,
    )
    summary = {k: float(v) if isinstance(v, (int, float)) else v for k, v in bt["summary"].items()}
    bh = bt.get("buy_hold_summary") or {}
    summary_bh = {f"bh_{k}": float(v) if isinstance(v, (int, float)) else v for k, v in bh.items()}
    return {"experiment": exp_id, "kind": "moe", "summary": summary, "summary_buy_hold": summary_bh, "meta": meta}


def run_experiment(exp_id: ExperimentId, df: pd.DataFrame, cfg: RunnerConfig) -> dict[str, Any]:
    if exp_id in ("ppo_base", "ppo_hmm", "ppo_hmm_lstm"):
        return train_and_backtest_ppo(exp_id, df, cfg)
    if exp_id in ("moe_hmm", "moe_hmm_lstm"):
        return train_and_backtest_moe(exp_id, df, cfg)
    raise ValueError(f"Unknown experiment: {exp_id}")


def run_experiments(
    experiment_ids: list[ExperimentId],
    cfg: RunnerConfig | None = None,
) -> pd.DataFrame:
    cfg = cfg or RunnerConfig()
    set_training_seed(int(cfg.env_cfg.seed))
    if not os.path.isfile(cfg.csv_path):
        raise FileNotFoundError(f"CSV not found: {cfg.csv_path}")

    os.makedirs(cfg.model_root, exist_ok=True)
    os.makedirs(cfg.result_root, exist_ok=True)

    df = pd.read_csv(cfg.csv_path)
    if cfg.resample_minutes > 5:
        orig_len = len(df)
        df = resample_bars(df, minutes=cfg.resample_minutes, close_col=cfg.close_col)
        resampled_path = cfg.csv_path.replace(".csv", f"_{cfg.resample_minutes}m_tmp.csv")
        os.makedirs(os.path.dirname(resampled_path) or ".", exist_ok=True)
        df.to_csv(resampled_path, index=False)
        cfg.csv_path = resampled_path
        print(
            f"[resample] {orig_len} rows @ 5m → {len(df)} rows @ {cfg.resample_minutes}m "
            f"(ratio {orig_len / max(len(df), 1):.1f}x)  saved to {resampled_path}"
        )
    rows: list[dict[str, Any]] = []

    for eid in experiment_ids:
        print(f"\n========== Experiment: {eid} ==========")
        try:
            out = run_experiment(eid, df, cfg)
            row: dict[str, Any] = {"experiment": eid, "kind": out["kind"], "error": ""}
            for k, v in out["summary"].items():
                row[k] = v
            for k, v in out.get("summary_buy_hold", {}).items():
                row[k] = v
            rows.append(row)
            print(f"[{eid}] summary:", out["summary"])
        except Exception as ex:
            err = f"{type(ex).__name__}: {ex}"
            print(f"[{eid}] FAILED: {err}")
            rows.append({"experiment": eid, "kind": "", "error": err})

    summary_df = pd.DataFrame(rows)
    out_path = os.path.join(cfg.result_root, "comparison_summary.csv")
    summary_df.to_csv(out_path, index=False)
    print(f"\nWrote comparison table: {out_path}")

    write_aggregate_backtest_reports(cfg, experiment_ids)
    return summary_df


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Run PPO / MoE regime experiments with aligned splits.")
    p.add_argument("--csv", default="data/data_e.csv", help="Path to data_e CSV")
    p.add_argument("--all", action="store_true", help="Run all five experiments")
    p.add_argument(
        "--only",
        type=str,
        default="",
        help="Comma-separated ids: ppo_base,ppo_hmm,ppo_hmm_lstm,moe_hmm,moe_hmm_lstm",
    )
    p.add_argument("--train-ratio", type=float, default=0.8, help="Fallback if split_config missing")
    p.add_argument("--split-config", type=str, default="", help="Optional path to split_config.json")
    p.add_argument(
        "--resample-minutes",
        type=int,
        default=0,
        help="Resample 5m bars to N-minute bars before training/test (0 = no resample, 15 = 15m).",
    )
    p.add_argument("--updates", type=int, default=200, help="PPO / MoE training updates")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--capital", type=float, default=10.0)
    p.add_argument("--model-root", default="model/experiments")
    p.add_argument("--result-root", default="result/experiments")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ent-coef", type=float, default=0.1, help="Entropy coefficient for PPO loss (lower → less random policy).")
    p.add_argument(
        "--ent-coef-end",
        type=float,
        default=None,
        help="If set, linearly anneal ent_coef from --ent-coef to this value over training. None = no annealing.",
    )
    p.add_argument(
        "--forward-return-bars",
        type=int,
        default=1,
        help="TradingEnv reward horizon: cumulative return over next N bars (1 = original 1-bar). Backtest equity still uses 1-bar info['ret'].",
    )
    p.add_argument(
        "--reward-scale",
        type=float,
        default=1000.0,
        help="Multiply environment reward after PnL/fees/Sharpe bonus (5m raw returns are ~1e-4).",
    )
    p.add_argument(
        "--fee-reward-discount",
        type=float,
        default=1.0,
        help="Reward subtracts fee * this (info['fee'] unchanged for backtest). <1 encourages trading.",
    )
    p.add_argument(
        "--turnover-penalty-coef",
        type=float,
        default=0.000,
        help="Reward subtracts coef * |Δposition| each step (before --reward-scale); 0 disables.",
    )
    p.add_argument(
        "--position-shaping-coef",
        type=float,
        default=0.00000,
        help="Reward adds coef*(|pos|-1)*h_eff (penalizes flat cash, encourages ±1); not in equity step_ret. 0=off.",
    )
    p.add_argument(
        "--rolling-sharpe-window",
        type=int,
        default=0,
        help="Past-only rolling Sharpe bonus in reward; 0 off. Uses W returns through bar t (no future leak).",
    )
    p.add_argument(
        "--rolling-sharpe-coef",
        type=float,
        default=0.01,
        help="Multiplier on clipped rolling Sharpe added to reward (before --reward-scale).",
    )
    args = p.parse_args(argv)

    if args.all:
        selected: list[ExperimentId] = list(EXPERIMENT_ORDER)
    elif args.only.strip():
        selected = []
        for part in args.only.split(","):
            part = part.strip()
            if not part:
                continue
            if part not in EXPERIMENT_ORDER:
                print(f"Unknown experiment id: {part!r}. Valid: {EXPERIMENT_ORDER}", file=sys.stderr)
                sys.exit(1)
            selected.append(part)  # type: ignore[arg-type]
    else:
        print("Specify --all or --only id1,id2,...", file=sys.stderr)
        sys.exit(1)

    cfg = RunnerConfig(
        csv_path=args.csv,
        train_ratio=args.train_ratio,
        split_config_path=args.split_config or None,
        resample_minutes=int(args.resample_minutes),
        total_updates=args.updates,
        log_every=args.log_every,
        capital=args.capital,
        model_root=args.model_root,
        result_root=args.result_root,
    )
    cfg.ppo_cfg.ent_coef = float(args.ent_coef)
    cfg.ppo_cfg.ent_coef_end = float(args.ent_coef_end) if args.ent_coef_end is not None else None
    cfg.env_cfg.seed = args.seed
    cfg.eval_env_cfg.seed = args.seed
    if args.forward_return_bars < 1:
        print("--forward-return-bars must be >= 1", file=sys.stderr)
        sys.exit(1)
    cfg.env_cfg.forward_return_bars = int(args.forward_return_bars)
    cfg.eval_env_cfg.forward_return_bars = int(args.forward_return_bars)
    if args.reward_scale <= 0:
        print("--reward-scale must be > 0", file=sys.stderr)
        sys.exit(1)
    cfg.env_cfg.reward_scale = float(args.reward_scale)
    cfg.eval_env_cfg.reward_scale = float(args.reward_scale)
    if args.fee_reward_discount < 0:
        print("--fee-reward-discount must be >= 0", file=sys.stderr)
        sys.exit(1)
    cfg.env_cfg.fee_reward_discount = float(args.fee_reward_discount)
    cfg.eval_env_cfg.fee_reward_discount = float(args.fee_reward_discount)
    if args.turnover_penalty_coef < 0:
        print("--turnover-penalty-coef must be >= 0", file=sys.stderr)
        sys.exit(1)
    cfg.env_cfg.turnover_penalty_coef = float(args.turnover_penalty_coef)
    cfg.eval_env_cfg.turnover_penalty_coef = float(args.turnover_penalty_coef)
    cfg.env_cfg.position_shaping_coef = float(args.position_shaping_coef)
    cfg.eval_env_cfg.position_shaping_coef = float(args.position_shaping_coef)
    if args.rolling_sharpe_window < 0:
        print("--rolling-sharpe-window must be >= 0", file=sys.stderr)
        sys.exit(1)
    cfg.env_cfg.rolling_sharpe_window = int(args.rolling_sharpe_window)
    cfg.eval_env_cfg.rolling_sharpe_window = int(args.rolling_sharpe_window)
    cfg.env_cfg.rolling_sharpe_coef = float(args.rolling_sharpe_coef)
    cfg.eval_env_cfg.rolling_sharpe_coef = float(args.rolling_sharpe_coef)

    run_experiments(selected, cfg)


if __name__ == "__main__":
    main()
