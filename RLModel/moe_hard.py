from __future__ import annotations
from typing import Any
from dataclasses import dataclass,asdict,is_dataclass
import json
import os
import warnings

import numpy as np
from ppo import PPOConfig, ActorCritic, TradingEnv, TradingEnvConfig
from ppo_test import summarize, plot_backtest, buy_and_hold
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

from split_bounds import (
    effective_rl_test_start,
    parse_split_bounds,
    rl_test_mask,
    rl_train_mask,
    try_load_split_config,
)


@dataclass
class MoERolloutBatch:
    obs:torch.Tensor
    actions:torch.Tensor
    old_log_probs:torch.Tensor
    advantages:torch.Tensor
    returns:torch.Tensor
    regime_ids:torch.Tensor


def _parse_cfg_timestamp(s: str | None) -> pd.Timestamp | None:
    if s is None or (isinstance(s, str) and not str(s).strip()):
        return None
    ts = pd.Timestamp(s)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def _timestamp_range_meta(df: pd.DataFrame, ts_col: str) -> dict[str, str | None]:
    if ts_col not in df.columns or len(df) == 0:
        return {"min": None, "max": None}
    ts = pd.to_datetime(df[ts_col], utc=True, errors="coerce").dropna()
    if ts.empty:
        return {"min": None, "max": None}
    return {"min": ts.min().isoformat(), "max": ts.max().isoformat()}


@dataclass
class MoEDataConfig:
    csv_path:str = "data/data_e.csv"
    close_col:str = "close_5m"
    regime_cols:tuple[str,...] = ("hmm_predicted_state_0","hmm_predicted_state_1","hmm_predicted_state_2","hmm_predicted_state_3")
    drop_cols:tuple[str,...] = ("timestamp","close_5m","hmm_predicted_state_0","hmm_predicted_state_1","hmm_predicted_state_2","hmm_predicted_state_3")
    train_ratio:float = 0.8  # fallback if split_config missing or time masks empty
    split_config_path:str | None = None  # optional explicit path to split JSON
    # Optional ISO-8601 UTC strings; only applied when split_config.json is used (with lstm_train_end / test_start).
    rl_train_start: str | None = None  # if set: require ts >= this (still enforces ts > lstm_train_end)
    rl_train_end: str | None = None  # if set: overrides split_config rl_train_end for the RL train window


def build_regime_ids(df,regime_cols):
    return df.loc[:,regime_cols].to_numpy(dtype = np.float32).argmax(axis = 1).astype(np.int64)


def drop_cols_existing(df: pd.DataFrame, drop_cols: tuple[str, ...]) -> list[str]:
    """Subset of ``drop_cols`` present in ``df`` (avoids ``Index.drop`` KeyError if CSV omits optional columns)."""
    return [c for c in drop_cols if c in df.columns]


_HMM_PRED_OH: tuple[str, ...] = tuple(f"hmm_predicted_state_{i}" for i in range(4))


def moe_train_test_split(
    df: pd.DataFrame,
    data_cfg: MoEDataConfig,
    ts_col: str = "timestamp",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """
    RL train: after LSTM cutoff, through rl_train_end, strictly before effective test start.
    RL test: ts >= max(test_start, rl_train_end_effective) so evaluation never starts before RL train ends.

    Optional ``MoEDataConfig.rl_train_start`` / ``rl_train_end`` tighten or override the calendar window
    when split_config.json is present. Falls back to ``train_ratio`` if no config or empty masks.
    """
    meta: dict[str, Any] = {
        "split_mode": "train_ratio",
        "train_ratio": data_cfg.train_ratio,
        "note": "No usable split_config (or missing timestamp column): calendar alignment vs LSTM/HMM not enforced.",
    }
    raw = try_load_split_config(data_cfg.split_config_path) if data_cfg.split_config_path else try_load_split_config()
    if raw is not None and ts_col in df.columns:
        try:
            bounds = parse_split_bounds(raw)
            lstm_end = bounds["lstm_train_end"]
            rl_end_split = bounds["rl_train_end"]
            test_start_cfg = bounds["test_start"]

            rl_end_eff = _parse_cfg_timestamp(data_cfg.rl_train_end) or rl_end_split
            bounds_use = {**bounds, "rl_train_end": rl_end_eff}
            test_start_eff = effective_rl_test_start(bounds_use)

            m_tr = rl_train_mask(df, bounds_use, ts_col=ts_col)
            m_te = rl_test_mask(df, bounds_use, ts_col=ts_col)
            if data_cfg.rl_train_start:
                rs = _parse_cfg_timestamp(data_cfg.rl_train_start)
                if rs is None:
                    raise ValueError("MoEDataConfig.rl_train_start is not a valid timestamp")
                ts = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
                m_tr = m_tr & (ts >= rs)

            df_tr = df.loc[m_tr].copy()
            df_te = df.loc[m_te].copy()
            if len(df_tr) > 0 and len(df_te) > 0:
                meta = {
                    "split_mode": "split_config.json",
                    "bounds_from_config": {k: str(v) for k, v in bounds.items()},
                    "rl_window_effective": {
                        "lstm_train_end": lstm_end.isoformat(),
                        "rl_train_start_config": data_cfg.rl_train_start,
                        "rl_train_start_applied": (
                            _parse_cfg_timestamp(data_cfg.rl_train_start).isoformat()
                            if data_cfg.rl_train_start
                            else None
                        ),
                        "rl_train_end_from_split": rl_end_split.isoformat(),
                        "rl_train_end_effective": rl_end_eff.isoformat(),
                        "rl_train_end_overridden_by_moe_cfg": bool(data_cfg.rl_train_end),
                        "test_start_from_split": test_start_cfg.isoformat(),
                        "test_start_effective": test_start_eff.isoformat(),
                        "test_start_effective_equals_max_of_test_start_and_rl_train_end": True,
                    },
                    "constraints_satisfied": {
                        "rl_train_only_after_lstm_train_end": True,
                        "rl_train_not_after_rl_train_end_effective": True,
                        "rl_train_strictly_before_effective_test_start": True,
                        "rl_test_not_before_rl_train_end_effective": True,
                    },
                    "train_rows": len(df_tr),
                    "test_rows": len(df_te),
                    "train_timestamp_range": _timestamp_range_meta(df_tr, ts_col),
                    "test_timestamp_range": _timestamp_range_meta(df_te, ts_col),
                }
                if test_start_cfg < rl_end_eff:
                    meta["warnings"] = [
                        "split_config test_start is earlier than rl_train_end_effective; "
                        "using test_start_effective = max(test_start, rl_train_end_effective) for RL test mask."
                    ]
                return df_tr, df_te, meta
            warnings.warn(
                "moe_train_test_split: split_config masks produced empty train or test set; "
                f"falling back to train_ratio={data_cfg.train_ratio}.",
                UserWarning,
                stacklevel=2,
            )
        except (KeyError, ValueError, TypeError) as e:
            warnings.warn(
                f"moe_train_test_split: split_config calendar path failed ({e!r}); "
                f"falling back to train_ratio={data_cfg.train_ratio}.",
                UserWarning,
                stacklevel=2,
            )
    else:
        if raw is None:
            warnings.warn(
                "moe_train_test_split: split_config.json not found; using train_ratio fallback.",
                UserWarning,
                stacklevel=2,
            )
        elif ts_col not in df.columns:
            warnings.warn(
                f"moe_train_test_split: column {ts_col!r} not in dataframe; using train_ratio fallback.",
                UserWarning,
                stacklevel=2,
            )
    split_idx = int(len(df) * data_cfg.train_ratio)
    df_tr = df.iloc[:split_idx].copy()
    df_te = df.iloc[split_idx:].copy()
    meta["train_timestamp_range"] = _timestamp_range_meta(df_tr, ts_col)
    meta["test_timestamp_range"] = _timestamp_range_meta(df_te, ts_col)
    return df_tr, df_te, meta

def _load_moe_data(bundle_path:str) -> dict:
    meta_path = os.path.join(os.path.dirname(bundle_path),"moe_meta.json")
    if not os.path.exists(meta_path):
        return {}
    with open(meta_path,"r") as f:
        return json.load(f)
    
def _ppo_config_from_meta(meta:dict[str,Any])->PPOConfig:
    cfg = PPOConfig()
    for k,v in meta.get("ppo_cfg",{}).items():
        if hasattr(cfg,k):
            setattr(cfg,k,v)
    return cfg

def _prepare_moe_arrays(df,feature_cols,close_col,regime_cols):
    use = df[feature_cols + [close_col] + list(regime_cols)].copy()
    use = use.dropna().reset_index(drop = True)
    X = use[feature_cols].to_numpy(dtype = np.float32)
    close = use[close_col].to_numpy(dtype = np.float32)
    regime_ids = (
        use.loc[:,regime_cols].to_numpy(dtype = np.float32).argmax(axis = 1).astype(np.int64)
    )
    return X,close,regime_ids,use

def get_current_regime(env)-> int:
    if env.regime is None:
        raise ValueError("Env has no regime ids")
    return int(env.regime[env.t])

def _ppo_cfg_to_dict(cfg: PPOConfig) -> dict[str, Any]:
    if is_dataclass(cfg):
        return asdict(cfg)
    return {
        "gamma": cfg.gamma,
        "gae_lambda": cfg.gae_lambda,
        "clip_eps": cfg.clip_eps,
        "ent_coef": cfg.ent_coef,
        "ent_coef_end": cfg.ent_coef_end,
        "vf_coef": cfg.vf_coef,
        "lr": cfg.lr,
        "max_grad_norm": cfg.max_grad_norm,
        "rollout_steps": cfg.rollout_steps,
        "minibatch_size": cfg.minibatch_size,
        "update_epochs": cfg.update_epochs,
        "hidden": cfg.hidden,
        "device": cfg.device,
    }

# save model at checkpoints
def save_moe_bundle(out_dir,agent,ppo_cfg,env_cfg = None,extra_meta = None):
    os.makedirs(out_dir,exist_ok=True)
    ckpt_path = os.path.join(out_dir,"moe_experts.pt")
    meta_path = os.path.join(out_dir,"moe_meta.json")

    payload = {
        "n_experts":len(agent.experts),
        "expert_state_dicts":[expert.state_dict() for expert in agent.experts],
    }
    torch.save(payload,ckpt_path)
    meta = {
        "n_experts":len(agent.experts),
        "ppo_cfg":_ppo_cfg_to_dict(ppo_cfg),
    }
    if env_cfg is not None:
        meta["env_cfg"] = {
            "fee_bps":env_cfg.fee_bps,
            "hold_cost_bps":env_cfg.hold_cost_bps,
            "max_episode_steps":env_cfg.max_episode_steps,
            "random_start":env_cfg.random_start,
            "start_index":env_cfg.start_index,
            "seed":env_cfg.seed,
            "reward_scale":env_cfg.reward_scale,
        }
    if extra_meta:
        meta["extra_meta"] = extra_meta
        sp = extra_meta.get("split")
        if isinstance(sp, dict):
            meta["train_test_split_meta"] = sp

    with open(meta_path,"w") as f:
        json.dump(meta,f,indent=2)
    
    print(f"[save] moe checkpoint: {ckpt_path}")
    print(f"[save] moe metadata: {meta_path}")
    return ckpt_path

def load_moe_bundle(bundle_path,agent,map_location = None):
    ckpt = torch.load(bundle_path,map_location=map_location)
    expert_state_dicts = ckpt["expert_state_dicts"]
    if len(expert_state_dicts)!=len(agent.experts):
        raise ValueError(
            f"Expert count mismatch"
        )
    
    for expert,state_dict in zip(agent.experts,expert_state_dicts):
        expert.load_state_dict(state_dict)
    print(f"[load] moe checkpoint loaded from {bundle_path}")
    return agent


class HardMoEAgent(nn.Module):
    def __init__(self,obs_dims,cfg,n_actions = 3,n_experts = 4):
        super().__init__()
        self.cfg = cfg
        self.device = cfg.device
        self.n_actions = n_actions
        self.n_experts = n_experts
        self.IDX_TO_ACT = np.array([-1,0,1],dtype = np.int32)
        self.experts = nn.ModuleList(
            [
                ActorCritic(obs_dim = obs_dims,
                n_actions = n_actions,
                hidden = cfg.hidden,).to(self.device)
                for _ in range(n_experts)
            ]
        )
        self.optimizers = [optim.Adam(expert.parameters(),lr = cfg.lr) for expert in self.experts]
    def train_mode(self)->None:
        for expert in self.experts:
            expert.train()
    def eval_mode(self)-> None:
        for expert in self.experts:
            expert.eval()
    def _check_regime_id(self,regime_id:int)->None:
        if regime_id < 0 or regime_id >= self.n_experts:
            raise ValueError("Invalid Regime IDs")
    def get_expert(self,regime_id:int) -> ActorCritic:
        self._check_regime_id(regime_id)
        return self.experts[regime_id]
    def forward_expert(self,obs:torch.Tensor,regime_id:int):
        expert = self.get_expert(regime_id)
        return expert(obs)
    
    @torch.no_grad()
    def act(self,obs,regime_id)->tuple:
        self._check_regime_id(regime_id)
        if obs.ndim !=1:
            raise ValueError("Obs dim is not 1")
        expert = self.get_expert(regime_id)
        logits,value = expert(obs.unsqueeze(0))
        dist = torch.distributions.Categorical(logits=logits)
        act_idx = dist.sample()
        logprob = dist.log_prob(act_idx)
        return act_idx.squeeze(0),logprob.squeeze(0),value.squeeze(0)
    
    @torch.no_grad()
    def act_deterministic(self,obs,regime_id):
        self._check_regime_id(regime_id)
        if obs.ndim != 1:
            raise ValueError("Obs dim is not 1")
        expert = self.get_expert(regime_id)
        logits,value = expert(obs.unsqueeze(0))
        act_idx = int(torch.argmax(logits,dim = -1).item())
        env_action = int(self.IDX_TO_ACT[act_idx])
        return env_action,float(value.item())
    def ppo_loss_for_subset(self,expert_id,mb_obs,mb_act,mb_oldlog,mb_adv,mb_ret,ent_coef=None):
        ec = ent_coef if ent_coef is not None else self.cfg.ent_coef
        expert = self.get_expert(expert_id)
        logits,values = expert(mb_obs)
        dist = torch.distributions.Categorical(logits=logits)
        newlog = dist.log_prob(mb_act)
        entropy = dist.entropy().mean()
        ratio = torch.exp(newlog - mb_oldlog)
        surr1 = ratio * mb_adv
        surr2 = torch.clamp(ratio,1.0-self.cfg.clip_eps,1.0+self.cfg.clip_eps)*mb_adv
        pi_loss = -torch.min(surr1,surr2).mean()

        v_loss = ((values - mb_ret)**2).mean()
        loss = pi_loss + self.cfg.vf_coef * v_loss - ec * entropy

        return loss,pi_loss,v_loss,entropy
    
    def update_one_expert(self,expert_id,mb_obs,mb_act,mb_oldlog,mb_adv,mb_ret,ent_coef=None):
        optimizer = self.optimizers[expert_id]
        optimizer.zero_grad()

        loss,pi_loss,v_loss,entropy = self.ppo_loss_for_subset(
            expert_id=expert_id,
            mb_obs = mb_obs,
            mb_act = mb_act,
            mb_oldlog = mb_oldlog,
            mb_adv = mb_adv,
            mb_ret = mb_ret,
            ent_coef=ent_coef,
        )

        loss.backward()
        nn.utils.clip_grad_norm_(self.get_expert(expert_id).parameters(),self.cfg.max_grad_norm,)
        optimizer.step()

        return {
            "loss":float(loss.item()),
            "pi_loss":float(pi_loss.item()),
            "v_loss":float(v_loss.item()),
            "entropy":float(entropy.item()),
            "n_samples":int(len(mb_obs)),
        }
    
    def update_from_buffer(self,batch:MoERolloutBatch,minibatch_size,update_epochs,ent_coef=None):
        if minibatch_size is None:
            minibatch_size = self.cfg.minibatch_size
        if update_epochs is None:
            update_epochs = self.cfg.update_epochs
        
        n = batch.obs.shape[0]
        stats = {
            "loss":0.0,
            "pi_loss":0.0,
            "v_loss":0.0,
            "entropy":0.0,
            "num_updates":0,
            "num_samples":0,
        }

        for _ in range(update_epochs):
            idx = torch.randperm(n,device=batch.obs.device)

            for start in range(0,n,minibatch_size):
                mb_idx = idx[start:start+minibatch_size]
                mb_obs = batch.obs[mb_idx]
                mb_act = batch.actions[mb_idx]
                mb_oldlog = batch.old_log_probs[mb_idx]
                mb_adv = batch.advantages[mb_idx]
                mb_ret = batch.returns[mb_idx]
                mb_regime = batch.regime_ids[mb_idx]

                for expert_id in range(self.n_experts):
                    mask = mb_regime == expert_id
                    if not torch.any(mask):
                        continue
                    result = self.update_one_expert(expert_id,mb_obs[mask],mb_act[mask],mb_oldlog[mask],mb_adv[mask],mb_ret[mask],ent_coef=ent_coef)
                    stats["loss"] += result["loss"]
                    stats["pi_loss"] += result["pi_loss"]
                    stats["v_loss"] += result["v_loss"]
                    stats["entropy"] += result["entropy"]
                    stats["num_updates"] += 1
                    stats["num_samples"] += result["n_samples"]
        denom = max(stats["num_updates"],1)
        return {
            "loss":stats["loss"]/denom,
            "pi_loss": stats["pi_loss"] / denom,
            "v_loss": stats["v_loss"] / denom,
            "entropy": stats["entropy"] / denom,
            "num_updates": stats["num_updates"],
            "num_samples": stats["num_samples"],
        }

class MoERollingBuffer:
    def __init__(self, size: int, obs_dim: int, device: str):
        self.size = size
        self.device = device

        self.obs = torch.zeros((size, obs_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((size,), dtype=torch.long, device=device)
        self.rewards = torch.zeros((size,), dtype=torch.float32, device=device)
        self.dones = torch.zeros((size,), dtype=torch.float32, device=device)
        self.logprobs = torch.zeros((size,), dtype=torch.float32, device=device)
        self.values = torch.zeros((size,), dtype=torch.float32, device=device)
        self.regime_ids = torch.zeros((size,), dtype=torch.long, device=device)

        self.advantages = torch.zeros((size,), dtype=torch.float32, device=device)
        self.returns = torch.zeros((size,), dtype=torch.float32, device=device)

        self.ptr = 0

    def add(self,obs,act_idx,reward,done,logprob,value,regime_id):
        if self.ptr >= self.size:
            raise IndexError("Rolling Buffer is FULL")
        i = self.ptr
        self.obs[i] = obs
        self.actions[i] = act_idx
        self.rewards[i] = reward
        self.dones[i] = done
        self.logprobs[i] = logprob
        self.values[i] = value
        self.regime_ids[i] = int(regime_id)

        self.ptr += 1

    def compute_gae(self,last_value,gamma,lam):
        adv = torch.zeros((),dtype = torch.float32,device = self.device)
        for t in reversed(range(self.size)):
            mask = 1.0 - self.dones[t]
            next_value = last_value if t == self.size - 1 else self.values[t+1]
            delta = self.rewards[t] + gamma * next_value * mask - self.values[t]
            adv = delta + gamma * lam * mask * adv
            self.advantages[t] = adv

        self.returns[:] = self.advantages + self.values
        adv_mean = self.advantages.mean()
        adv_std = self.advantages.std()
        self.advantages[:] = (self.advantages - adv_mean) / (adv_std+1e-12)
    def as_batch(self):
        if self.ptr != self.size:
            raise ValueError(f"Buffer not full, only export after full rollout")
        return MoERolloutBatch(
            obs = self.obs,
            actions= self.actions,
            old_log_probs=self.logprobs,
            advantages=self.advantages,
            returns = self.returns,
            regime_ids=self.regime_ids,
        )

# Train experts(main code here)
def train_moe_experts(agent:HardMoEAgent,env,total_updates:int = 200,log_every = 10):
    agent.train_mode()
    obs_dim = env.obs_dim()
    obs_np = env.reset()
    obs = torch.tensor(obs_np,dtype=torch.float32,device = agent.device)

    ent_start = float(agent.cfg.ent_coef)
    ent_end = float(agent.cfg.ent_coef_end) if agent.cfg.ent_coef_end is not None else ent_start

    history = {
        "upd": [],
        "rollout_reward_mean": [],
        "rollout_reward_sum": [],
        "rollout_abspos_mean": [],
        "rollout_turnover_mean": [],
        "rollout_fee_sum": [],
        "loss": [],
        "pi_loss": [],
        "v_loss": [],
        "entropy": [],
    }
    for expert_id in range(agent.n_experts):
        history[f"expert_{expert_id}_steps"] = []
    for upd in range(total_updates):
        frac = upd / max(total_updates - 1, 1)
        ent_coef_t = ent_start + (ent_end - ent_start) * frac

        buf = MoERollingBuffer(agent.cfg.rollout_steps,obs_dim,agent.device)
        r_sum = 0.0
        fee_sum = 0.0
        abspos_sum = 0.0
        turnover_sum = 0.0
        regime_counts = np.zeros(agent.n_experts,dtype = np.int64)

        for _ in range(agent.cfg.rollout_steps):
            regime_id = get_current_regime(env)
            regime_counts[regime_id] += 1

            with torch.no_grad():
                act_idx,logprob,value = agent.act(obs,regime_id)

            env_action = int(agent.IDX_TO_ACT[int(act_idx.item())])
            next_obs_np,reward,done,info = env.step(env_action)

            r_sum += float(reward)
            fee_sum += float(info.get("fee",0.0))
            abspos_sum += abs(float(info.get("pos",0.0)))
            turnover_sum += float(info.get("turnover",0.0))


            buf.add(
                obs = obs,
                act_idx=act_idx,
                reward = torch.tensor(reward,dtype = torch.float32,device = agent.device),
                done = torch.tensor(float(done),dtype = torch.float32,device = agent.device),
                logprob=logprob,
                value = value,
                regime_id=regime_id,
            )
            obs_np = env.reset() if done else next_obs_np

            obs = torch.tensor(obs_np,dtype=torch.float32,device = agent.device)
        with torch.no_grad():
            last_regime_id = get_current_regime(env)
            _,last_value = agent.forward_expert(obs.unsqueeze(0),last_regime_id)
            last_value = float(last_value.squeeze(0).item())

        buf.compute_gae(last_value,agent.cfg.gamma,agent.cfg.gae_lambda)
        batch = buf.as_batch()

        train_stats = agent.update_from_buffer(
            batch=batch,
            minibatch_size=agent.cfg.minibatch_size,
            update_epochs=agent.cfg.update_epochs,
            ent_coef=ent_coef_t,
        )

        if (upd + 1) % log_every == 0 or upd == 0:
            r_mean = r_sum / agent.cfg.rollout_steps
            abspos_mean = abspos_sum / agent.cfg.rollout_steps
            turnover_mean = turnover_sum / agent.cfg.rollout_steps

            history["upd"].append(upd + 1)
            history["rollout_reward_mean"].append(r_mean)
            history["rollout_reward_sum"].append(r_sum)
            history["rollout_abspos_mean"].append(abspos_mean)
            history["rollout_turnover_mean"].append(turnover_mean)
            history["rollout_fee_sum"].append(fee_sum)
            history["loss"].append(train_stats["loss"])
            history["pi_loss"].append(train_stats["pi_loss"])
            history["v_loss"].append(train_stats["v_loss"])
            history["entropy"].append(train_stats["entropy"])

            for expert_id in range(agent.n_experts):
                history[f"expert_{expert_id}_steps"].append(int(regime_counts[expert_id]))

            print(
                f"[HardMoE] upd={upd+1}/{total_updates} "
                f"r_mean={r_mean:.3e} r_sum={r_sum:.3e} "
                f"|pos|={abspos_mean:.3f} turn={turnover_mean:.3f} fee_sum={fee_sum:.3e} "
                f"loss={train_stats['loss']:.3e} pi={train_stats['pi_loss']:.3e} "
                f"v={train_stats['v_loss']:.3e} ent={train_stats['entropy']:.3e} "
                f"ent_coef={ent_coef_t:.4f} "
                f"exp_counts={regime_counts.tolist()}"
            )
    return agent,history

# BackTest

@torch.no_grad()
def backtest_moe(
    agent: HardMoEAgent,
    env,
    capital: float = 1.0,
    hmm_regime_ids: np.ndarray | None = None,
) -> pd.DataFrame:
    agent.eval_mode()
    obs_np = env.reset()
    done = False
    rows = []
    cum_reward = 0.0
    step = 0

    while not done:
        regime_id = get_current_regime(env)
        obs = torch.tensor(obs_np,dtype = torch.float32,device=agent.device)
        action,value_est = agent.act_deterministic(obs,regime_id)
        next_obs_np,reward,done,info = env.step(action)

        t = int(info["t"])
        simple_ret = float(info["ret"])
        pos = int(info["pos"])
        fee = float(info["fee"])
        hold = float(info["hold_cost"])
        step_ret = pos*simple_ret - fee - hold
        equity =(rows[-1]["equity"] if rows else 1.0)*(1.0 + step_ret)
        nav = equity * float(capital)
        cum_reward += float(reward)

        row = {
            "step":step,
            "t":t,
            "close":float(env.close[t]),
            "ret":simple_ret,
            "step_ret":float(step_ret),
            "action":int(action),
            "pos": pos,
            "turnover": float(info["turnover"]),
            "fee": fee,
            "hold_cost": hold,
            "reward": float(reward),
            "cum_reward": float(cum_reward),
            "equity": float(equity),
            "nav": float(nav),
            "regime_id": int(regime_id),
            "expert_id": int(regime_id),
            "value_est": float(value_est),
        }
        if "regime" in info:
            row["next_regime_id"] = int(info["regime"])
        if hmm_regime_ids is not None and 0 <= t < len(hmm_regime_ids):
            row["hmm_regime_id"] = int(hmm_regime_ids[t])

        rows.append(row)

        obs_np = next_obs_np
        step += 1

    return pd.DataFrame(rows)

def run_moe_backtest(bundle_path,env_cfg,data_cfg = None,capital = 1.0,out_csv = None,plot_dir = None,plot_prefix = "moe_hard",compare_buy_hold = True):
    if data_cfg is None:
        data_cfg = MoEDataConfig()
    meta = _load_moe_data(bundle_path)
    ppo_cfg = _ppo_config_from_meta(meta)
    n_experts = int(meta.get("n_experts",len(data_cfg.regime_cols)))

    df = pd.read_csv(data_cfg.csv_path)
    _, df_test, _ = moe_train_test_split(df, data_cfg)
    feature_cols = df_test.columns.drop(drop_cols_existing(df_test, data_cfg.drop_cols)).to_list()
    X_test,close_test,regime_ids_test,df_test_used = _prepare_moe_arrays(df = df_test,feature_cols=feature_cols,close_col=data_cfg.close_col,regime_cols=data_cfg.regime_cols)

    if len(regime_ids_test) != len(X_test):
        regime_ids_test = regime_ids_test[-len(X_test):]
    eval_env_cfg = TradingEnvConfig(
        fee_bps=env_cfg.fee_bps,
        hold_cost_bps=env_cfg.hold_cost_bps,
        max_episode_steps=None,
        random_start=False,
        start_index=env_cfg.start_index,
        seed=env_cfg.seed,
        forward_return_bars=env_cfg.forward_return_bars,
        fee_reward_discount=env_cfg.fee_reward_discount,
        turnover_penalty_coef=env_cfg.turnover_penalty_coef,
        position_shaping_coef=env_cfg.position_shaping_coef,
        reward_scale=env_cfg.reward_scale,
        rolling_sharpe_window=env_cfg.rolling_sharpe_window,
        rolling_sharpe_coef=env_cfg.rolling_sharpe_coef,
        rolling_sharpe_clip=env_cfg.rolling_sharpe_clip,
    )
    env = TradingEnv(X_test,close_test,eval_env_cfg,regime_ids=regime_ids_test)
    agent = HardMoEAgent(obs_dims=env.obs_dim(),cfg=ppo_cfg,n_experts=n_experts)
    load_moe_bundle(bundle_path,agent,map_location=agent.device)
    hmm_regime_ids = None
    if all(c in df_test_used.columns for c in _HMM_PRED_OH):
        hmm_regime_ids = (
            df_test_used[list(_HMM_PRED_OH)].to_numpy(dtype=np.float32).argmax(axis=1).astype(np.int64)
        )
    steps = backtest_moe(agent, env, capital=capital, hmm_regime_ids=hmm_regime_ids)
    summary = summarize(steps)
    bh_steps = pd.DataFrame()
    bh_summary = {}
    if compare_buy_hold:
        bh_steps = buy_and_hold(df_test_used,close_col=data_cfg.close_col,capital=capital)
        bh_summary = summarize(bh_steps) if len(bh_steps)>0 else {}

    if out_csv:
        os.makedirs(os.path.dirname(out_csv) or ".",exist_ok=True)
        steps.to_csv(out_csv,index=False)
    if plot_dir:
        plot_backtest(steps,out_dir=plot_dir,prefix=plot_prefix,show = False)
        if compare_buy_hold and len(bh_steps) >0:
            os.makedirs(plot_dir,exist_ok=True)
            plt.figure()
            plt.plot(steps["step"],steps["equity"],label = "moe_equity")
            plt.plot(bh_steps["step"],bh_steps["equity"],label = "buy_hold_equity")
            plt.title("Equity Comparison")
            plt.xlabel("step")
            plt.ylabel("equity")
            plt.legend()
            compare_path = os.path.join(plot_dir,f"{plot_prefix}_vs_buyhold.png")
            plt.savefig(compare_path,dpi=150,bbox_inches = "tight")
            plt.close()
            print("Successfully Save Comparison Image")

    return{
        "steps":steps,
        "summary":summary,
        "buy_hold_steps":bh_steps,
        "buy_hold_summary":bh_summary,
    }


    
if __name__ == "__main__":
    from seed_utils import set_training_seed

    data_cfg = MoEDataConfig(
        csv_path="data/data_e.csv",
        close_col="close_5m",
        regime_cols=("hmm_predicted_state_0", "hmm_predicted_state_1", "hmm_predicted_state_2", "hmm_predicted_state_3"),
        drop_cols=("timestamp", "close_5m", "hmm_predicted_state_0", "hmm_predicted_state_1", "hmm_predicted_state_2", "hmm_predicted_state_3"),
        train_ratio=0.8,
    )
    ppo_cfg = PPOConfig()
    env_cfg = TradingEnvConfig(
        fee_bps=5,
        hold_cost_bps=0.2,
        max_episode_steps=10000,
        random_start=True,
        start_index=1,
        seed=56,
    )
    set_training_seed(int(env_cfg.seed))
    model_dir = "model/moe_hard"
    out_csv = "result/moe_backtest_steps.csv"
    plot_dir = "plots"
    plot_prefix = "moe_hard"
    capital = 10.0
    total_updates = 200
    log_every = 10
    df = pd.read_csv(data_cfg.csv_path)
    df_train, _, split_meta = moe_train_test_split(df, data_cfg)
    feature_cols = df_train.columns.drop(drop_cols_existing(df_train, data_cfg.drop_cols)).tolist()
    X_train, close_train, regime_ids_train, _ = _prepare_moe_arrays(
        df=df_train,
        feature_cols=feature_cols,
        close_col=data_cfg.close_col,
        regime_cols=data_cfg.regime_cols,
    )

    train_env = TradingEnv(
        X_train,
        close_train,
        env_cfg,
        regime_ids=regime_ids_train,
    )

    agent = HardMoEAgent(
        obs_dims=train_env.obs_dim(),
        cfg=ppo_cfg,
        n_experts=len(data_cfg.regime_cols),
    )

    agent, history = train_moe_experts(
        agent=agent,
        env=train_env,
        total_updates=total_updates,
        log_every=log_every,
    )

    save_moe_bundle(
        out_dir=model_dir,
        agent=agent,
        ppo_cfg=ppo_cfg,
        env_cfg=env_cfg,
        extra_meta={
            "data_cfg": {
                "csv_path": data_cfg.csv_path,
                "close_col": data_cfg.close_col,
                "regime_cols": list(data_cfg.regime_cols),
                "drop_cols": list(data_cfg.drop_cols),
                "train_ratio": data_cfg.train_ratio,
                "split_config_path": data_cfg.split_config_path,
                "rl_train_start": data_cfg.rl_train_start,
                "rl_train_end": data_cfg.rl_train_end,
            },
            "train_rows": int(len(df_train)),
            "total_updates": total_updates,
            "split": split_meta,
        },
    )

    result = run_moe_backtest(
        bundle_path=os.path.join(model_dir, "moe_experts.pt"),
        env_cfg=env_cfg,
        data_cfg=data_cfg,
        capital=capital,
        out_csv=out_csv,
        plot_dir=plot_dir,
        plot_prefix=plot_prefix,
        compare_buy_hold=True,
    )

    print("==== MoE Backtest Summary ====")
    for k, v in result["summary"].items():
        print(f"{k:>14s}: {v}")

    if result["buy_hold_summary"]:
        print("==== Buy&Hold Summary ====")
        for k, v in result["buy_hold_summary"].items():
            print(f"{k:>14s}: {v}")

    print(f"Saved MoE model to: {os.path.join(model_dir, 'moe_experts.pt')}")
    print(f"Saved MoE backtest csv to: {out_csv}")
        