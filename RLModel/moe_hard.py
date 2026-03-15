from __future__ import annotations
from typing import Any
from dataclasses import dataclass,asdict,is_dataclass
import numpy as np
import json
import os
from ppo import PPOConfig,ActorCritic
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

@dataclass
class MoERolloutBatch:
    obs:torch.Tensor
    actions:torch.Tensor
    old_log_probs:torch.Tensor
    advantages:torch.Tensor
    returns:torch.Tensor
    regime_ids:torch.Tensor


@dataclass
class MoEDataConfig:
    csv_path:str = "data/data_e.csv"
    close_col:str = "close_5m"
    regime_cols:tuple[str,...] = ("state_0","state_1","state_2","state_3")
    drop_cols:tuple[str,...] = ("timestamp","regime","close_5m","state_0","state_1","state_2","state_3")
    train_ratio:float = 0.8


def build_regime_ids(df,regime_cols):
    return df.loc[:,regime_cols].to_numpy(dtype = np.float32).argmax(axis = 1).astype(np.int64)

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
        }
    if extra_meta:
        meta["extra_meta"] = extra_meta

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
    def ppo_loss_for_subset(self,expert_id,mb_obs,mb_act,mb_oldlog,mb_adv,mb_ret):
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
        loss = pi_loss + self.cfg.vf_coef * v_loss - self.cfg.ent_coef * entropy

        return loss,pi_loss,v_loss,entropy
    
    def update_one_expert(self,expert_id,mb_obs,mb_act,mb_oldlog,mb_adv,mb_ret):
        optimizer = self.optimizers[expert_id]
        optimizer.zero_grad()

        loss,pi_loss,v_loss,entropy = self.ppo_loss_for_subset(
            expert_id=expert_id,
            mb_obs = mb_obs,
            mb_act = mb_act,
            mb_oldlog = mb_oldlog,
            mb_adv = mb_adv,
            mb_ret = mb_ret,
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
    
    def update_from_buffer(self,batch:MoERolloutBatch,minibatch_size,update_epochs):
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
                    result = self.update_one_expert(expert_id,mb_obs[mask],mb_act[mask],mb_oldlog[mask],mb_adv[mask],mb_ret[mask])
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
    obs_dim = env.X.shape[1]
    obs_np = env.reset()
    obs = torch.tensor(obs_np,dtype=torch.float32,device = agent.device)
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
        "expert_0_steps": [],
        "expert_1_steps": [],
        "expert_2_steps": [],
        "expert_3_steps": [],
    }
    for upd in range(total_updates):
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
                f"exp_counts={regime_counts.tolist()}"
            )
    return agent,history

# BackTest

@torch.no_grad()
def backtest_moe(agent:HardMoEAgent,env,capital:float=1.0) -> pd.DataFrame:
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

        rows.append(row)

        obs_np = next_obs_np
        step += 1

    return pd.DataFrame(rows)


if __name__ == "__main__":
    pass
        