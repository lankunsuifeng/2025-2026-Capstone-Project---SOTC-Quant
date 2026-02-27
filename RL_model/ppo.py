# PPO Trader with Volatility Regime Switching

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, Dict, Optional,List

import torch
import torch.nn as nn
import torch.optim as optim

# Align Regime to Bars

def align_hourly_regimes(bars:pd.DataFrame,hourly:pd.DataFrame,time_col:str = "datetime") -> pd.DataFrame:
    b = bars.copy()
    h = hourly.copy()

    b[time_col] = pd.to_datetime(b[time_col])
    h[time_col] = pd.to_datetime(h[time_col])
    b = b.sort_values(time_col).reset_index(drop=True)
    h = h.sort_values(time_col).reset_index(drop=True)

    merged = pd.merge_asof(b,h,on = time_col,direction="backward")
    for c in ["p0","p1","p2"]:
        # Uniformly fill NaNs
        merged[c] = merged[c].astype("float").fillna(1/3)
    return merged

def df_to_arrays(df:pd.DataFrame,feature_cols:List[str],close_col:str = "close_5m",dropna:bool = True)->Tuple[np.ndarray,np.ndarray]:
    use = df[feature_cols+[close_col]].copy()
    if dropna:
        use = use.dropna().reset_index(drop=True)
    X = use[feature_cols].to_numpy(dtype = np.float32)
    close = use[close_col].to_numpy(dtype=np.float32)
    return X,close

# Environment
@dataclass
class TradingEnvConfig:
    fee_bps: float = 2.0
    hold_cost_bps : float = 0.0
    max_episode_steps : Optional[int] = 5000
    random_start: bool = True
    start_index: int = 1
    seed: int = 42

class TradingEnv:
    def __init__(self,X,close,cfg = TradingEnvConfig()):
        assert isinstance(X,np.ndarray) and isinstance(close,np.ndarray)
        assert X.ndim == 2 and close.ndim == 1
        assert len(X) == len(close)
        assert cfg.start_index >= 1

        self.X = X.astype(np.float32)
        self.close = close.astype(np.float32)
        self.cfg = cfg

        self.fee = cfg.fee_bps / 1e4
        self.hold_cost = cfg.hold_cost_bps/1e4
        self.rng = np.random.default_rng(cfg.seed)
        self.T = len(close)

        self.t = cfg.start_index
        self.pos = 0
        self.steps = 0
        self.episode_start = cfg.start_index

    def obs_dim(self)->int:
        return self.X.shape[1]
    def reset(self)-> np.ndarray:
        self.pos = 0
        self.steps = 0

        if self.cfg.random_start and self.cfg.max_episode_steps is not None:
            max_start = self.T - self.cfg.max_episode_steps - 1
            max_start = max(max_start,self.cfg.start_index)
            self.episode_start = int(self.rng.integers(self.cfg.start_index,max_start+1))
        else:
            self.episode_start = self.cfg.start_index

        self.t = self.episode_start
        return self.X[self.t]
    
    def step(self,action:int):

        if action not in (-1,0,1):
            raise ValueError("Action Error")
        new_pos = action

        # LogReturn
        ret = float(np.log((self.close[self.t]+1e-12)/(self.close[self.t-1]+1e-12)))
        turnover = abs(new_pos - self.pos)
        cost_fee = self.fee * turnover
        cost_hold = self.hold_cost * abs(self.pos)
        reward = self.pos * ret - cost_fee - cost_hold

        self.pos = new_pos
        self.t += 1
        self.steps += 1

        done = self.t >= self.T
        if self.cfg.max_episode_steps is not None:
            done = done or (self.steps >= self.cfg.max_episode_steps)
        obs = self.X[self.t-1] if not done else self.X[self.T-1]

        info = {

            "t":self.t - 1,
            "ret":ret,
            "pos":self.pos,
            "turnover":turnover,
            "fee":cost_fee,
            "hold_cost":cost_hold,
        }

        return obs, float(reward), bool(done), info
    
# PPO Agent:

class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    lr: float = 3e-4
    max_grad_norm: float = 0.5
    rollout_steps: int = 2048
    minibatch_size: int = 256
    update_epochs: int = 10
    hidden: int = 128

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class ActorCritic(nn.Module):
    def __init__(self,obs_dim,n_actions=3,hidden=128):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim,hidden),
            nn.ReLU(),
            nn.Linear(hidden,hidden),
            nn.ReLU(),
        )
        self.pi = nn.Linear(hidden,n_actions)
        self.v = nn.Linear(hidden,1)
    
    def forward(self,x):
        h = self.backbone(x)
        logits = self.pi(h)
        value = self.v(h).squeeze(-1)
        return logits,value
class RollingBuffer:

    def __init__(self,size,obs_dim,device):
        self.size = size
        self.device = device
        self.obs = torch.zeros((size,obs_dim),device=device)
        self.actions = torch.zeros((size,),dtype = torch.long,device=device)
        self.rewards = torch.zeros((size,),device=device)
        self.dones = torch.zeros((size,),device=device)
        self.logprobs = torch.zeros((size,),device=device)
        self.values = torch.zeros((size,),device=device)

        self.advantages = torch.zeros((size,),device=device)
        self.returns = torch.zeros((size,),device=device)
        self.ptr = 0

    def add(self,obs,act_idx,reward,done,logprob,value):
        i = self.ptr
        self.obs[i] = obs
        self.actions[i] = act_idx
        self.rewards[i] = reward
        self.dones[i] = done
        self.logprobs[i] = logprob
        self.values[i] = value
        self.ptr += 1

    def compute_gae(self,last_value,gamma,lam):
        adv = 0.0
        for t in reversed(range(self.size)):
            mask = 1.0 - self.dones[t]
            next_value = last_value if t == self.size -1 else self.values[t+1]
            delta = self.rewards[t] + gamma*next_value*mask - self.values[t]
            adv = delta + gamma*lam*mask*adv
            self.advantages[t] = adv
        # returns = advantages + values (for value function training)
        self.returns = self.advantages + self.values
    def minibatches(self,batch_size):
        idx = torch.randperm(self.size,device=self.device)
        for start in range(0,self.size,batch_size):
            mb = idx[start:start + batch_size]
            yield self.obs[mb], self.actions[mb],self.logprobs[mb],self.advantages[mb],self.returns[mb]


class PPOAgent:

    IDX_TO_ACT = np.array([-1,0,1],dtype = np.int32)

    def __init__(self,env,cfg:PPOConfig,n_actions = 3):
        self.env = env
        self.cfg = cfg
        self.device = cfg.device
        obs_dim = env.X.shape[1]
        self.model = ActorCritic(obs_dim=obs_dim,n_actions=n_actions,hidden = cfg.hidden).to(self.device)
        self.opt = optim.Adam(self.model.parameters(),lr = cfg.lr)

    def act(self,obs):
        logits,value = self.model(obs.unsqueeze(0))
        dist = torch.distributions.Categorical(logits=logits)
        act_idx = dist.sample()
        logprob = dist.log_prob(act_idx)
        return act_idx.squeeze(0), logprob, value.squeeze(0)
    
    def train(self,total_updates = 200,log_every = 10):
        obs_dim = self.env.X.shape[1]
        obs_np = self.env.reset()
        obs = torch.tensor(obs_np,device = self.device)

        for upd in range(total_updates):
            buf = RollingBuffer(self.cfg.rollout_steps,obs_dim,self.device)

            for _ in range(self.cfg.rollout_steps):
                with torch.no_grad():
                    act_idx, logprob, value = self.act(obs)
                env_action = int(self.IDX_TO_ACT[int(act_idx.item())])
                next_obs_np, reward, done, _info = self.env.step(env_action)

                buf.add(obs=obs,
                        act_idx=act_idx,
                        reward = torch.tensor(reward,device = self.device),
                        done = torch.tensor(float(done),device = self.device),
                        logprob=logprob,
                        value = value,)
                
                if done:
                    obs_np = self.env.reset()
                else:
                    obs_np = next_obs_np
                obs = torch.tensor(obs_np,device=self.device)


            with torch.no_grad():
                _, last_value = self.model(obs.unsqueeze(0))
            buf.compute_gae(float(last_value.item()),self.cfg.gamma,self.cfg.gae_lambda)

            last_pi_loss = last_v_loss = last_ent = 0.0

            for _ in range(self.cfg.update_epochs):
                for mb_obs, mb_act, mb_oldlog, mb_adv, mb_ret in buf.minibatches(self.cfg.minibatch_size):
                    logits,values = self.model(mb_obs)
                    dist = torch.distributions.Categorical(logits = logits)
                    newlog = dist.log_prob(mb_act)
                    entropy = dist.entropy().mean()

                    ratio = torch.exp(newlog-mb_oldlog)
                    surr1 = ratio * mb_adv
                    surr2 = torch.clamp(ratio,1.0-self.cfg.clip_eps,1.0+self.cfg.clip_eps)*mb_adv
                    pi_loss = -torch.min(surr1,surr2).mean()

                    v_loss = ((values-mb_ret)**2).mean()
                    loss = pi_loss + self.cfg.vf_coef*v_loss-self.cfg.ent_coef*entropy

                    self.opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(),self.cfg.max_grad_norm)
                    self.opt.step()

                    last_pi_loss = float(pi_loss.item())
                    last_v_loss = float(v_loss.item())
                    last_ent = float(entropy.item())

            if (upd + 1) % log_every == 0:
                print(f"[PPO] upd={upd+1}/{total_updates}")

        return self
    

    def save(self,path):
        torch.save(
            {
                "state_dict":self.model.state_dict(),
                "cfg":self.cfg 

            },
            path,
        )

    def load(self,path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["state_dict"])
        return self
    



if __name__ == "__main__":
    cfg = PPOConfig()
    bar_p = "BTCUSDT_combined_klines_20210201_20260201_states4_labeled.csv"
    # hour_p = "hour"
    bar = pd.read_csv(bar_p)
    # hourly = pd.read_csv(hour_p)

    # df = align_hourly_regimes(bar,hourly,time_col="datetime")
    df = bar.copy()
    # TODO: 指定特征列，例如：
    # feature_cols = ["log_ret_1_5m", "ema_ratio_9_21_5m", "macd_hist_5m", "adx_5m", 
    #                 "atr_norm_5m", "bb_width_5m", "rsi_14_5m", "volume_zscore_50_5m"]
    feature_cols = ["volume_5m","log_ret_1_5m","ema_ratio_9_21_5m",
                    "macd_hist_5m","adx_5m","atr_norm_5m","bb_width_5m","rsi_14_5m", "volume_zscore_50_5m",
                    "state"]

    X,close = df_to_arrays(df,feature_cols,close_col="close_5m",dropna=True)
    env_cfg = TradingEnvConfig(fee_bps=2.0,max_episode_steps=5000,random_start=True,seed=42)
    env = TradingEnv(X,close,env_cfg)
    agent = PPOAgent(env, cfg)
    agent.train(total_updates=200)
    agent.save("ppo_policy.pt")
    print("Saved")



        



