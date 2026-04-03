# PPO Trader with Volatility Regime Switching

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, Dict, Optional,List
import matplotlib.pyplot as plt
import os
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

def df_to_arrays(df:pd.DataFrame,feature_cols:List[str],close_col:str = "close",dropna:bool = True)->Tuple[np.ndarray,np.ndarray]:
    use = df[feature_cols+[close_col]].copy()
    if dropna:
        use = use.dropna().reset_index(drop=True)
    X = use[feature_cols].to_numpy(dtype = np.float32)
    close = use[close_col].to_numpy(dtype=np.float32)
    return X,close


def _rolling_sharpe_past_bars(close: np.ndarray, t: int, window: int, eps: float = 1e-8) -> float:
    """
    Sharpe-like ratio from **past** simple returns only (no future close).
    Uses ``close[t-window : t+1]`` → ``window`` returns ending at bar t-1→t.
    """
    if window < 2 or t < window:
        return 0.0
    seg = close[t - window : t + 1].astype(np.float64, copy=False)
    prev = seg[:-1]
    rets = (seg[1:] - prev) / np.maximum(prev, 1e-15)
    if rets.size < 2:
        return 0.0
    m = float(np.mean(rets))
    s = float(np.std(rets, ddof=1))
    return float(m / (s + eps))


# Environment
@dataclass
class TradingEnvConfig:
    fee_bps: float = 2.0
    hold_cost_bps : float = 2.0
    max_episode_steps : Optional[int] = 5000
    random_start: bool = True
    start_index: int = 1
    seed: int = 42
    # Reward uses cumulative return over the next ``forward_return_bars`` bars (clipped near series end).
    # Mark-to-market / backtest still use info["ret"] = 1-bar return (t -> t+1).
    forward_return_bars: int = 1
    # In reward only: multiply fee penalty by this (<1 encourages turnover). info["fee"] stays full fee.
    fee_reward_discount: float = 0.9
    # Past-only rolling Sharpe (window simple returns ending at t); 0 disables.
    rolling_sharpe_window: int = 0
    rolling_sharpe_coef: float = 0.01
    rolling_sharpe_clip: float = 5.0


class TradingEnv:
    def __init__(self,X,close,cfg = TradingEnvConfig(),regime_ids=None):
        assert isinstance(X,np.ndarray) and isinstance(close,np.ndarray)
        assert X.ndim == 2 and close.ndim == 1
        assert len(X) == len(close)
        assert cfg.start_index >= 1
        
        if regime_ids is not None:
            regime_ids = np.asarray(regime_ids,dtype = np.int64)
            assert regime_ids.ndim == 1
            assert len(regime_ids) == len(close)

        self.X = X.astype(np.float32)
        self.close = close.astype(np.float32)
        self.cfg = cfg
        self.regime = regime_ids
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
    def reset(self) -> np.ndarray:
        self.pos = 0
        self.steps = 0

        h_need = max(1, int(self.cfg.forward_return_bars))
        if self.cfg.random_start and self.cfg.max_episode_steps is not None:
            # First step needs t + h_need <= T - 1 for full forward window when possible
            max_start = self.T - self.cfg.max_episode_steps - 1
            max_start = max(max_start, self.cfg.start_index)
            max_start = min(max_start, self.T - 2, self.T - 1 - h_need)
            if max_start < self.cfg.start_index:
                max_start = self.cfg.start_index
            self.episode_start = int(self.rng.integers(self.cfg.start_index, max_start + 1))
        else:
            self.episode_start = min(self.cfg.start_index, self.T - 2, self.T - 1 - h_need)
            if self.episode_start < self.cfg.start_index:
                self.episode_start = min(self.cfg.start_index, self.T - 2)

        self.t = self.episode_start
        return self.X[self.t]
    
    def step(self,action:int):

        if action not in (-1,0,1):
            raise ValueError("Action Error")
        
        if self.t >= self.T - 1:
            raise RuntimeError("No Future Data Available")
        new_pos = action

        t = self.t
        h = max(1, int(self.cfg.forward_return_bars))
        max_ahead = (self.T - 1) - t
        h_eff = int(min(h, max_ahead))

        # 1-bar return: used by backtest (mark-to-market) and logged as info["ret"]
        ret_1 = (self.close[t + 1] + 1e-15) / (self.close[t] + 1e-15) - 1.0
        # Multi-bar cumulative return: used in reward (same position held over h_eff bars)
        ret_fwd = (self.close[t + h_eff] + 1e-15) / (self.close[t] + 1e-15) - 1.0

        turnover = abs(new_pos - self.pos)
        cost_fee = self.fee * turnover
        cost_hold = self.hold_cost * (abs(new_pos)-1) * float(h_eff)
        fee_in_reward = cost_fee * float(self.cfg.fee_reward_discount)

        sharpe_bonus = 0.0
        roll_sh = 0.0
        w_sh = int(self.cfg.rolling_sharpe_window)
        if w_sh >= 2:
            roll_sh = _rolling_sharpe_past_bars(self.close, t, w_sh)
            c = float(self.cfg.rolling_sharpe_coef)
            clip = float(self.cfg.rolling_sharpe_clip)
            if clip > 0:
                roll_sh = float(np.clip(roll_sh, -clip, clip))
            sharpe_bonus = c * roll_sh

        reward = new_pos * ret_fwd - fee_in_reward - cost_hold + sharpe_bonus
        reward = 2000 * reward  #Scale
        self.pos = new_pos
        self.t += 1
        self.steps += 1

        done = self.t >= self.T-1
        if self.cfg.max_episode_steps is not None:
            done = done or (self.steps >= self.cfg.max_episode_steps)
        next_t = min(self.t, self.T - 1)
        obs = self.X[next_t]

        info = {

            "t":next_t,
            "ret": ret_1,
            "ret_forward": ret_fwd,
            "forward_horizon_used": h_eff,
            "pos":self.pos,
            "turnover":turnover,
            "fee":cost_fee,
            "fee_in_reward": float(fee_in_reward),
            "hold_cost":cost_hold,
            "rolling_sharpe_past": float(roll_sh),
            "sharpe_bonus_unscaled": float(sharpe_bonus),
        }
        if self.regime is not None:
            info["regime"] = int(self.regime[next_t])
        return obs, float(reward), bool(done), info
    
# PPO Agent:

class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.1
    vf_coef: float = 0.5
    lr: float = 3e-4
    max_grad_norm: float = 0.5
    rollout_steps: int = 2048
    minibatch_size: int = 256
    update_epochs: int = 10
    hidden:int = 128
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class ActorCritic(nn.Module):
    def __init__(self,obs_dim,n_actions=3,hidden=128):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim,hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden,hidden),
            nn.LayerNorm(hidden),
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
        adv = torch.zeros((), device=self.device)
        for t in reversed(range(self.size)):
            mask = 1.0 - self.dones[t]
            next_value = last_value if t == self.size -1 else self.values[t+1]
            delta = self.rewards[t] + gamma*next_value*mask - self.values[t]
            adv = delta + gamma*lam*mask*adv
            self.advantages[t] = adv
        self.returns[:] = self.advantages + self.values
        adv_mean = self.advantages.mean()
        adv_std = self.advantages.std() + 1e-12
        self.advantages[:] = (self.advantages - adv_mean)/adv_std
        
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
        return act_idx.squeeze(0),logprob.squeeze(0), value.squeeze(0)
    
    def train(self, total_updates=200, log_every=10, plot_path: str = "RLModel/train_plots/training_curves.png"):
        os.makedirs(os.path.dirname(plot_path) or ".", exist_ok=True)

        obs_dim = self.env.X.shape[1]
        obs_np = self.env.reset()
        obs = torch.tensor(obs_np, device=self.device)

        history = {
            "upd": [],
            "rollout_reward_mean": [],
            "rollout_reward_sum": [],
            "rollout_abspos_mean": [],
            "rollout_turnover_mean": [],
            "rollout_fee_sum": [],
            "pi_loss": [],
            "v_loss": [],
            "entropy": [],
        }

        for upd in range(total_updates):
            buf = RollingBuffer(self.cfg.rollout_steps, obs_dim, self.device)

            # --- rollout stats ---
            r_sum = 0.0
            fee_sum = 0.0
            abspos_sum = 0.0
            turnover_sum = 0.0

            for _ in range(self.cfg.rollout_steps):
                with torch.no_grad():
                    act_idx, logprob, value = self.act(obs)

                env_action = int(self.IDX_TO_ACT[int(act_idx.item())])
                next_obs_np, reward, done, info = self.env.step(env_action)

                # stats (use info from env)
                r_sum += float(reward)
                fee_sum += float(info.get("fee", 0.0))
                abspos_sum += abs(float(info.get("pos", 0.0)))
                turnover_sum += float(info.get("turnover", 0.0))

                buf.add(
                    obs=obs,
                    act_idx=act_idx,
                    reward=torch.tensor(reward, device=self.device),
                    done=torch.tensor(float(done), device=self.device),
                    logprob=logprob,
                    value=value,
                )

                obs_np = self.env.reset() if done else next_obs_np
                obs = torch.tensor(obs_np, device=self.device)

            with torch.no_grad():
                _, last_value = self.model(obs.unsqueeze(0))
            buf.compute_gae(float(last_value.item()), self.cfg.gamma, self.cfg.gae_lambda)

            pi_sum = v_sum = ent_sum = 0.0
            n_mb = 0.0

            # --- PPO update ---
            for _ in range(self.cfg.update_epochs):
                for mb_obs, mb_act, mb_oldlog, mb_adv, mb_ret in buf.minibatches(self.cfg.minibatch_size):
                    logits, values = self.model(mb_obs)
                    dist = torch.distributions.Categorical(logits=logits)
                    newlog = dist.log_prob(mb_act)
                    entropy = dist.entropy().mean()

                    ratio = torch.exp(newlog - mb_oldlog)
                    surr1 = ratio * mb_adv
                    surr2 = torch.clamp(ratio, 1.0 - self.cfg.clip_eps, 1.0 + self.cfg.clip_eps) * mb_adv
                    pi_loss = -torch.min(surr1, surr2).mean()

                    v_loss = ((values - mb_ret) ** 2).mean()
                    loss = pi_loss + self.cfg.vf_coef * v_loss - self.cfg.ent_coef * entropy

                    self.opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                    self.opt.step()

                    pi_sum += float(pi_loss.item())
                    v_sum  += float(v_loss.item())
                    ent_sum += float(entropy.item())
                    n_mb += 1
            avg_pi_loss = pi_sum/max(n_mb,1)
            avg_v_loss = v_sum/max(n_mb,1)
            avg_ent = ent_sum/max(n_mb,1)

            # --- logging & record ---
            if (upd + 1) % log_every == 0 or upd == 0:
                r_mean = r_sum / self.cfg.rollout_steps
                abspos_mean = abspos_sum / self.cfg.rollout_steps
                turnover_mean = turnover_sum / self.cfg.rollout_steps

                history["upd"].append(upd + 1)
                history["rollout_reward_mean"].append(r_mean)
                history["rollout_reward_sum"].append(r_sum)
                history["rollout_abspos_mean"].append(abspos_mean)
                history["rollout_turnover_mean"].append(turnover_mean)
                history["rollout_fee_sum"].append(fee_sum)
                history["pi_loss"].append(avg_pi_loss)
                history["v_loss"].append(avg_v_loss)
                history["entropy"].append(avg_ent)

                print(
                    f"[PPO] upd={upd+1}/{total_updates} "
                    f"r_mean={r_mean:.3e} r_sum={r_sum:.3e} "
                    f"|pos|={abspos_mean:.3f} turn={turnover_mean:.3f} fee_sum={fee_sum:.3e} "
                    f"pi={avg_pi_loss:.3e} v={avg_v_loss:.3e} ent={avg_ent:.3e}"
                )

        # --- plot after training ---
        if len(history["upd"]) > 0:
            x = np.array(history["upd"], dtype=int)

            import matplotlib.pyplot as plt
            plt.figure()
            plt.plot(x, np.array(history["rollout_reward_mean"], dtype=float))
            plt.title("Rollout Reward Mean")
            plt.xlabel("update")
            plt.ylabel("mean reward")
            plt.savefig(plot_path.replace(".png", "_reward_mean.png"), dpi=150, bbox_inches="tight")
            plt.close()

            plt.figure()
            plt.plot(x, np.array(history["rollout_reward_sum"], dtype=float))
            plt.title("Rollout Reward Sum")
            plt.xlabel("update")
            plt.ylabel("sum reward")
            plt.savefig(plot_path.replace(".png", "_reward_sum.png"), dpi=150, bbox_inches="tight")
            plt.close()

            plt.figure()
            plt.plot(x, np.array(history["rollout_abspos_mean"], dtype=float), label="mean |pos|")
            plt.plot(x, np.array(history["rollout_turnover_mean"], dtype=float), label="mean turnover")
            plt.title("Position / Turnover")
            plt.xlabel("update")
            plt.legend()
            plt.savefig(plot_path.replace(".png", "_pos_turn.png"), dpi=150, bbox_inches="tight")
            plt.close()

            plt.figure()
            plt.plot(x, np.array(history["entropy"], dtype=float))
            plt.title("Policy Entropy")
            plt.xlabel("update")
            plt.ylabel("entropy")
            plt.savefig(plot_path.replace(".png", "_entropy.png"), dpi=150, bbox_inches="tight")
            plt.close()

            plt.figure()
            plt.plot(x, np.array(history["pi_loss"], dtype=float), label="pi_loss")
            plt.plot(x, np.array(history["v_loss"], dtype=float), label="v_loss")
            plt.title("Losses")
            plt.xlabel("update")
            plt.legend()
            plt.savefig(plot_path.replace(".png", "_loss.png"), dpi=150, bbox_inches="tight")
            plt.close()

            # optional: save history to csv
            hist_df = pd.DataFrame(history)
            hist_csv = plot_path.replace(".png", "_history.csv")
            hist_df.to_csv(hist_csv, index=False)
            print(f"[train] saved plots + history: {plot_path.replace('.png','_*.png')} and {hist_csv}")

        return self
        

    def save(self,path):
        torch.save(
            {
                "state_dict":self.model.state_dict()
            },
            path,
        )

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt["state_dict"])
        return self
    



if __name__ == "__main__":
    cfg = PPOConfig()

    bar = pd.read_csv("data/data_e.csv")
    _excl = [c for c in ("timestamp", "close_5m") if c in bar.columns]
    feature_cols = bar.columns.drop(_excl).tolist()
    df = bar.copy()
    spilt_idx = int(len(df) * 0.8)
    df_train = df.iloc[:spilt_idx].copy()
    df_test = df.iloc[spilt_idx:].copy()
    X,close = df_to_arrays(df_train,feature_cols,close_col="close_5m",dropna=True)
    env_cfg = TradingEnvConfig(fee_bps=0.2,hold_cost_bps=0.2,max_episode_steps=10000,random_start=True,seed=42)
    env = TradingEnv(X,close,env_cfg)
    agent = PPOAgent(env, cfg)
    agent.train(total_updates=200)
    agent.save("model/ppo_policy.pt")
    print("Saved")



        



