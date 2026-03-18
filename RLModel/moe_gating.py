# 使用监督形式的router进行，推理不依赖
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from ppo import PPOConfig,TradingEnv,TradingEnvConfig
from ppo_test import summarize,plot_backtest,buy_and_hold
from moe_hard import(
    MoEDataConfig,
    HardMoEAgent,
    _load_moe_data,
    _ppo_cfg_to_dict,
    _ppo_config_from_meta,
    _prepare_moe_arrays,
    train_moe_experts
)

@dataclass
class RouterTrainConfig:
    epochs:int = 20
    batch_size:int = 128
    lr:float = 1e-3
    val_ratio:float = 0.1
    weight_decay:float = 1e-5
    class_balance:bool = True

class RouteNet(nn.Module):
    def __init__(self,obs_dim,hidden,n_experts ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim,hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden,hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden,n_experts),
        )
    def forward(self,x):
        return self.net(x)
    

class ConservativeGatedMoEAgent(HardMoEAgent):
    # 推理时由router选择expert，不依赖外部regime

    def __init__(self,obs_dims,cfg,n_actions,n_experts,router_hidden,router_lr):
        super().__init__(obs_dims=obs_dims,cfg=cfg,n_actions=n_actions,n_experts=n_experts)
        hidden = cfg.hidden if router_hidden is None else router_hidden

        lr = cfg.lr if router_lr is None else router_lr
        self.router = RouteNet(obs_dim=obs_dims,hidden=hidden,n_experts=n_experts).to(self.device)
        self.router_optimizer = optim.Adam(self.router.parameters(),lr = lr)

    def train_mode(self):
        super().train_mode()
        self.router.train()
    def eval_mode(self):
        super().eval_mode()
        self.router.eval()

    def router_logits(self,obs):
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        return self.router(obs)
    
    @ torch.no_grad()
    def route(self,obs,deterministic = True):
        logits = self.router_logits(obs)
        probs = torch.softmax(logits,dim = -1)
        if deterministic:
            expert_id = int(torch.argmax(probs,dim = -1).item())
        else:
            dist = torch.distributions.Categorical(logits=logits)
            expert_id = int(dist.sample().item())
        confidence = float(probs[0,expert_id].item())
        return expert_id,confidence,probs.squeeze(0)
    
    @ torch.no_grad()
    def act_deterministic_routed(self,obs):
        expert_id,confidence,probs = self.route(obs,deterministic=True)
        action,value = self.act_deterministic(obs,expert_id)
        return action,value,expert_id,confidence,probs