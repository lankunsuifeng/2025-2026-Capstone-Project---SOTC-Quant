# backtest.py
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import os
from dataclasses import dataclass
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


@dataclass
class BacktestPaths:
    policy_path: str = "model/ppo_policy.pt"
    test_csv : str = "data/data_e.csv"
    out_csv: str = "result/backtest_steps.csv"
    plot_dir: str = "plots"


class PPOBacktesting:
    def __init__(self,env_cfg,ppo_cfg,close_col = "close_5m",ts_col = "timestamp"):
        pass

    