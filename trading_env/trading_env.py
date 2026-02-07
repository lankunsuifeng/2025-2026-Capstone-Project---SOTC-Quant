"""
加密货币日内交易环境
用于 stable-baselines3 的 PPO agent 训练
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict, Any, Optional

from .config import (
    INITIAL_CAPITAL,
    TRANSACTION_FEE_RATE,
    MAX_POSITION_RATIO,
    LOOKBACK_WINDOW,
    NUM_REGIMES,
    DTYPE
)
from .reward import calculate_reward


class CryptoTradingEnv(gym.Env):
    """
    加密货币交易环境
    
    状态空间：
    - 价格特征（LOOKBACK_WINDOW 根 K 线）：open, high, low, close, volume
    - 当前仓位信息：[持仓比例(0~1), 当前回报率]
    - Regime 概率向量：[p1, p2, ..., pN]
    
    动作空间：
    - 0: Hold（持仓不动）
    - 1: Buy（买入）
    - 2: Sell（卖出全部仓位）
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(self, data: pd.DataFrame, render_mode: Optional[str] = None):
        """
        初始化环境
        
        Args:
            data: DataFrame，列名为 ['open', 'high', 'low', 'close', 'volume', 'regime_probs']
                 其中 regime_probs 这一列存的是 list 或 numpy array（classifier 的输出）
            render_mode: 渲染模式
        """
        super().__init__()
        
        # 验证数据格式
        required_columns = ['open', 'high', 'low', 'close', 'volume', 'regime_probs']
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"数据必须包含以下列: {required_columns}")
        
        # 复制数据并重置索引
        self.data = data.copy().reset_index(drop=True)
        self.n_steps = len(self.data)
        
        if self.n_steps < LOOKBACK_WINDOW + 1:
            raise ValueError(f"数据长度 ({self.n_steps}) 必须至少为 {LOOKBACK_WINDOW + 1}")
        
        # 解析 regime_probs 列
        self._parse_regime_probs()
        
        # 计算状态空间维度
        # 价格特征: LOOKBACK_WINDOW × 5 (open, high, low, close, volume)
        price_features_dim = LOOKBACK_WINDOW * 5
        # 仓位信息: 2 (持仓比例, 当前回报率)
        position_info_dim = 2
        # Regime 概率: NUM_REGIMES
        regime_dim = NUM_REGIMES
        
        self.observation_dim = price_features_dim + position_info_dim + regime_dim
        
        # 定义观察空间和动作空间
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.observation_dim,),
            dtype=DTYPE
        )
        self.action_space = spaces.Discrete(3)
        
        self.render_mode = render_mode
        
        # 初始化状态变量
        self.reset()
    
    def _parse_regime_probs(self):
        """解析 regime_probs 列，确保格式统一为 numpy array"""
        regime_probs_list = []
        for idx, probs in enumerate(self.data['regime_probs']):
            if isinstance(probs, (list, tuple)):
                probs = np.array(probs, dtype=DTYPE)
            elif isinstance(probs, np.ndarray):
                probs = probs.astype(DTYPE)
            else:
                raise ValueError(f"第 {idx} 行的 regime_probs 格式不正确")
            
            if len(probs) != NUM_REGIMES:
                raise ValueError(f"第 {idx} 行的 regime_probs 长度 ({len(probs)}) 必须等于 NUM_REGIMES ({NUM_REGIMES})")
            
            regime_probs_list.append(probs)
        
        self.regime_probs_array = np.array(regime_probs_list, dtype=DTYPE)
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        重置环境
        
        Returns:
            observation: 初始观察
            info: 信息字典
        """
        super().reset(seed=seed)
        
        # 重置到初始状态
        self.current_step = LOOKBACK_WINDOW  # 从 LOOKBACK_WINDOW 开始，确保有足够的历史数据
        self.cash = INITIAL_CAPITAL
        self.position = 0.0  # 持仓数量（币的数量）
        self.portfolio_value_previous = INITIAL_CAPITAL
        self.portfolio_value_peak = INITIAL_CAPITAL
        
        # 计算初始观察
        observation = self._get_observation()
        
        info = {
            'portfolio_value': INITIAL_CAPITAL,
            'position': 0.0,
            'cash': INITIAL_CAPITAL,
            'current_step': self.current_step
        }
        
        return observation.astype(DTYPE), info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        执行一步动作
        
        Args:
            action: 动作 (0=Hold, 1=Buy, 2=Sell)
            
        Returns:
            observation: 新观察
            reward: 奖励
            terminated: 是否终止（到达最后一步）
            truncated: 是否截断（本环境不使用）
            info: 信息字典
        """
        # 执行交易动作
        self._execute_action(action)
        
        # 更新步数
        self.current_step += 1
        
        # 计算当前资产总值
        current_price = self.data.iloc[self.current_step - 1]['close']
        portfolio_value_current = self.cash + self.position * current_price
        
        # 更新峰值
        if portfolio_value_current > self.portfolio_value_peak:
            self.portfolio_value_peak = portfolio_value_current
        
        # 计算奖励
        reward = calculate_reward(
            portfolio_value_current,
            self.portfolio_value_previous,
            self.portfolio_value_peak
        )
        
        # 更新上一步资产总值
        self.portfolio_value_previous = portfolio_value_current
        
        # 检查是否终止
        terminated = (self.current_step >= self.n_steps)
        
        # 获取新观察
        if terminated:
            # 如果已经终止，返回最后一步的观察
            observation = self._get_observation()
        else:
            observation = self._get_observation()
        
        # 计算当前回报率
        total_return = (portfolio_value_current - INITIAL_CAPITAL) / INITIAL_CAPITAL
        position_ratio = (self.position * current_price) / portfolio_value_current if portfolio_value_current > 0 else 0.0
        
        info = {
            'portfolio_value': portfolio_value_current,
            'position': self.position,
            'cash': self.cash,
            'current_step': self.current_step,
            'total_return': total_return,
            'position_ratio': position_ratio
        }
        
        return observation.astype(DTYPE), reward, terminated, False, info
    
    def _execute_action(self, action: int):
        """
        执行交易动作
        
        Args:
            action: 动作 (0=Hold, 1=Buy, 2=Sell)
        """
        current_price = self.data.iloc[self.current_step]['close']
        
        if action == 0:  # Hold
            pass
        
        elif action == 1:  # Buy
            # 计算可用资金
            available_cash = self.cash * MAX_POSITION_RATIO
            
            # 计算可以买入的数量（扣除手续费）
            # 买入后资产 = cash - cost + position * price
            # cost = quantity * price * (1 + fee_rate)
            # 所以: quantity = available_cash / (price * (1 + fee_rate))
            if available_cash > 0 and current_price > 0:
                quantity = available_cash / (current_price * (1 + TRANSACTION_FEE_RATE))
                
                # 如果现金足够买一个最小单位，执行买入
                if quantity > 0:
                    cost = quantity * current_price * (1 + TRANSACTION_FEE_RATE)
                    if cost <= self.cash:
                        self.position += quantity
                        self.cash -= cost
        
        elif action == 2:  # Sell
            # 卖出全部仓位
            if self.position > 0 and current_price > 0:
                # 卖出收入 = position * price * (1 - fee_rate)
                revenue = self.position * current_price * (1 - TRANSACTION_FEE_RATE)
                self.cash += revenue
                self.position = 0.0
        
        else:
            raise ValueError(f"无效动作: {action}")
    
    def _get_observation(self) -> np.ndarray:
        """
        获取当前观察
        
        Returns:
            observation: 拼接后的状态向量
        """
        # 1. 价格特征（LOOKBACK_WINDOW 根 K 线）
        # 获取从 current_step - LOOKBACK_WINDOW 到 current_step - 1 的数据
        start_idx = max(0, self.current_step - LOOKBACK_WINDOW)
        end_idx = self.current_step
        
        price_data = self.data.iloc[start_idx:end_idx][['open', 'high', 'low', 'close', 'volume']].values
        
        # 如果数据不足 LOOKBACK_WINDOW，用第一行填充
        if len(price_data) < LOOKBACK_WINDOW:
            padding = np.tile(price_data[0:1], (LOOKBACK_WINDOW - len(price_data), 1))
            price_data = np.vstack([padding, price_data])
        
        # Min-max normalization（在当前 window 内归一化到 [0,1]）
        price_features = self._normalize_price_features(price_data)
        
        # 2. 当前仓位信息
        current_price = self.data.iloc[self.current_step - 1]['close']
        portfolio_value = self.cash + self.position * current_price
        
        if portfolio_value > 0:
            position_ratio = (self.position * current_price) / portfolio_value
        else:
            position_ratio = 0.0
        
        # 当前回报率（相对于初始资金）
        total_return = (portfolio_value - INITIAL_CAPITAL) / INITIAL_CAPITAL
        
        position_info = np.array([position_ratio, total_return], dtype=DTYPE)
        
        # 3. Regime 概率向量
        regime_probs = self.regime_probs_array[self.current_step - 1]
        
        # 拼接所有特征
        observation = np.concatenate([
            price_features.flatten(),
            position_info,
            regime_probs
        ]).astype(DTYPE)
        
        return observation
    
    def _normalize_price_features(self, price_data: np.ndarray) -> np.ndarray:
        """
        对价格特征进行 min-max normalization
        
        Args:
            price_data: 形状为 (LOOKBACK_WINDOW, 5) 的价格数据
            
        Returns:
            normalized_data: 归一化后的数据，范围 [0, 1]
        """
        # 对每一列（每个特征）分别进行归一化
        normalized = np.zeros_like(price_data, dtype=DTYPE)
        
        for col_idx in range(5):  # 5 个特征：open, high, low, close, volume
            col_data = price_data[:, col_idx]
            min_val = col_data.min()
            max_val = col_data.max()
            
            if max_val > min_val:
                normalized[:, col_idx] = (col_data - min_val) / (max_val - min_val)
            else:
                # 如果最大值等于最小值，设为 0.5
                normalized[:, col_idx] = 0.5
        
        return normalized
    
    def render(self):
        """渲染当前状态"""
        if self.render_mode == "human" or self.render_mode is None:
            current_price = self.data.iloc[self.current_step - 1]['close']
            portfolio_value = self.cash + self.position * current_price
            total_return = (portfolio_value - INITIAL_CAPITAL) / INITIAL_CAPITAL
            
            print(f"Step: {self.current_step}/{self.n_steps}")
            print(f"Price: {current_price:.2f}")
            print(f"Cash: {self.cash:.2f}")
            print(f"Position: {self.position:.6f}")
            print(f"Portfolio Value: {portfolio_value:.2f}")
            print(f"Total Return: {total_return*100:.2f}%")
            print("-" * 40)
