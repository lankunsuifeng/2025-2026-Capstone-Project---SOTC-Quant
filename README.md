# 2025–2026 Capstone — SOTC-Quant

面向加密货币（以 BTCUSDT 5m 为主）的 **市场状态（HMM）+ LSTM 预测 + 强化学习交易** 流水线：在 **HMMLSTM** 中完成数据、特征与 regime 标注；在 **RLModel** 中构建 `data_e`、训练 **PPO** 与 **Hard MoE**，并与买入持有等指标对齐评估。

---

## 仓库结构（与当前工作相关）

| 路径 | 作用 |
|------|------|
| `HMMLSTM/cryptoregimeclassifier/` | Streamlit 看板、特征工程、HMM 标注、LSTM 训练/调参、历史预测导出 |
| `RLModel/` | `data_process` → `data_e.csv`；`ppo.py` / `moe_hard.py`；`experiment_runner.py` 统一五组实验 |
| `split_config.json`（仓库根目录） | **时间切分单一事实来源**：HMM 拟合截止、LSTM 训练截止、RL 训练窗、测试起点（ISO-8601 UTC） |
| `RLModel/split_bounds.py` | RL 侧读取 `split_config`、构造 RL 训练/测试 mask（与 HMMLSTM 中逻辑对齐） |
| `.cursor/plans/` | 项目规划与数据协议笔记（可选阅读） |

其他目录（如 `DP_FE/`、`Regime-Switch/`、`trading_env/`）为历史或并行实验代码，**主路径以 HMMLSTM + RLModel 为准**。

---

## 端到端流程

### 1. HMMLSTM：特征与 Regime 标签

1. **依赖**（建议在独立 venv 中安装）：

   ```bash
   cd HMMLSTM/cryptoregimeclassifier
   pip install -r requirements.txt
   ```

2. **启动看板**（多页面应用）：

   ```bash
   cd HMMLSTM/cryptoregimeclassifier/dashboard
   streamlit run app.py
   ```

3. **推荐页面顺序**（侧边栏）：

   - **Fetch data**：拉取/整理 K 线等到 `data/`  
   - **Compute features**：合并 5m 等源并计算指标；输入需匹配 `data/*combined*.csv` 命名习惯；输出一般为 `*_features.csv`  
   - **Regime classifier（HMM labeling）**：读取 `data/*_features.csv`，按 `split_config.json` 中的 **`hmm_fit_end`** 等做无泄露拟合，导出带 `state` / `regime` 的 `*_labeled.csv`  
   - **Model training（LSTM）**：在 labeled 数据上训练下一根 regime 预测；导出带 `lstm_predicted_regime` / 概率列等的预测 CSV（供下游合并）  
   - **HMM / LSTM tuning**、**Historical prediction**、**Live**：按需使用  

4. **与 RL 衔接**：将 **合并后的宽表**（含 5m OHLC/特征、HMM 与 LSTM 预测列）保存为 RLModel `data_process` 的 **`input_csv`**（文件名可与 `data_process.py` 中默认一致或自行指定）。

### 2. RLModel：构建 `data_e` 并跑实验

1. **安装**（示例，按你本地环境补全 `torch` 等）：

   ```bash
   cd RLModel
   pip install pandas numpy torch matplotlib scikit-learn joblib
   ```

2. **生成 `data/data_e.csv`**：运行 `data_process.data_engineering()`（可直接执行 `python data_process.py` 或从交互环境调用），指定 HMMLSTM 导出的合并 CSV。可选：

   - `normalize_features` / `winsorize_features`  
   - `multi_horizon_features=True`：滚动波动、多周期收益、ADX 斜率等（见 `data_process.py`）  
   - 归一化/winsor 的拟合窗与 **`split_config.json`** 中 **`lstm_train_end` / `rl_train_end`** 一致（RL 训练 mask 规则见 `split_bounds.rl_train_mask`）

3. **统一跑五组实验**（在 `RLModel/` 下）：

   ```bash
   python experiment_runner.py --csv data/data_e.csv --all
   ```

   只跑部分：

   ```bash
   python experiment_runner.py --csv data/data_e.csv --only ppo_base,moe_hmm_lstm
   ```

4. **实验 ID 与含义**

   | ID | 说明 |
   |----|------|
   | `ppo_base` | PPO，仅技术指标类特征（不含 HMM/LSTM regime 列） |
   | `ppo_hmm` | PPO + HMM 预测 state one-hot |
   | `ppo_hmm_lstm` | PPO + HMM + LSTM 预测 one-hot |
   | `moe_hmm` | Hard MoE，按 HMM one-hot 路由；观测不含 HMM one-hot |
   | `moe_hmm_lstm` | Hard MoE，按 LSTM one-hot 路由；观测含 HMM one-hot |

5. **主要输出**

   - 模型：`RLModel/model/experiments/<exp_id>/`  
   - 回测步序列、图：`RLModel/result/experiments/`  
   - 元数据：`RLModel/result/experiment_meta/<exp_id>.json`  
   - 汇总：`comparison_metrics.csv`、`equity_overlay.png` 等（由 runner 写入 `result_root`）

### 3. `experiment_runner.py` 常用参数

| 参数 | 说明 |
|------|------|
| `--csv` | `data_e.csv` 路径 |
| `--all` / `--only` | 全部或逗号分隔子集 |
| `--split-config` | `split_config.json` 路径；空则回退比例切分 |
| `--updates` / `--log-every` | 训练步数与日志间隔 |
| `--forward-return-bars` | `TradingEnv` 奖励中使用的**未来累计收益**根数（≥1）；观测仍无未来价 |
| `--fee-reward-discount` | 奖励里手续费惩罚系数（`<1` 略鼓励换手）；`info["fee"]` 回测仍为全额 |
| `--rolling-sharpe-window` | >0 时在奖励中加入**仅历史收益**的 rolling Sharpe 项；0 关闭 |
| `--rolling-sharpe-coef` | 上述 Sharpe 项系数 |

---

## `split_config.json` 字段（摘要）

- **`hmm_fit_end`**：HMM（及 scaler/PCA）拟合使用的数据截止时间（与 HMMLSTM 页面对齐）。  
- **`lstm_train_end`**：LSTM 训练截止时间。  
- **`rl_train_end`**：RL 训练窗上界（与 `test_start` 等共同定义 RL train/test）。  
- **`test_start`**：样本外测试起点（与 `rl_train_end` 取 max 得到有效测试起点，见 `split_bounds.effective_rl_test_start`）。

修改日期后需重新跑 HMMLSTM / `data_process` / RL，以保证全链路一致。

---

## 交易环境奖励（直觉说明）

`RLModel/ppo.py` 中 **`TradingEnv`**：每步仍按 bar 更新仓位（-1/0/1）。**奖励**大致为：

- 主项：**仓位 × 未来 `forward_return_bars` 根上的累计简单收益**（序列末尾自动缩短 horizon）；  
- 减：**打折后的换手手续费**、**与 horizon 对齐的持仓成本**；  
- 可选加：**过去窗口上的 rolling Sharpe 形状奖励**（不含未来价）。  

**回测净值**仍用 **`info["ret"]` 的 1-bar 收益** 与全额 `fee`，与训练奖励尺度可分离。

---

## 其他

- **早期独立 MVP 技术指标脚本**：见 `DP_FE/mvp_feature.py`（与 HMMLSTM `compute_features` 管线并行，非主路径）。  
- **远程仓库**：若已配置 `origin`，可与课程仓库 `2025-2026-Capstone-Project---SOTC-Quant` 同步。  

---

## Requirements（汇总）

- **HMMLSTM**：`HMMLSTM/cryptoregimeclassifier/requirements.txt`（含 TensorFlow 2.13、Streamlit、hmmlearn 等）。  
- **RLModel**：Python 3.10+ 推荐；`pandas`、`numpy`、`torch`、`matplotlib`；若启用 RobustScaler/winsorize 需 `scikit-learn`、`joblib`。  

具体版本以你机器与 CUDA 环境为准。
