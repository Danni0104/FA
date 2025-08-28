# FA
AGV-system
# 基于 PPO 的 AGV 调度实验

这个仓库主要是关于自动导引车 (AGV) 的任务调度和充电策略。  
方法上使用了强化学习里的 PPO 算法，并结合了 Transformer 结构。

## 主要文件
- `AGVEnv.py`：自定义的 AGV 环境
- `ppo_agent.py`：PPO 算法实现
- `train.py` / `train_ppo.py`：训练入口
- `main.py`：示例运行
- `config/default.py`：配置文件
- `pmsp.yml`：conda 环境配置

## 使用方法
1. 克隆代码
    ```bash
   git clone https://github.com/<你的用户名>/<仓库名>.git
   cd <仓库名>
   
创建环境（需要 conda）
conda env create -f pmsp.yml
conda activate pmsp

运行训练
python train.py
