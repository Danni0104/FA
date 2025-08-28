# Validation/scripts/main.py

from env_agv.AGVEnv import AGVEnv
from model.ppo_agent import PPOAgent
from config.default     import config

def main():
    # 1. 拆 config
    env_cfg        = config["env"]
    ppo_cfg        = config["ppo"]
    transformer_cfg = config["transformer"]

    # 2. 创建环境
    env = AGVEnv(
        num_agvs=env_cfg["num_agvs"],
        num_jobs=env_cfg["num_jobs"],
        charger_capacity=env_cfg["charger_capacity"],
        max_steps=env_cfg["max_steps"]
    )

    # 3. 构造 PPOAgent
    input_dim  = transformer_cfg["embed_dim"]
    action_dim = env_cfg["num_jobs"] + 1
    hidden_dim = transformer_cfg["embed_dim"]
    n_heads    = transformer_cfg["num_heads"]

    agent = PPOAgent(input_dim, action_dim, hidden_dim, n_heads)

    # 4. 演示跑几集
    n_episodes = config.get("n_episodes", 2)
    for epi in range(n_episodes):
        obs = env.reset()
        done = False
        total_r = 0
        while not done:
            a, logp, ent = agent.choose_action(obs)
            obs, r, done, _ = env.step(a)
            total_r += r
        print(f"Episode {epi:3d}: Reward = {total_r:.2f}")

if __name__ == "__main__":
    main()
