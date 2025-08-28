# scripts/train.py

# 将项目根目录加入 import 路径，这样 config、model、env_agv 都能被正确导入
from pathlib import Path
import sys
# __file__ 是当前脚本路径，.parent 是上一级目录
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch.nn.functional as F
from typing import List, Deque
import numpy as np
import torch
from config.default import config
from env_agv.AGVEnv import AGVEnv
from model.ppo_agent import PPOAgent
from collections import deque
from pathlib import Path
import matplotlib.pyplot as plt

# ==== 全局调试开关 ====
VERBOSE = False    # 需要详细日志时改成 True

Path("logs").mkdir(parents=True, exist_ok=True)

def _to_list_actions(a, num_agvs):
    """
    把 agent 返回的 a（可能是 scalar / numpy / torch / list / tuple）规范为
    长度为 num_agvs 的 python list[int]，或抛错提示。
    临时策略：若 agent 返回单个标量且 num_agvs>1，则把该标量复制为每个 AGV 的动作。
    """
    # numpy / torch -> list
    if isinstance(a, np.ndarray):
        a_list = a.tolist()
    elif isinstance(a, torch.Tensor):
        a_list = a.cpu().numpy().tolist()
    elif isinstance(a, (list, tuple)):
        a_list = list(a)
    elif isinstance(a, (int, np.integer)):
        a_list = [int(a)]
    else:
        # 尝试转换为 list（若失败则抛 TypeError）
        try:
            a_list = list(a)
        except Exception:
            raise TypeError("Unsupported action type: " + str(type(a)))

    # 如果 agent 返回 scalar（如 14），但 env 期望 vector（num_agvs>1）
    if len(a_list) == 1 and num_agvs > 1:
        # 临时策略：复制标量到每个 AGV（可改成 raise，如果你希望强制 agent 输出向量）
        a_list = [int(a_list[0])] * num_agvs

    # 最后类型/长度检查
    if len(a_list) != num_agvs:
        raise ValueError(f"Converted action length {len(a_list)} != num_agvs {num_agvs}. Raw action: {a}")

    # 保证全为 int
    a_list = [int(x) for x in a_list]
    return a_list


def compute_gae(rewards, values, dones, gamma, lam):
    advantages, gae = [], 0.0
    for idx in reversed(range(len(rewards))):
        # δ_t = r_t + γ V(s_{t+1}) (1−done) − V(s_t)
        delta = rewards[idx] + gamma * values[idx+1] * (1 - dones[idx]) - values[idx]
        gae   = delta + gamma * lam * (1 - dones[idx]) * gae
        advantages.insert(0, gae)
    return advantages

# === 统计与可视化容器（只初始化一次） ===
ep_rewards: List[float] = []
ma_rewards: List[float] = []
ma_window = int(config['ppo'].get('moving_avg_window', 50))
reward_history: Deque[float] = deque(maxlen=ma_window)
done_all_hist: List[int] = []
on_time_ratio_hist: List[float] = []




def train():
    # 1) 创建环境 & 读取维度
    env      = AGVEnv(**config['env'])
    # env.reset() 返回 (tokens, mask)
    # tokens.shape = [T, feat_dim]
    # action_space 是 MultiDiscrete
    # action_space.nvec[0] 就是第一个动作空间的大小
    obs, _       = env.reset()
    dummy_tokens = obs['tokens']
    feat_dim     = dummy_tokens.shape[1]
    act_dim = int(np.asarray(getattr(env.action_space, "nvec"))[0])

    # 2) 初始化 PPO Agent（兼容多种可能的构造签名）

    # 准备常用参数/候选字典
    T = dummy_tokens.shape[0]  # token 行数 (例如 25)
    feat_dim = feat_dim  # 每行特征数 (例如 5)
    num_agvs = getattr(env, "num_agvs", None) or getattr(env, "num_agents", None)

    cand_kwargs = []

    # 常见旧签名：input_dim, action_dim, hidden_dim, n_heads
    cand_kwargs.append({
        "input_dim": feat_dim,
        "action_dim": act_dim,
        "hidden_dim": config['transformer']['embed_dim'],
        "n_heads": config['transformer']['num_heads'],
    })

    # 新签名示例：feat_dim, T, action_dim, num_agvs, hidden_dim, lr_*
    cand_kwargs.append({
        "feat_dim": feat_dim,
        "T": T,
        "action_dim": act_dim,
        "num_agvs": num_agvs,
        "hidden_dim": config.get('transformer', {}).get('embed_dim', 128),
        "lr_actor": config.get('ppo', {}).get('actor_lr', 3e-4),
        "lr_critic": config.get('ppo', {}).get('critic_lr', 1e-3),
    })

    # 另一种常见变体：input_dim + num_agvs
    cand_kwargs.append({
        "input_dim": feat_dim,
        "action_dim": act_dim,
        "num_agvs": num_agvs,
        "hidden_dim": config.get('transformer', {}).get('embed_dim', 128),
        "n_heads": config.get('transformer', {}).get('num_heads', 8),
    })

    # 尝试构造，直到成功为止
    last_exc = None
    agent = None
    for i, kw in enumerate(cand_kwargs, start=1):
        # 过滤掉值为 None 的 key（避免传 num_agvs=None 等）
        kw_clean = {k: v for k, v in kw.items() if v is not None}
        try:
            agent = PPOAgent(**kw_clean)
            print(
                f"DEBUG: PPOAgent initialized with candidate #{i}, kwargs keys: {list(kw_clean.keys())}")
            break
        except TypeError as e:
            print(f"DEBUG: candidate #{i} failed: {e}")
            last_exc = e

    if agent is None:
        # 没一个候选成功 -> 抛出最后一次 TypeError 以便你能看到具体错误并贴出来
        raise last_exc

    # 确认 agent 有关键属性，便于后续运行
    print("DEBUG: PPOAgent type:", type(agent), "module:",
          getattr(agent, "__module__", "<unknown>"))
    print("DEBUG: agent has attributes:",
          [n for n in ["num_agvs", "device", "optimizer_actor", "optimizer_critic"] if
           hasattr(agent, n)])

    # 3) 主训练循环
    for ep in range(config['ppo']['n_episodes']):
        # ====== 2) episode 开始 ======
        obs, _  = env.reset()
        tokens  = obs['tokens']
        mask    = obs['mask']

        obs_buf, act_buf, logp_buf = [], [], []
        rew_buf, val_buf, done_buf = [], [], []
        per_agent_rew_buf = []
        action_mask_buf = []  # 逐步存掩码
        agent_active_buf = []

        done = False
        # 每集最多跑 n_steps 步
        a = None  # 仅为日志打印兜底

        while not done and len(obs_buf) < config['ppo']['n_steps']:

            # === 3.1) 构造当前步动作掩码 ===
            num_jobs = env.num_jobs
            action_dim = num_jobs + 1  # 最后一列 = 充电
            job_free = obs["job_mask"].astype(bool)  # [num_jobs] 1=可选, 0=不可选
            am_step = np.zeros((env.num_agvs, action_dim), dtype=bool)

            # --- 读取每个 job 的 slack_n（来自 _get_obs 第3列）并判断是否有急单 ---
            job_tokens = obs["tokens"][env.num_agvs: env.num_agvs + num_jobs]
            slack = job_tokens[:, 3]  # 第3列是我们在 _get_obs 里放的 slack_n
            urgent_exists = bool((slack < 0.0).any())

            for i, agv in enumerate(env.agvs):
                if agv["status"] == 0:  # idle 才能接新任务
                    am_step[i, :num_jobs] = job_free
                    # 有急单且电量>60%时先不让它去充电；否则按原规则
                    if urgent_exists and agv["battery"] > 0.6:
                        am_step[i, num_jobs] = False
                    else:
                        am_step[i, num_jobs] = (agv["battery"] < 0.999)

                else:
                    # 非 idle：本步不会接新任务；留个“充电”保底，避免整行全 False
                    am_step[i, num_jobs] = True

            # 防御：若某行全 False，强制把“充电”打开
            rows_none = ~am_step.any(axis=1)
            am_step[rows_none, num_jobs] = True

            # === 新增：本步“活跃代理”掩码（只对 idle 的车回传梯度）===
            agent_active = np.array([agv["status"] == 0 for agv in env.agvs], dtype=bool)

            # 选动作并缓存两个掩码（用于 evaluate）
            a, logp, _ = agent.choose_action(tokens, mask, action_mask=am_step)

            action_mask_buf.append(am_step)  # [num_agvs, action_dim]  (bool)
            agent_active_buf.append(agent_active)  # [num_agvs]              (bool)

            # 把 logp 规范化存为 cpu 上的 0-d tensor（便于 torch.stack）
            if isinstance(logp, torch.Tensor):
                logp_cpu = logp.detach().cpu()
            else:
                logp_cpu = torch.tensor(float(logp), dtype=torch.float32)

            v = agent.get_value(tokens, mask)
            val_buf.append(float(v))

            # debug：打印 choose_action 返回的类型和 logp 的当前形状（安全）
            print("DEBUG choose_action -> type(a):", type(a), "len act_buf sample:", getattr(a, '__len__', lambda: None)())
            print("DEBUG logp (stored) shape:", getattr(logp_cpu, "shape", None), "value:", (logp_cpu.item() if logp_cpu.numel()==1 else str(logp_cpu.shape)))

            # 3.2) 执行动作（先把 a 转成 env 能接受的形式）
            a_to_env = _to_list_actions(a, env.num_agvs)

            # 执行动作并命名 info（不要用 _ 丢弃）
            obs2, r, term, trunc, info = env.step(a_to_env)

            # 先取得 next 状态（以便最后更新 tokens/mask）
            next_tokens = obs2['tokens']
            next_mask   = obs2['mask']
            obs = obs2  # 下一步需要新的 job_mask



            # --- 规范化 r 和从 info 中提取 per-agent reward（优先 r 自身，其次 info） ---
            if isinstance(r, np.ndarray):
                r_print = r.tolist()
            else:
                r_print = r

            # --- 从 info 里找 per-agent 奖励（可能不存在）---
            per_agent_from_info = None
            if isinstance(info, dict):
                for key in (
                'per_agent_reward', 'rewards', 'reward_per_agent', 'agent_rewards', 'rewards_list'):
                    if key in info:
                        per_agent_from_info = info[key]
                        break

            # --- 生成 per_agent_rewards（list[float]），按优先级三选一 ---
            if isinstance(r_print, (list, tuple, np.ndarray)):
                # 情况1：env.step 直接返回了每个 agent 的列表/数组
                per_agent_rewards = [float(x) for x in (
                    list(r_print) if not isinstance(r_print, np.ndarray) else r_print.tolist())]

            elif per_agent_from_info is not None:
                # 情况2：从 info 里拿到每个 agent 的奖励
                try:
                    if isinstance(per_agent_from_info, np.ndarray):
                        per_agent_rewards = per_agent_from_info.astype(np.float64).tolist()
                    elif isinstance(per_agent_from_info, (list, tuple)):
                        per_agent_rewards = [float(x) for x in per_agent_from_info]
                    elif isinstance(per_agent_from_info, dict):
                        per_agent_rewards = [float(x) for x in per_agent_from_info.values()]
                    else:
                        per_agent_rewards = [float(per_agent_from_info)] * env.num_agvs
                except (TypeError, ValueError):
                    per_agent_rewards = [0.0] * env.num_agvs

            else:
                # 情况3：把 r 当成总奖励（标量），复制给每个 agent
                try:
                    scalar = float(r_print)
                except (TypeError, ValueError):
                    scalar = 0.0
                    print("WARN: cannot parse scalar reward r; using 0.0. r:", r_print, " info:",
                          info)
                per_agent_rewards = [scalar] * env.num_agvs

            # --- 统一转成 ndarray，便于后面 sum/mean ---
            try:
                per_agent_arr = np.array(per_agent_rewards, dtype=np.float64)
            except Exception:
                per_agent_arr = np.zeros(env.num_agvs, dtype=np.float64)

            # --- 计算 step_total（训练用的标量奖励）---
            # 若 r_print 本来就是标量且 info 也没给 per-agent，就把它视为 total；
            # 否则按 per_agent_arr 求和（你也可改成 mean）
            if (not isinstance(r_print, (list, tuple, np.ndarray))) and (
                    per_agent_from_info is None):
                try:
                    step_total = float(r_print)
                except Exception:
                    step_total = float(np.sum(per_agent_arr))
            else:
                step_total = float(np.sum(per_agent_arr))

            # 打印调试信息（降低频率）
            step_idx = len(obs_buf)
            if step_idx % 50 == 0 or step_idx == 0 or done:
                print(f"STEP {step_idx:03d} actions -> {a_to_env}")
                print(f"  step_total used for training: {step_total}")

            # 3.3) 存 buffer（训练用 rew_buf 存 scalar total；per_agent_rew_buf 存向量用于分析）
            obs_buf.append(tokens)        # current tokens（未更新）
            act_buf.append(a_to_env)  # 存入已经转换为 [num_agvs] 的 int 列表  原始 agent 输出（list/tuple）
            logp_buf.append(logp_cpu)     # 你之前标准化的 logp_cpu（确保存在）

            rew_buf.append(step_total)                # 训练用标量 reward（float）
            per_agent_rew_buf.append(per_agent_arr)   # 存 numpy array（或 list），仅用于分析/可视化

            done_buf.append(term or trunc)

            # 3.4) 更新状态
            tokens, mask = next_tokens, next_mask
            done = term or trunc


        # 3.5) 收尾：再推一步 value
        v_last = agent.get_value(tokens, mask)
        if isinstance(v_last, torch.Tensor):
            v_last = float(v_last.detach().cpu())
        else:
            v_last = float(v_last)
        val_buf.append(v_last)

        # 4) 计算 GAE 和 returns
        advs    = compute_gae(rew_buf, val_buf, done_buf,
                              config['ppo']['gamma'],
                              config['ppo']['gae_lambda'])
        returns = [a + v for a, v in zip(advs, val_buf[:-1])]

        # 5) 转成张量（先合并成 numpy，再转 torch）
        # 保证 obs_buf 中每个元素都是 numpy array（若为 torch.Tensor 则转 numpy）
        states_np = np.stack(
            [t if isinstance(t, np.ndarray) else t.detach().cpu().numpy() for t in obs_buf], axis=0)

        states    = torch.from_numpy(states_np).float().to(agent.device)
        # actions here likely is [N, num_agvs] if you stored per-step list of actions
        actions_np = np.array(act_buf, dtype=np.int64)  # [N] or [N, num_agvs]
        actions    = torch.from_numpy(actions_np).to(agent.device)
        oldlogp    = torch.stack(logp_buf).to(agent.device)  # may be [N] or [N,1] or [N]
        # normalize oldlogp shape: prefer 1-D [N]
        if oldlogp.dim() > 1:
            oldlogp = oldlogp.squeeze(-1)
        # returns / advs
        rets_np    = np.array(returns, dtype=np.float32)
        rets       = torch.from_numpy(rets_np).to(agent.device)
        advs_np    = np.array(advs, dtype=np.float32)
        advs_t     = torch.from_numpy(advs_np).to(agent.device)

        # detach everything (safe)
        states  = states.detach()
        actions = actions.detach()
        oldlogp = oldlogp.detach()
        rets    = rets.detach()
        advs_t  = advs_t.detach()

        # === 准备动作掩码 ===
        if len(action_mask_buf) > 0:
            # [N, num_agvs, action_dim]  -> bool
            am_np = np.stack(action_mask_buf, axis=0).astype(np.bool_)
            am_t = torch.from_numpy(am_np).to(agent.device)
            # 防御：每个代理至少有一个 True
            if not torch.all(am_t.any(dim=-1)):
                raise ValueError("action_mask has a row with no valid action.")
        else:
            am_t = None

        # （可选）活跃代理掩码：只允许 idle 的 AGV 更新策略/动作
        if len(agent_active_buf) > 0:
            # [N, num_agvs] -> bool
            active_np = np.stack(agent_active_buf, axis=0).astype(np.bool_)
            active_t = torch.from_numpy(active_np).to(agent.device)
        else:
            active_t = None

        # 6) PPO 更新
        for _ in range(config['ppo']['n_epochs']):
            # 1) 重新计算新策略下的 log-prob、entropy、value
            try:
                logp_new, entropy, value = agent.evaluate(states, actions, action_mask=am_t)
                # 期望形状: [N], [N], [N] 或带第二维 [N,num_agvs]
            except Exception:
                import traceback as _tb
                print("ERROR: agent.evaluate raised an exception. Traceback follows:", flush=True)
                _tb.print_exc()
                print("  states.shape:", getattr(states, "shape", None), "dtype:", getattr(states, "dtype", None), flush=True)
                print("  actions.shape:", getattr(actions, "shape", None), "dtype:", getattr(actions, "dtype", None), flush=True)
                print("  oldlogp.shape:", getattr(oldlogp, "shape", None), "dtype:", getattr(oldlogp, "dtype", None), flush=True)
                raise

            # --- 准备基本变量（注意先定义 N 和 device） ---
            N = states.shape[0]
            device = agent.device

            # 转为 tensor 并放到 device（evaluate 可能返回 numpy/torch 在 cpu 上等）
            logp_new = torch.as_tensor(logp_new, device=device)
            entropy  = torch.as_tensor(entropy, device=device)
            value    = torch.as_tensor(value, device=device)

            # 把 oldlogp 也移动到同一 device 并确保为 tensor
            oldlogp = torch.as_tensor(oldlogp, device=device)

            # debug: 打印原始 shapes
            print("DEBUG raw evaluate shapes: logp_new", getattr(logp_new, "shape", None),
                  "entropy", getattr(entropy, "shape", None),
                  "value", getattr(value, "shape", None),
                  "oldlogp", getattr(oldlogp, "shape", None), flush=True)
            # —— 统一成 1-D [N] ——
            # 如果 evaluate 返回的是按代理的 [N, num_agvs]，把它们合并
            if logp_new.dim() == 2:  # [N, num_agvs]
                # 若你有 active_t（[N, num_agvs]，True=活跃），可以只统计活跃代理
                if 'active_t' in locals() and active_t is not None:
                    act_f = active_t.to(logp_new.device).float()
                    logp_new = (logp_new * act_f).sum(dim=1)  # 合并为 [N]
                    entropy = (entropy * act_f).sum(dim=1) / act_f.sum(dim=1).clamp_min(1)  # 有效平均
                else:
                    logp_new = logp_new.sum(dim=1)  # 对代理求和 -> [N]
                    entropy = entropy.mean(dim=1)  # 对代理求均值 -> [N]

            # value 通常应是 [N] 或 [N,1]；防御性 squeeze
            if value.dim() > 1:
                value = value.squeeze(-1)

            # oldlogp 期望是 [N]；如果是 [N,1] 就 squeeze
            if oldlogp.dim() > 1:
                oldlogp = oldlogp.squeeze(-1)

            # 保证 dtype 一致
            logp_new = logp_new.to(torch.float32)
            entropy = entropy.to(torch.float32)
            value = value.to(torch.float32)
            oldlogp = oldlogp.to(torch.float32)

            # 如果 evaluate 对每个 AGV 返回了单独的 logprob/entropy（shape == [N, num_agvs]），
            # 我们把它们合并为单个标量（sum），以与 choose_action 返回的“单一 logp 标量”保持一致。
            def reduce_if_per_agent(t, name):
                if t.dim() == 2:
                    # 如果第二维等于 agent.num_agvs，则按 AGV 维度合并为单个标量
                    if t.size(1) == getattr(agent, "num_agvs", None):
                        return t.sum(dim=1)
                    # 如果第二维是 1，直接 squeeze
                    if t.size(1) == 1:
                        return t.squeeze(1)
                    # 其它情况暂时报错，便于你检查
                    raise ValueError(f"{name} has unexpected 2-D shape {t.shape}; expected [N,num_agvs] or [N,1].")
                return t

            logp_new = reduce_if_per_agent(logp_new, "logp_new")
            entropy  = reduce_if_per_agent(entropy,  "entropy")

            # 确保都是 1-d [N]
            def ensure_1d(t, name):
                if t.dim() == 0:
                    if N == 1:
                        return t.unsqueeze(0)
                    raise ValueError(f"{name} is 0-d but batch N={N} > 1.")
                if t.dim() == 2 and t.size(1) == 1:
                    return t.squeeze(1)
                if t.dim() != 1:
                    raise ValueError(f"{name} has unexpected shape {t.shape}; expected [N].")
                return t

            logp_new = ensure_1d(logp_new, "logp_new")
            entropy  = ensure_1d(entropy,  "entropy")
            value    = ensure_1d(value,    "value")

            # oldlogp 也应该是 [N]
            if oldlogp.shape != logp_new.shape:
                raise ValueError(f"Shape mismatch after normalization: oldlogp.shape={oldlogp.shape} vs logp_new.shape={logp_new.shape}. "
                                 "If your agent returns per-AGV logprobs, ensure both choose_action and evaluate use the same aggregation (sum/mean).")

            # 2) 计算 ratio
            ratio = torch.exp(logp_new - oldlogp)  # [N]

            # 3) Advantage 归一化
            advs_norm = (advs_t - advs_t.mean()) / (advs_t.std() + 1e-8)  # [N]

            # 4) 计算 Actor 损失（带裁剪）
            clip_eps = config['ppo']['clip_range']
            unclipped = ratio * advs_norm
            clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advs_norm
            loss_pi = -torch.min(unclipped, clipped).mean()

            # 5) Critic 损失
            loss_v = config['ppo']['vf_coef'] * F.mse_loss(value, rets)

            # 6) 熵惩罚
            loss_ent = -config['ppo']['ent_coef'] * entropy.mean()

            # 7) 总损失
            loss = loss_pi + loss_v + loss_ent

            print(
                f"[PPO] loss {loss.item():.6f} pi {loss_pi.item():.6f} v {loss_v.item():.6f} ent {loss_ent.item():.6f}"
            )

            # 8) 反向 & 优化
            agent.optimizer_actor.zero_grad()
            agent.optimizer_critic.zero_grad()
            loss.backward()
            agent.optimizer_actor.step()
            agent.optimizer_critic.step()


        # ====== 7) 日志 & 保存（在每个 episode 末） ======
        total_reward = float(sum(rew_buf))

        # ——累计与移动均值（更新外层容器）——
        ep_rewards.append(total_reward)
        reward_history.append(total_reward)
        ma = float(np.mean(reward_history))  # 用 deque 直接算最近 ma_window 的均值
        ma_rewards.append(ma)

        print(f"len(ep_rewards)={len(ep_rewards)} "
              f"len(ma_rewards)={len(ma_rewards)} "
              f"head={ep_rewards[:3]} tail={ep_rewards[-3:]}")

        # === 额外统计：全完成率&按时率 ===
        summary = env.summarize_assignments()
        done_all = int(env.completed_jobs == env.num_jobs)

        on_time = sum(1 for r in summary if (r["completion_time"] is not None and r["on_time"]))
        finished = sum(1 for r in summary if r["completion_time"] is not None)
        on_time_ratio = (on_time / finished) if finished > 0 else 0.0

        done_all_hist.append(done_all)
        on_time_ratio_hist.append(on_time_ratio)
        # === 额外统计：全完成率&按时率 ===
        summary = env.summarize_assignments()
        done_all = int(env.completed_jobs == env.num_jobs)

        on_time = sum(1 for r in summary if (r["completion_time"] is not None and r["on_time"]))
        finished = sum(1 for r in summary if r["completion_time"] is not None)
        on_time_ratio = (on_time / finished) if finished > 0 else 0.0

        done_all_hist.append(done_all)
        on_time_ratio_hist.append(on_time_ratio)

        # === 在这里粘贴“每集20个任务得分”代码块 ===
        import csv, os

        # （可选）偶尔打印看看
        if (ep % 50 == 0) or (ep == config['ppo']['n_episodes'] - 1):
            print("\n[Episode {:d}] Job-by-job scores:".format(ep))
            print("job  dur  due  done_at  on_time  assigned_to  job_reward")
            for r in summary:
                print(f"{r['job_id']:>3}  {r['duration']:>3.1f}  {r['delivery_due']:>3d}  "
                      f"{str(r['completion_time']):>7}  {int(bool(r['on_time'])):>7}  "
                      f"{str(r['assigned_to']):>11}  {r['job_reward']:>9.1f}")
            print()

        # 持久化到 CSV：logs/jobs_per_episode.csv
        csv_path = "logs/jobs_per_episode.csv"
        file_exists = os.path.exists(csv_path)

        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "episode", "job_id", "assigned_to", "assigned",
                "duration", "delivery_due", "completion_time", "on_time", "job_reward"
            ])
            if not file_exists:
                writer.writeheader()

            for r in summary:
                writer.writerow({
                    "episode": ep,
                    "job_id": r["job_id"],
                    "assigned_to": r["assigned_to"],
                    "assigned": r["assigned"],
                    "duration": r["duration"],
                    "delivery_due": r["delivery_due"],
                    "completion_time": r["completion_time"],
                    "on_time": int(bool(r["on_time"])),
                    "job_reward": r["job_reward"],
                })

        # ——写 CSV（方便画图/复盘）——
        Path("logs").mkdir(parents=True, exist_ok=True)
        np.savetxt(
            "logs/episode_rewards.csv",
            np.c_[
                np.arange(len(ep_rewards)), np.array(ep_rewards, dtype=float), np.array(ma_rewards,
                                                                                        dtype=float)],
            delimiter=",",
            header="episode,reward,ma",
            comments=""
        )

        # ——日志（每 10 回合打印一次）——
        if ep % 10 == 0:
            print(f"Episode {ep:4d}  Reward {total_reward:.2f}  |  moving_avg({ma_window})={ma:.3f}")
            print("  step rewards:", rew_buf)
            last_step_reward = rew_buf[-1] if len(rew_buf) > 0 else None
            print(f"  last step reward: {last_step_reward}")
            if a is not None:
                try:
                    import numpy as _np, torch as _th
                    if isinstance(a, _th.Tensor):
                        a_repr = a.cpu().numpy().tolist()
                    elif isinstance(a, _np.ndarray):
                        a_repr = a.tolist()
                    else:
                        a_repr = a
                    print("  DEBUG action (final):", a_repr)
                except Exception as e:
                    print("  DEBUG action conversion error:", e)

        # ——保存模型——
        if ep % 100 == 0:
            Path("logs").mkdir(parents=True, exist_ok=True)
            agent.save("checkpoints/ppo_agv.pt")

        if (ep + 1) % 250 == 0:  # 每 250 个 episode 出一张
            # === 训练结束后画一张奖励曲线图 ===
            fig = plt.figure()
            x = np.arange(len(ep_rewards))

            if len(ep_rewards) >= 2:
                plt.plot(x, ep_rewards, label="episode reward")
                plt.plot(x, ma_rewards, label=f"moving avg ({ma_window})")
            else:
                plt.plot([0], [ep_rewards[0]], marker="o", label="episode reward")
                plt.plot([0], [ma_rewards[0]], marker="o", label=f"moving avg ({ma_window})")

            plt.xlabel("episode")
            plt.ylabel("reward")
            plt.title("PPO Training Rewards")
            plt.legend()

            # 安全的 y 轴范围
            ys = list(ep_rewards) + list(ma_rewards)
            if ys:
                ymin, ymax = float(min(ys)), float(max(ys))
                if ymin == ymax:
                    ymin -= 1.0  # ← 注意这里没有逗号
                    ymax += 1.0  # ← 注意这里没有逗号
                plt.ylim(ymin, ymax)

            plt.tight_layout()
            plt.show()
            plt.close(fig)


        # --- Debug / Ensure train() runs when module executed ---
if __name__ == "__main__":
    print(">>> scripts.train __main__ start", flush=True)
    print("  Python executable:", sys.executable, flush=True)
    print("  File:", __file__, flush=True)
    print("  CWD:", Path.cwd(), flush=True)
    try:
        from config.default import config
        print("  config['ppo'] n_episodes:", config['ppo'].get('n_episodes'), flush=True)
    except Exception as e:
        print("  Could not read config:", repr(e), flush=True)

    try:
        train()
    except Exception as e:
        import traceback
        print("  train() raised exception:", flush=True)
        traceback.print_exc()

    print(">>> scripts.train __main__ end", flush=True)
