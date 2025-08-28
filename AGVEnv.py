import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from typing import Optional
from typing import TypedDict
from typing import List


class Job(TypedDict):
    duration: float
    assigned: int
    assigned_to: Optional[int]  # 新增：被哪台AGV接了
    delivery_due: int
    completion_time: Optional[int]
    reward_accum: float  # 新增：该任务累计得分

class AGV(TypedDict):
    status: int
    task_timer: float
    battery: float
    current_job_id: Optional[int]

class AGVEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    # ------------------------------ 初始化 ------------------------------
    def __init__(
            self,
            num_agvs: int = 5,
            num_jobs: int = 20,
            charger_capacity: int = 2,
            battery_threshold: float = 0.0,
            max_steps: int = 400,
            charge_rate: float = 0.2,
            due_r: float = 0.10,
            due_R: float = 0.50,
            reward_scale: float = 0.1,
            **kwargs,  #  接收额外参数
    ):
        super().__init__()

        self.num_agvs = num_agvs
        self.num_jobs = num_jobs
        self.charger_capacity = charger_capacity
        self.battery_threshold = battery_threshold
        self.max_steps = max_steps
        self.charge_rate = charge_rate
        self.due_r = due_r
        self.due_R = due_R
        self.MAX_JOBS = num_jobs
        self.reward_scale = reward_scale if reward_scale is not None else kwargs.get("reward_scale",
                                                                                     1.0)

        # ----------------- 观测 / 动作空间 -----------------
        # ---------- 1. 观测空间 ----------
        self.observation_space = spaces.Dict({
            "tokens": spaces.Box(
                low=0.0,
                high=1.0,
                shape=(self.num_agvs + self.MAX_JOBS, 5),  # 行数 × 5 列不变
                dtype=np.float32
            ),
            "mask": spaces.MultiBinary(self.num_agvs + self.MAX_JOBS),
            "job_mask": spaces.MultiBinary(self.num_jobs),
        })
        # 每台车的动作数量 = num_jobs (任务编号 0..num_jobs-1) + 1 (充电选项)
        # 确保 self.num_jobs 和 self.num_agvs 在此之前已被赋值
        n_options = int(self.num_jobs) + 1

        # 将动作空间设为 MultiDiscrete：
        # num_agvs=5, num_jobs=20 -> MultiDiscrete([21,21,21,21,21])
        self.action_space = spaces.MultiDiscrete([n_options] * int(self.num_agvs))

        # 占位变量
        self.time             = 0
        self.agvs             = []
        self.jobs             = []
        self.completed_jobs   = 0
        self.charging_queue   = []

        # 完整重置
        self.reset()

    # ----------------  r-R 公式生成 Job 列表 ----------------
    def _generate_jobs(self, r: float, R: float, M: float = 1.0):
        jobs = []
        for _ in range(self.num_jobs):
            duration = random.uniform(1, 5)                          # p_j
            lower    = M * duration * (1 - r - R / 2)
            upper    = M * duration * (1 - r + R / 2)
            due = max(1, int(random.uniform(lower, upper)))

            jobs.append({
                "duration"       : duration,
                "assigned"       : 0,
                "assigned_to": None,  # 哪台AGV接了这个任务
                "delivery_due"   : due,
                "completion_time": None,
                "reward_accum": 0.0  # 该任务的专属得分(仅计“分配/完成”部分)
            })
        return jobs

    # ------------------------------ reset ------------------------------

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):

        super().reset(seed=seed)   # 先给 Gym、Python random、NumPy RNG 都设定同一个种子
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.time           = 0
        self.completed_jobs = 0
        self.charging_queue: List[int] = []

        # AGV 初始化
        self.agvs = [{
            "battery"    : 1.0,
            "status"     : 0,      # 0 idle | 1 transporting | 2 charging
            "task_timer" : 0,
            "current_job_id": None
        } for _ in range(self.num_agvs)]

        # Job 初始化（r-R 公式）
        self.jobs = self._generate_jobs(r=self.due_r, R=self.due_R, M=6.0)

        # 返回观测 & 空 info
        return self._get_obs(), {}

    # ----------------------------- 状态向量 -----------------------------
    def _get_obs(self):
        tokens = []
        # 1) AGV 行
        for agv in self.agvs:
            onehot = [0, 0, 0, 0]
            onehot[agv["status"]] = 1
            tokens.append([agv["battery"], *onehot])

        # 2) Job 行
        for job in self.jobs:
            # —— 新增：紧迫度与逾期标记 ——
            # slack = 剩余时间 = ddl - 当前时刻 - 该任务时长
            slack = job["delivery_due"] - self.time - job["duration"]
            # 截断再归一化到 [-1, 1] 左右
            slack = max(-self.max_steps, min(self.max_steps, slack))
            slack_n = slack / float(self.max_steps)

            overdue = 1.0 if self.time > job["delivery_due"] else 0.0

            tokens.append([
                (job["duration"] - 1) / 4.0,  # 列0：时长
                int(job["assigned"]),  # 列1：是否已分配
                job["delivery_due"] / float(self.max_steps),  # 列2：绝对 ddl
                slack_n,  # 列3：**紧迫度**
                overdue  # 列4：**是否已逾期**
            ])

        # 3) Padding
        for _ in range(self.MAX_JOBS - len(self.jobs)):
            tokens.append([0.0] * 5)

        obs  = np.asarray(tokens, dtype=np.float32)
        mask = np.array(
            [1] * self.num_agvs +
            [1] * len(self.jobs) +
            [0] * (self.MAX_JOBS - len(self.jobs)),
            dtype=np.int8
        )

        # job_mask: length num_jobs, 1 means job is available (not yet assigned)
        job_mask = np.array([0 if self.jobs[i]["assigned"] else 1 for i in range(self.num_jobs)], dtype=np.int8)

        return {"tokens": obs, "mask": mask, "job_mask": job_mask}

    # -------------------------------------------------------------------


    # 辅助：当前正在充电的 AGV 数
    def _count_charging(self):
        return sum(1 for agv in self.agvs if agv["status"] == 2)

    # ----------------------------- 单步逻辑 -----------------------------
    def step(self, actions):
        """
        actions: 长度 = num_agvs 的列表/数组，对应每台 AGV 的动作
        """
        self.time += 1
        reward = 0.0

        # ---------- 规范化 actions -> list[int] ----------
        import numpy as _np
        if isinstance(actions, (int, _np.integer)):  # 单个整数（仅当 num_agvs==1 时允许）
            actions = [int(actions)]
        else:
            if hasattr(actions, "tolist"):  # tensor / ndarray
                actions = actions.tolist()
            else:
                actions = list(actions)

        if len(actions) != self.num_agvs:
            raise ValueError(f"actions 长度应为 {self.num_agvs}, 但收到 {len(actions)}: {actions}")

        # ========== 1) 仅对 idle 车处理“新动作请求” ==========
        for idx, act in enumerate(actions):
            agv: AGV = self.agvs[idx]

            # 非 idle（运输/充电/排队）的 AGV 本步忽略新动作
            if agv["status"] != 0:
                continue

            try:
                action = int(act)
            except Exception as e:
                raise TypeError(
                    f"action must be int-convertible, got {type(act)} at index {idx}") from e

            if action < 0 or action > self.num_jobs:
                raise ValueError(
                    f"action out of bounds for agent {idx}: {action} (allow 0..{self.num_jobs})")

            # ---- A) 请求充电（动作 == num_jobs）----
            if action == self.num_jobs:
                if self._count_charging() < self.charger_capacity:
                    agv["status"] = 2  # 直接进入充电
                else:
                    if idx not in self.charging_queue:
                        self.charging_queue.append(idx)
                    agv["status"] = 3  # 排队
                    reward -= 0.05  # 请求充电的固定成本
                continue  # <<<< 关键：避免落入“接任务”分支

            # ---- B) 请求具体任务（0..num_jobs-1）----
            elif 0 <= action < self.num_jobs:  # <<<< 用 elif 且显式范围判断
                job: Job = self.jobs[action]
                if job["assigned"] == 0:
                    agv["status"] = 1
                    agv["task_timer"] = job["duration"]
                    agv["current_job_id"] = action

                    # 扣电并夹紧到 [0,1]
                    agv["battery"] = max(0.0, agv["battery"] - job["duration"] / 10.0)

                    job["assigned"] = 1
                    job["assigned_to"] = idx

                    # 分配奖励（全局 + 任务专属）
                    reward += 5.0
                    job["reward_accum"] = job.get("reward_accum", 0.0) + 5.0
                else:
                    reward -= 0.1  # 任务已被占用

            else:
                # 理论到不了这里（前面有范围检查），加个保险
                continue

        # ========== 1.5) 排队 -> 充电桩分配 ==========
        while self._count_charging() < self.charger_capacity and self.charging_queue:
            next_idx = self.charging_queue.pop(0)
            self.agvs[next_idx]["status"] = 2

        # ========== 2) 动力学推进（统一） ==========
        for agv in self.agvs:
            if agv["status"] == 1:  # 运输
                agv["task_timer"] -= 1
            elif agv["status"] == 2:  # 充电
                agv["battery"] = min(1.0, agv["battery"] + self.charge_rate)
                if agv["battery"] >= 1.0:
                    agv["status"] = 0  # 充满转 idle
            elif agv["status"] == 3:  # 排队
                reward -= 0.01  # 等候惩罚

            # 统一夹紧电量到 [0,1]
            if agv["battery"] > 1.0:
                agv["battery"] = 1.0
            elif agv["battery"] < 0.0:
                agv["battery"] = 0.0


        # ========== 3) 任务完成判定（关键：对所有运输车判断） ==========
        for agv in self.agvs:
            if agv["status"] == 1 and agv["task_timer"] <= 0:
                agv["status"] = 0
                self.completed_jobs += 1
                job_idx = agv.get("current_job_id")
                if job_idx is not None:
                    job = self.jobs[job_idx]
                    job["completion_time"] = self.time
                    agv["current_job_id"] = None

                    if self.time <= job["delivery_due"]:  # 准时/提前
                        reward += 10.0
                        job["reward_accum"] = job.get("reward_accum", 0.0) + 10.0
                    else:  # 逾期
                        reward -= 1.0
                        job["reward_accum"] = job.get("reward_accum", 0.0) - 1.0

                    # --- OLD LOGIC（保留以便回退）---
                    # if self.time <= job["delivery_due"]:
                    #     reward += 10.0
                    # else:
                    #     reward -= 0.25

        # ========== 4) 小额步进惩罚 ==========
        # 正在执行但未完成的任务（鼓励尽快完成）
        for job in self.jobs:
            if job["assigned"] == 1 and job["completion_time"] is None:
                reward -= 0.001

        # 空闲惩罚（鼓励分配任务）
        for agv in self.agvs:
            if agv["status"] == 0:
                reward -= 0.001

        # ========== 5) 终止/截断 ==========
        # OLD LOGIC: 更保守的逾期重罚（仅供回退参考）
        # any_late = any(
        #     job["assigned"] == 1 and job["completion_time"] is None and self.time > job["delivery_due"]
        #     for job in self.jobs
        # )
        # if any_late:
        #     reward -= 10.0

        if self.completed_jobs == self.num_jobs:
            reward += 100.0  # 提前完成奖励

        terminated = (self.completed_jobs == self.num_jobs)  # 任务全部完成
        truncated = (self.time >= self.max_steps)  # 步数超限

        # ========== 6) info & 观测 ==========
        raw_reward = reward                  # 保留原始奖励（未缩放）
        reward = reward * self.reward_scale  # 统一缩放，训练用

        info = {
            "raw_reward": raw_reward,        # 原始奖励，方便调试/分析
            "reward_scale": self.reward_scale
        }

        obs = self._get_obs()
        return obs, reward, terminated, truncated, info


    # ----------------- 新增方法 -----------------
    def summarize_assignments(self):
        rows = []
        for j_idx, job in enumerate(self.jobs):
            ra = float(job.get("reward_accum", 0.0))
            rows.append({
                "job_id": j_idx,
                "assigned_to": job.get("assigned_to"),
                "assigned": int(job.get("assigned", 0)),
                "delivery_due": job.get("delivery_due"),
                "completion_time": job.get("completion_time"),
                "on_time": (job.get("completion_time") is not None and job["completion_time"] <= job["delivery_due"]),
                "job_reward": round(ra, 3),
                "duration": round(float(job.get("duration", 0.0)), 3),
            })
        return rows


    # ----------------------------- 渲染（调试用） -----------------------------
    def render(self, mode="human"):
        print(f"\n--- Time {self.time} ---")
        for idx, agv in enumerate(self.agvs):
            print(f"AGV {idx} | Battery {agv['battery']:.2f} | Status {agv['status']}")
        print(f"Completed jobs: {self.completed_jobs}/{self.num_jobs}")
        print(f"Charging queue: {self.charging_queue}")

if __name__ == "__main__":
    env = AGVEnv(max_steps=100, num_jobs=20, num_agvs=5)  # 请用你的构造参数
    print("action_space:", env.action_space)              # 应显示 MultiDiscrete([...])
    obs, _ = env.reset()
    print("obs keys:", obs.keys())
    print("tokens shape:", obs['tokens'].shape)
    print("mask shape:", obs['mask'].shape)
    print("mask shape:", obs['mask'].shape)

