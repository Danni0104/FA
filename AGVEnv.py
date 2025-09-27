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
            reward_scale: float = 1.0,
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
        self.jobs = self._generate_jobs(r=self.due_r, R=self.due_R, M=4.0)

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

    # ... 在你的 AGVEnv 类中 ...

    def step(self, actions):
        self.time += 1
        reward = 0.0

        # ---------- 规范化 actions -> list[int] ----------
        import numpy as _np
        if isinstance(actions, (int, _np.integer)):
            actions = [int(actions)]
        else:
            if hasattr(actions, "tolist"):
                actions = actions.tolist()
            else:
                actions = list(actions)

        if len(actions) != self.num_agvs:
            raise ValueError(f"actions 长度应为 {self.num_agvs}, 但收到 {len(actions)}: {actions}")

        # ========== 1) 仅对 idle 车处理“新动作请求” ==========
        for idx, act in enumerate(actions):
            agv: AGV = self.agvs[idx]
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
                    agv["status"] = 2
                else:
                    if idx not in self.charging_queue:
                        self.charging_queue.append(idx)
                    agv["status"] = 3
            # ---- B) 请求具体任务（0..num_jobs-1）----
            elif 0 <= action < self.num_jobs:
                job: Job = self.jobs[action]
                if job["assigned"] == 0:
                    battery_needed = job["duration"] / 10.0
                    if agv["battery"] < battery_needed:
                        reward -= 5.0  # 电量耗尽惩罚
                        continue
                    else:
                        agv["status"] = 1
                        agv["task_timer"] = job["duration"]
                        agv["current_job_id"] = action
                        agv["battery"] = agv["battery"] - battery_needed
                        job["assigned"] = 1
                        job["assigned_to"] = idx
                else:
                    pass

        # ========== 1.5) 排队 -> 充电桩分配 ==========
        while self._count_charging() < self.charger_capacity and self.charging_queue:
            next_idx = self.charging_queue.pop(0)
            self.agvs[next_idx]["status"] = 2

        # ========== 2) 动力学推进与惩罚 ==========
        for agv in self.agvs:
            if agv["status"] == 1:  # 运输
                agv["task_timer"] -= 1
            elif agv["status"] == 2:  # 充电
                agv["battery"] = min(1.0, agv["battery"] + self.charge_rate)
                reward -= 0.25  # 充电惩罚
                if agv["battery"] >= 1.0:
                    agv["status"] = 0
            elif agv["status"] == 3:  # 排队
                reward -= 0.5  # 排队惩罚

            # 统一夹紧电量到 [0,1]，这一步是多余的，因为电池电量永远不会低于 0
            if agv["battery"] > 1.0:
                agv["battery"] = 1.0

        # ========== 3) 任务完成判定 ==========
        for agv in self.agvs:
            if agv["status"] == 1 and agv["task_timer"] <= 0:
                agv["status"] = 0
                self.completed_jobs += 1
                job_idx = agv.get("current_job_id")
                if job_idx is not None:
                    job = self.jobs[job_idx]
                    job["completion_time"] = self.time
                    agv["current_job_id"] = None

                    # 逾期惩罚
                    if self.time > job["delivery_due"]:
                        overdue_steps = self.time - job["delivery_due"]
                        penalty = overdue_steps * 2.0
                        reward -= penalty
                        job["reward_accum"] = job.get("reward_accum", 0.0) - penalty
                    # 准时完成没有奖励，但不会有惩罚
                    else:
                        job["reward_accum"] = job.get("reward_accum", 0.0)

        # ========== 4) 终止条件 ==========
        terminated = (self.completed_jobs == self.num_jobs)
        truncated = (self.time >= self.max_steps)

        # 5) info & 观测
        raw_reward = reward
        # 注意：最终奖励可能为负，这是正常的
        if terminated:
            reward += 40.0  # 移除这个奖励，让智能体专注于逾期惩罚

        reward = reward * self.reward_scale

        info = {
            "raw_reward": raw_reward,
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



