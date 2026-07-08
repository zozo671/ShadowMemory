"""
analyzer.py
历史行为分析模块。
基于 MemoryStore 保存的历史数据，用简单数学算法分析用户行为趋势。
不训练模型，仅做计数、均值、方差等统计。
"""

import numpy as np

import config as cfg


class BehaviorAnalyzer:
    """行为分析器：从记忆中提取重复次数、停留时间、移动频率等趋势。"""

    def __init__(self):
        self.prev_center = None       # 上一帧人体重心
        self.move_history = []        # 近期每帧位移大小
        self.action_start = None      # 当前动作开始时间
        self.last_seen = None         # 上一帧检测到人的时间

    def _center_of_mass(self, landmarks):
        """计算人体重心（所有关键点均值），返回 (x, y)。"""
        if not landmarks:
            return None
        xs = [lm["x"] for lm in landmarks]
        ys = [lm["y"] for lm in landmarks]
        return (sum(xs) / len(xs), sum(ys) / len(ys))

    def analyze_frame(self, pose_data, memory_store):
        """
        分析当前帧并输出记忆状态字典。
        :param pose_data: PoseTracker 返回的姿态字典
        :param memory_store: MemoryStore 实例
        :return: 记忆状态字典，供渲染器使用
        """
        landmarks = pose_data.get("landmarks", [])
        now = pose_data.get("timestamp", 0.0)

        # ---- 计算当前重心与位移 ----
        center = self._center_of_mass(landmarks)
        move_dist = 0.0
        if center and self.prev_center:
            move_dist = np.hypot(
                center[0] - self.prev_center[0],
                center[1] - self.prev_center[1],
            )
        if center:
            self.prev_center = center

        # ---- 移动频率（滑动窗口均值） ----
        self.move_history.append(move_dist)
        if len(self.move_history) > cfg.MOVE_FREQ_WINDOW:
            self.move_history.pop(0)
        move_freq = float(np.mean(self.move_history)) if self.move_history else 0.0

        # ---- 停留判定 ----
        is_staying = move_dist < cfg.STAY_THRESHOLD

        # ---- 动作持续时间 ----
        if landmarks:
            if self.action_start is None:
                self.action_start = now
            duration = now - self.action_start
            self.last_seen = now
        else:
            # 人体消失，结束当前动作
            duration = 0.0
            self.action_start = None

        # ---- 重复动作统计（基于历史记录） ----
        records = memory_store.get_all()
        total_actions = len(records)
        # 统计时间窗口内的动作次数
        recent = [r for r in records
                  if now - r.get("time", 0) <= cfg.REPEAT_ACTION_WINDOW]
        recent_count = len(recent)
        high_freq = recent_count >= cfg.HIGH_FREQ_COUNT

        # ---- 记忆强度（供残影渲染） ----
        # 高频动作 + 长停留 → 强度更高
        intensity = 1.0
        if high_freq:
            intensity += 0.5
        if is_staying:
            intensity += 0.3
        intensity = min(intensity, 3.0)

        return {
            "move_freq": move_freq,
            "is_staying": is_staying,
            "duration": duration,
            "total_actions": total_actions,
            "recent_count": recent_count,
            "high_freq": high_freq,
            "intensity": intensity,
        }