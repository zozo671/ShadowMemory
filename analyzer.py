"""
analyzer.py
历史行为分析模块。
基于 MemoryStore 保存的历史数据，用简单数学算法分析用户行为趋势。
不训练模型，仅做计数、均值、方差等统计，并从 MediaPipe Pose 关键点推导
离散动作类别，建立"长期行为记忆"供数字影子产生反馈。
"""

import numpy as np

import config as cfg

# MediaPipe Pose 关键点索引（与 pose_tracker 保持一致）
_L_SHOULDER, _R_SHOULDER = 11, 12
_L_WRIST, _R_WRIST = 15, 16


class BehaviorAnalyzer:
    """行为分析器：从记忆中提取重复次数、停留时间、移动频率等趋势，
    并推导当前动作类别，输出供渲染器使用的记忆状态。"""

    def __init__(self):
        self.prev_center = None       # 上一帧人体重心
        self.move_history = []        # 近期每帧位移大小
        self.presence_start = None    # 人体出现起始时间
        self.last_seen = None         # 上一帧检测到人的时间
        self.prev_action = None       # 上一帧动作类别
        self.action_start = None      # 当前动作类别起始时间
        self.stay_timer = 0.0         # 连续停留时间
        self.last_frame_time = None   # 上一帧时间戳
        self.last_pose_saved_time = 0.0  # 上一次保存历史姿态的时间
        self.last_recall_time = 0.0   # 上一次触发召回的时间

    def _center_of_mass(self, landmarks):
        """计算人体重心（所有关键点均值），返回 (x, y)。"""
        if not landmarks:
            return None
        xs = [lm["x"] for lm in landmarks]
        ys = [lm["y"] for lm in landmarks]
        return (sum(xs) / len(xs), sum(ys) / len(ys))

    def _classify_action(self, landmarks, move_freq, center):
        """由 Pose 关键点推导一个离散动作类别标签（无需训练模型）。"""
        if not landmarks or len(landmarks) <= _R_WRIST:
            return "unknown"
        ls = landmarks[_L_SHOULDER]
        rs = landmarks[_R_SHOULDER]
        lw = landmarks[_L_WRIST]
        rw = landmarks[_R_WRIST]
        shoulder_y = (ls["y"] + rs["y"]) / 2.0
        wrist_y = (lw["y"] + rw["y"]) / 2.0
        shoulder_span = abs(ls["x"] - rs["x"]) + 1e-6
        wrist_span = abs(lw["x"] - rw["x"])

        if move_freq > cfg.ACTION_MOVE_FREQ:
            return "moving"
        if wrist_y < shoulder_y - cfg.ACTION_ARMS_UP_MARGIN:
            return "arms_up"
        if wrist_span > shoulder_span * cfg.ACTION_SPREAD_RATIO:
            return "arms_spread"
        if center is not None and center[1] > cfg.ACTION_CROUCH_Y:
            return "crouch"
        return "standing"

    def _infer_direction(self, center, prev_center):
        """从重心位移推断简单方向标签。"""
        if center is None or prev_center is None:
            return "idle"

        dx = center[0] - prev_center[0]
        dy = center[1] - prev_center[1]
        if np.hypot(dx, dy) < cfg.STAY_THRESHOLD:
            return "idle"
        if abs(dx) > abs(dy):
            return "right" if dx > 0 else "left"
        return "down" if dy > 0 else "up"

    def analyze_frame(self, pose_data, memory_store, body_mask=None):
        """
        分析当前帧并输出记忆状态字典。
        :param pose_data: PoseTracker 返回的姿态字典
        :param memory_store: MemoryStore 实例（用于读取长期行为画像）
        :return: 记忆状态字典，供渲染器使用
        """
        landmarks = pose_data.get("landmarks", [])
        now = pose_data.get("timestamp", 0.0)

        # ---- 计算当前重心与位移 ----
        center = self._center_of_mass(landmarks)
        move_dist = 0.0
        move_dir = "idle"
        if center and self.prev_center:
            move_dist = np.hypot(
                center[0] - self.prev_center[0],
                center[1] - self.prev_center[1],
            )
            move_dir = self._infer_direction(center, self.prev_center)
        if center:
            self.prev_center = center
        else:
            self.prev_center = None

        # ---- 移动频率（滑动窗口均值） ----
        self.move_history.append(move_dist)
        if len(self.move_history) > cfg.MOVE_FREQ_WINDOW:
            self.move_history.pop(0)
        move_freq = float(np.mean(self.move_history)) if self.move_history else 0.0

        # ---- 停留判定 ----
        is_staying = move_dist < cfg.STAY_THRESHOLD
        dt = 0.0
        if self.last_frame_time is not None:
            dt = now - self.last_frame_time
        self.last_frame_time = now
        if landmarks and is_staying:
            self.stay_timer += dt
        else:
            self.stay_timer = 0.0

        # ---- 人体出现时长 ----
        if landmarks:
            if self.presence_start is None:
                self.presence_start = now
            duration = now - self.presence_start
            self.last_seen = now
        else:
            duration = 0.0
            self.presence_start = None

        # ---- 动作类别 + 当前类别持续时长 ----
        action_category = self._classify_action(landmarks, move_freq, center)
        action_changed = False
        if landmarks:
            if action_category != self.prev_action:
                action_changed = True
                self.action_start = now
            action_duration = (now - self.action_start) if self.action_start else 0.0
            self.prev_action = action_category
        else:
            action_duration = 0.0
            self.prev_action = None
            self.action_start = None

        # ---- 历史姿态样本保存（仅在停留较久时少量保存，避免影响性能） ----
        recall_target = None
        recall_triggered = False
        if landmarks and action_category != "unknown" and is_staying and self.stay_timer >= 1.5:
            if now - self.last_pose_saved_time >= 2.0:
                # 保存历史姿态同时保存当时的 body_mask（若有）以便后续召回显示为真正剪影
                memory_store.save_pose_example(now, landmarks, action_category, body_mask=body_mask)
                self.last_pose_saved_time = now

        # ---- 长期行为画像（来自 MemoryStore） ----
        stats = memory_store.get_action_stats(action_category)
        freq_count = stats["count"]
        avg_duration = stats["avg_duration"]
        is_known = freq_count >= 1                       # 曾经出现过
        is_frequent = freq_count >= cfg.HIGH_FREQ_ACTION_COUNT   # 高频动作
        # 当前动作已保持较久，或该动作历史平均就偏长 → 视为"长时间保持"
        is_long_held = (action_duration >= cfg.LONG_ACTION_DURATION) or \
                       (is_known and avg_duration >= cfg.LONG_ACTION_DURATION)

        # ---- 简化习惯摘要 ----
        habit_summary = memory_store.get_habit_summary()
        dominant_action = habit_summary.get("dominant_action", action_category)
        dominant_direction = habit_summary.get("dominant_direction", "idle")
        habit_strength = habit_summary.get("habit_strength", 0.0)
        if freq_count >= 1:
            habit_strength += 0.2
        if action_category == dominant_action:
            habit_strength += 0.2
        if move_dir != "idle" and dominant_direction == move_dir:
            habit_strength += 0.1
        habit_strength = float(np.clip(habit_strength, 0.0, 1.0))
        memory_echo = is_known and action_changed and action_category == dominant_action

        # ---- 过去动作召回：用户长时间停留时，从历史姿态样本中随机挑一个，触发平滑回忆 ----
        if landmarks and is_staying and self.stay_timer >= 1.0 and now - self.last_recall_time >= 1.5:
            recall_target = memory_store.sample_pose_example()
            if recall_target is not None:
                recall_triggered = True
                self.last_recall_time = now
                print(f"[analyzer] recall triggered: stay_timer={self.stay_timer:.2f} target={recall_target.get('action')}")
                # debug: 输出 recall 目标的键，确认是否包含 mask 字段
                try:
                    keys = list(recall_target.keys())
                except Exception:
                    keys = []
                print(f"[recall] target keys: {keys}")
            else:
                print("[analyzer] recall target missing")
        else:
            print(f"[analyzer] stay={is_staying:.0f} stay_timer={self.stay_timer:.2f} landmarks={bool(landmarks)}")

        # ---- 重复动作统计（基于历史记录，兼容旧逻辑） ----
        total_actions = len(memory_store.get_all())
        recent_count = memory_store.get_recent_count(now)
        high_freq = recent_count >= cfg.HIGH_FREQ_COUNT

        # ---- 记忆强度（供残影渲染） ----
        # 高频动作 + 长停留 + 习惯回响 → 强度更高
        intensity = 1.0
        if is_frequent:
            intensity += 0.6
        if is_long_held:
            intensity += 0.4
        if is_staying:
            intensity += 0.2
        if habit_strength > 0.2:
            intensity += 0.25 * habit_strength
        intensity = min(intensity, 3.0)

        return {
            "move_freq": move_freq,
            "move_dir": move_dir,
            "is_staying": is_staying,
            "duration": duration,
            # —— 行为学习相关信号 ——
            "action_category": action_category,
            "action_changed": action_changed,
            "action_duration": action_duration,
            "is_known": is_known,
            "is_frequent": is_frequent,
            "is_long_held": is_long_held,
            "freq_count": freq_count,
            "avg_duration": avg_duration,
            "dominant_action": dominant_action,
            "dominant_direction": dominant_direction,
            "habit_strength": habit_strength,
            "memory_echo": memory_echo,
            "action_matches_habit": action_category == dominant_action,
            # —— 兼容旧字段 ——
            "total_actions": total_actions,
            "recent_count": recent_count,
            "high_freq": high_freq,
            "intensity": intensity,
            "recall_triggered": recall_triggered,
            "recall_target": recall_target,
            "recall_duration": 2.2,
        }