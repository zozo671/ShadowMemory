"""
memory_store.py
数字影子记忆模块（项目核心）。
使用 JSON 文件保存用户历史行为，不使用数据库、不使用机器学习。
"""

import os
import json
import cv2
import numpy as np

import config as cfg


class MemoryStore:
    """行为记忆存储器：负责历史动作的保存、加载与读取。"""

    def __init__(self):
        # 确保数据目录存在
        if not os.path.exists(cfg.MEMORY_DIR):
            os.makedirs(cfg.MEMORY_DIR)

        self.file_path = cfg.MEMORY_FILE
        # 内存中的行为记录列表（原始帧级记录，供近期统计）
        self.records = []
        # 行为画像：按动作类别聚合的长期记忆（频率/时长/习惯）
        self.profile = {}
        # 过去姿态样本：仅保存少量关键点，供“过去动作召回”使用
        self.pose_examples = []
        # 加载已有历史（若存在）
        self.load()

    def load(self):
        """从 JSON 文件加载过去的历史行为数据（兼容旧版纯列表格式）。"""
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # 新格式：{"records": [...], "profile": {...}, "pose_examples": [...]}
                if isinstance(data, dict):
                    self.records = data.get("records", [])
                    self.profile = data.get("profile", {})
                    self.pose_examples = data.get("pose_examples", [])
                else:
                    # 旧格式：纯列表，迁移为 records，profile 留空
                    self.records = data if isinstance(data, list) else []
                    self.profile = {}
                    self.pose_examples = []
            except (json.JSONDecodeError, IOError):
                # 文件损坏或读取失败则重置为空
                self.records = []
                self.profile = {}
                self.pose_examples = []
        else:
            # 文件不存在则创建空文件
            self._write_file()
            self.records = []
            self.profile = {}
            self.pose_examples = []

    def _write_file(self):
        """将内存中的记录与行为画像写入 JSON 文件。"""
        data = {"records": self.records, "profile": self.profile, "pose_examples": self.pose_examples}
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def save_action(self, timestamp, landmarks, duration, action_category="unknown", move_direction="idle"):
        """
        保存一次用户动作数据（仅写入内存，由 flush 定期落盘），
        并同步更新按动作类别聚合的长期行为画像。
        :param timestamp: 动作发生时间（time.time()）
        :param landmarks: 人体关键点列表（来自 PoseTracker，保留兼容参数）
        :param duration: 该动作持续的时间（秒）
        :param action_category: 由 analyzer 推导的动作类别标签
        :param move_direction: 简化的移动方向标签
        """
        record = {
            "time": timestamp,
            "duration": duration,
            "action": action_category,
            "direction": move_direction,
        }
        self.records.append(record)

        # 限制内存记录数量，防止无限增长
        if len(self.records) > cfg.MAX_MEMORY_RECORDS:
            self.records.pop(0)

        # ---- 更新长期行为画像（按动作类别聚合） ----
        stat = self.profile.get(action_category)
        if stat is None:
            stat = {
                "count": 0,
                "total_duration": 0.0,
                "avg_duration": 0.0,
                "first_seen": timestamp,
                "last_seen": timestamp,
                "direction_counts": {},
            }
            self.profile[action_category] = stat
        stat["count"] += 1
        stat["total_duration"] += float(duration)
        stat["avg_duration"] = stat["total_duration"] / stat["count"]
        stat["last_seen"] = timestamp
        direction_counts = stat.setdefault("direction_counts", {})
        direction_counts[move_direction] = direction_counts.get(move_direction, 0) + 1

    def _select_pose_landmarks(self, landmarks):
        """仅保存少量关键点，减小内存占用并保持召回效果。"""
        indices = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
        selected = []
        for idx in indices:
            if idx < len(landmarks):
                lm = landmarks[idx]
                selected.append({
                    "x": float(lm.get("x", 0.0)),
                    "y": float(lm.get("y", 0.0)),
                    "z": float(lm.get("z", 0.0)),
                })
            else:
                selected.append({"x": 0.0, "y": 0.0, "z": 0.0})
        return selected

    def _pose_similarity(self, a, b):
        """轻量相似度判断，避免保存过多重复姿态。"""
        if not a or not b:
            return 1.0
        total = 0.0
        for p1, p2 in zip(a, b):
            total += abs(p1.get("x", 0.0) - p2.get("x", 0.0)) + abs(p1.get("y", 0.0) - p2.get("y", 0.0))
        return total / max(1, len(a))

    def save_pose_example(self, timestamp, landmarks, action_category="unknown", body_mask=None):
        """保存一个简化的历史姿态样本，最多保留少量条目。

        同时可选择保存当时的 `body_mask`（uint8 0/255），将其缩放为
        `cfg.MASK_SAVE_SIZE` 并以二维整数列表形式保存到 JSON 中，供 recall 时
        直接还原为轮廓使用。
        """
        if not landmarks or len(landmarks) <= 16:
            return None

        example = {
            "time": timestamp,
            "action": action_category,
            "landmarks": self._select_pose_landmarks(landmarks),
        }

        # 如果提供了 body_mask，则缩放并保存为小尺寸二维数组（0/255）
        if body_mask is not None:
            try:
                small = cv2.resize(body_mask, (cfg.MASK_SAVE_SIZE, cfg.MASK_SAVE_SIZE), interpolation=cv2.INTER_NEAREST)
                # 确保为 0/255 且为 Python 原生类型以便 JSON 序列化
                small = (small.astype(np.uint8)).tolist()
                example["mask"] = small
            except Exception:
                example["mask"] = None

        for existing in self.pose_examples:
            if self._pose_similarity(example["landmarks"], existing.get("landmarks", [])) < 0.12:
                return existing

        # debug: 输出保存时是否包含下采样的 mask
        has_mask = True if example.get("mask") else False
        print(f"[memory] saved example: has_mask={has_mask}")

        self.pose_examples.append(example)
        if len(self.pose_examples) > 6:
            self.pose_examples.pop(0)
        return example

    def get_pose_examples(self, limit=4):
        """返回最近保存的几个姿态样本。"""
        if not self.pose_examples:
            return []
        return self.pose_examples[-limit:]

    def sample_pose_example(self, limit=4):
        """随机抽取一个历史姿态样本用于召回。"""
        examples = self.get_pose_examples(limit)
        if not examples:
            return None
        import random
        return random.choice(examples)

    def flush(self):
        """将内存中的记录写入 JSON 文件（建议定期或退出时调用）。"""
        self._write_file()

    def get_recent(self, n=10):
        """
        获取最近 n 次动作记录。
        :param n: 返回的记录条数
        :return: 最近 n 条记录的列表（按时间从旧到新）
        """
        return self.records[-n:]

    def get_all(self):
        """返回全部历史记录。"""
        return self.records

    def count(self):
        """返回已保存的动作总次数。"""
        return len(self.records)

    def get_profile(self):
        """返回按动作类别聚合的长期行为画像。"""
        return self.profile

    def get_action_stats(self, action_category):
        """
        返回某动作类别的聚合统计；若从未出现则返回零值。
        :return: dict {count, total_duration, avg_duration, first_seen, last_seen}
        """
        return self.profile.get(action_category, {
            "count": 0,
            "total_duration": 0.0,
            "avg_duration": 0.0,
            "first_seen": 0.0,
            "last_seen": 0.0,
            "direction_counts": {},
        })

    def get_recent_count(self, now, window_seconds=None):
        """返回最近一段时间内的记录数量，避免每帧遍历大量完整历史。"""
        if window_seconds is None:
            window_seconds = cfg.REPEAT_ACTION_WINDOW
        cutoff = now - window_seconds
        count = 0
        for record in self.records:
            if record.get("time", 0.0) >= cutoff:
                count += 1
        return count

    def get_habit_summary(self):
        """从简化画像中返回一个轻量的长期习惯摘要。"""
        if not self.profile:
            return {
                "dominant_action": "unknown",
                "dominant_direction": "idle",
                "habit_strength": 0.0,
            }

        dominant_action = max(
            self.profile.items(),
            key=lambda item: (item[1].get("count", 0), item[1].get("last_seen", 0.0)),
        )[0]

        direction_counts = {}
        for stat in self.profile.values():
            for direction, count in stat.get("direction_counts", {}).items():
                direction_counts[direction] = direction_counts.get(direction, 0) + count

        dominant_direction = "idle"
        if direction_counts:
            dominant_direction = max(direction_counts.items(), key=lambda item: (item[1], item[0]))[0]

        total_count = sum(stat.get("count", 0) for stat in self.profile.values())
        habit_strength = min(1.0, total_count / 10.0)
        return {
            "dominant_action": dominant_action,
            "dominant_direction": dominant_direction,
            "habit_strength": habit_strength,
        }
