"""
memory_store.py
数字影子记忆模块（项目核心）。
使用 JSON 文件保存用户历史行为，不使用数据库、不使用机器学习。
"""

import os
import json

import config as cfg


class MemoryStore:
    """行为记忆存储器：负责历史动作的保存、加载与读取。"""

    def __init__(self):
        # 确保数据目录存在
        if not os.path.exists(cfg.MEMORY_DIR):
            os.makedirs(cfg.MEMORY_DIR)

        self.file_path = cfg.MEMORY_FILE
        # 内存中的行为记录列表
        self.records = []
        # 加载已有历史（若存在）
        self.load()

    def load(self):
        """从 JSON 文件加载过去的历史行为数据。"""
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    self.records = json.load(f)
            except (json.JSONDecodeError, IOError):
                # 文件损坏或读取失败则重置为空
                self.records = []
        else:
            # 文件不存在则创建空文件
            self._write_file()
            self.records = []

    def _write_file(self):
        """将内存中的记录写入 JSON 文件。"""
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(self.records, f, ensure_ascii=False, indent=2)

    def save_action(self, timestamp, landmarks, duration):
        """
        保存一次用户动作数据（仅写入内存，由 flush 定期落盘）。
        :param timestamp: 动作发生时间（time.time()）
        :param landmarks: 人体关键点列表（来自 PoseTracker）
        :param duration: 该动作持续的时间（秒）
        """
        record = {
            "time": timestamp,
            "landmarks": landmarks,
            "duration": duration,
        }
        self.records.append(record)

        # 限制内存记录数量，防止无限增长
        if len(self.records) > cfg.MAX_MEMORY_RECORDS:
            self.records.pop(0)

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