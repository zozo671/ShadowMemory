"""
pose_tracker.py
封装 OpenCV 摄像头读取、MediaPipe Pose 人体关键点检测，
以及 MediaPipe Selfie Segmentation 人体轮廓（mask）提取。
不训练模型，仅调用 MediaPipe 预训练接口。
"""

import cv2
import mediapipe as mp
import numpy as np
import time

import config as cfg


class PoseTracker:
    """实时人体跟踪器：打开摄像头，同时提取关键点与人体分割 mask。"""

    def __init__(self):
        # ---- MediaPipe Pose（用于行为记忆 / 分析）----
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            model_complexity=cfg.POSE_MODEL_COMPLEXITY,
            min_detection_confidence=cfg.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=cfg.MIN_TRACKING_CONFIDENCE,
        )

        # ---- MediaPipe Selfie Segmentation（用于人体剪影）----
        self.mp_selfie = mp.solutions.selfie_segmentation
        self.segmenter = self.mp_selfie.SelfieSegmentation(
            model_selection=cfg.SEGMENTATION_MODEL
        )

        # 打开摄像头
        self.cap = cv2.VideoCapture(cfg.CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.WINDOW_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.WINDOW_HEIGHT)

    def read_frame(self):
        """读取一帧原始图像，返回 (ret, frame)。"""
        return self.cap.read()

    def get_pose_data(self, frame):
        """
        对一帧 BGR 图像做姿态检测。
        返回字典：
        {
            "timestamp": 时间,
            "landmarks": [ {x, y, z}, ... ]  # 未检测到人体时为空列表
        }
        """
        # BGR 转 RGB（MediaPipe 需要 RGB 输入）
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)

        landmarks = []
        if results.pose_landmarks:
            for lm in results.pose_landmarks.landmark:
                landmarks.append({
                    "x": lm.x,
                    "y": lm.y,
                    "z": lm.z,
                })

        return {
            "timestamp": time.time(),
            "landmarks": landmarks,
        }

    def get_body_mask(self, frame):
        """
        对一帧 BGR 图像做人像分割，返回二值人体 mask（uint8, 0/255）。
        mask 中 255 表示人体像素，0 表示背景。
        未检测到人体时返回全 0 mask。
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.segmenter.process(rgb)
        if results.segmentation_mask is None:
            h, w = frame.shape[:2]
            return np.zeros((h, w), dtype=np.uint8)

        # segmentation_mask 取值 0~1，阈值化得到二值 mask
        mask = (results.segmentation_mask > cfg.SEGMENTATION_THRESHOLD).astype(np.uint8) * 255
        return mask

    def release(self):
        """释放摄像头与模型资源。"""
        self.cap.release()
        self.pose.close()
        self.segmenter.close()
        cv2.destroyAllWindows()