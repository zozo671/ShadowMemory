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
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # 分割掩码时间稳定状态（用于抑制闪烁、短暂丢失时兜底）
        self._prev_seg = None          # 上一帧平滑后的分割概率图
        self._last_good_mask = None    # 最近一帧有效的人体 mask（兜底用）
        # 推理性能优化状态
        self._frame_idx = 0            # 帧计数（用于 Pose 间隔运行）
        self._pose_cache = None        # 缓存的关键点（间隔帧复用）
        self._seg_frame_idx = 0        # 帧计数（用于分割间隔运行）
        self._seg_cache = None        # 缓存的分割结果（间隔帧复用）
        self._last_pose_valid = False  # 上一帧是否成功得到有效关键点

    def read_frame(self):
        """读取一帧原始图像，返回 (ret, frame)。"""
        return self.cap.read()

    def get_pose_and_mask(self, frame):
        """
        对一帧 BGR 图像做姿态检测，并返回 (pose_data, body_mask)。
        数字影子由 MediaPipe Selfie Segmentation 生成真实人体剪影形状，
        并对分割概率做时间平滑（EMA）+ 短暂丢失兜底，以抑制快速动作时
        的闪烁与手臂缺失，使影子稳定、连续（类似真实影子的轻微延迟）。

        性能优化：
          - 在降分辨率图上运行 MediaPipe（INFER_SCALE），mask 再放大回原尺寸；
            关键点归一化，降分辨率不影响行为分析精度。
          - Pose 每 POSE_INTERVAL 帧才运行一次，其余帧复用缓存，降低开销。
        """
        h, w = frame.shape[:2]

        # 降分辨率推理：MediaPipe 在较小图上运行，mask 再放大回原尺寸
        scale = cfg.INFER_SCALE
        if scale < 1.0:
            small = cv2.resize(frame, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_LINEAR)
        else:
            small = frame
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        sh, sw = small.shape[:2]

        # 人体分割：先做分割，判断当前帧是否真的有可用人体，避免在空帧上继续跑 Pose
        self._seg_frame_idx += 1
        if self._seg_frame_idx % cfg.SEGMENT_INTERVAL == 0 or self._seg_cache is None:
            seg_results = self.segmenter.process(rgb)
            if seg_results.segmentation_mask is None:
                seg = np.zeros((sh, sw), dtype=np.float32)
            else:
                seg = seg_results.segmentation_mask.astype(np.float32)
            self._seg_cache = seg
        else:
            seg = self._seg_cache if self._seg_cache is not None else np.zeros((sh, sw), dtype=np.float32)

        # 时间平滑：对分割概率做 EMA，抑制逐帧闪烁/突变
        a = cfg.SEG_TEMPORAL_ALPHA
        if self._prev_seg is None:
            self._prev_seg = seg
        else:
            self._prev_seg = self._prev_seg * (1.0 - a) + seg * a

        # 阈值化得到稳定二值 mask（小图）
        body_mask_s = (self._prev_seg > cfg.SEGMENTATION_THRESHOLD).astype(np.uint8) * 255

        # 兜底：本帧分割几乎为空（人体短暂丢失）时，沿用上一帧有效 mask，
        # 避免影子突然消失；否则更新"最近有效 mask"
        min_area = int(cfg.SEG_MIN_AREA * sh * sw)
        if cv2.countNonZero(body_mask_s) < min_area and self._last_good_mask is not None:
            body_mask_s = self._last_good_mask
        else:
            self._last_good_mask = body_mask_s.copy()

        # 放大回原尺寸（最近邻保持二值）
        if scale < 1.0:
            body_mask = cv2.resize(body_mask_s, (w, h),
                                   interpolation=cv2.INTER_NEAREST)
        else:
            body_mask = body_mask_s

        body_visible = cv2.countNonZero(body_mask_s) >= max(1, int(cfg.SEG_MIN_AREA * sh * sw))

        # 姿态检测：仅当出现有效人体且满足降频条件时运行 Pose，其余帧复用上一帧关键点并做轻量平滑
        self._frame_idx += 1
        if body_visible and (self._pose_cache is None or self._frame_idx % cfg.POSE_INTERVAL == 0):
            pose_results = self.pose.process(rgb)
            raw_landmarks = []
            if pose_results.pose_landmarks:
                for lm in pose_results.pose_landmarks.landmark:
                    raw_landmarks.append({
                        "x": lm.x,
                        "y": lm.y,
                        "z": lm.z,
                    })
                self._last_pose_valid = True
            else:
                raw_landmarks = []
                self._last_pose_valid = False

            if raw_landmarks:
                if self._pose_cache is None:
                    self._pose_cache = [dict(lm) for lm in raw_landmarks]
                else:
                    smoothed_landmarks = []
                    alpha = cfg.POSE_SMOOTHING_ALPHA
                    for i, lm in enumerate(raw_landmarks):
                        prev_lm = self._pose_cache[i] if i < len(self._pose_cache) else None
                        if prev_lm is None:
                            smoothed_landmarks.append(dict(lm))
                        else:
                            smoothed_landmarks.append({
                                "x": prev_lm["x"] * (1.0 - alpha) + lm["x"] * alpha,
                                "y": prev_lm["y"] * (1.0 - alpha) + lm["y"] * alpha,
                                "z": prev_lm["z"] * (1.0 - alpha) + lm["z"] * alpha,
                            })
                    self._pose_cache = smoothed_landmarks
            else:
                self._pose_cache = self._pose_cache if self._pose_cache is not None else []
        else:
            raw_landmarks = self._pose_cache if self._pose_cache is not None else []

        pose_data = {
            "timestamp": time.time(),
            "landmarks": raw_landmarks,
        }
        return pose_data, body_mask

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
        对一帧 BGR 图像返回二值人体 mask（uint8, 0/255）。
        mask 中 255 表示人体像素，0 表示背景。
        由 MediaPipe Selfie Segmentation 生成真实人体剪影形状。
        未检测到人体时返回全 0 mask。
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        results = self.segmenter.process(rgb)
        if results.segmentation_mask is None:
            return np.zeros((h, w), dtype=np.uint8)
        return (results.segmentation_mask > cfg.SEGMENTATION_THRESHOLD).astype(np.uint8) * 255

    def release(self):
        """释放摄像头与模型资源。"""
        self.cap.release()
        self.pose.close()
        self.segmenter.close()
        cv2.destroyAllWindows()