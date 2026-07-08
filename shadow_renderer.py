"""
shadow_renderer.py
数字影子视觉渲染模块。
使用 MediaPipe Selfie Segmentation 得到的人体 mask，
在纯色背景上绘制白色（或黑色）人体剪影，
并将历史剪影以透明度递减的残影叠加，形成"数字残像"。
不依赖 TouchDesigner，纯 Python + OpenCV 窗口显示。
"""

import cv2
import numpy as np

import config as cfg


class ShadowRenderer:
    """数字影子渲染器：当前人体剪影 + 历史残影（透明度递减）。"""

    def __init__(self):
        # 残影缓冲：保存历史帧的人体 mask（uint8 0/255）
        self.afterimage_buffer = []
        self.frame_count = 0

    def _mask_to_silhouette(self, mask, fg_color, blur=0):
        """
        将二值 mask 转换为单色剪影图层（与画布同尺寸、3 通道）。
        mask 中 255 区域填充 fg_color，其余为 0。
        """
        # 用 mask 作为权重，生成彩色剪影（仅在人体区域有颜色）
        fg = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        fg[mask == 255] = fg_color
        if blur > 0:
            fg = cv2.GaussianBlur(fg, (blur, blur), 0)
        return fg

    def update(self, frame, pose_data, body_mask, memory_state=None):
        """
        渲染一帧数字影子。
        :param frame: 原始 BGR 帧（仅用于尺寸参考，不再叠加显示）
        :param pose_data: PoseTracker 返回的姿态字典（用于判定是否有人）
        :param body_mask: PoseTracker.get_body_mask 返回的人体二值 mask
        :param memory_state: analyzer 输出的记忆状态字典（可选）
        :return: 渲染后的显示帧（黑/白底 + 剪影 + 残影）
        """
        h, w = frame.shape[:2]
        landmarks = pose_data.get("landmarks", [])
        has_body = len(landmarks) > 0 and body_mask is not None and np.any(body_mask)

        # 背景画布（黑底或白底）
        if cfg.SHADOW_BG_MODE == "white":
            canvas = np.full((h, w, 3), cfg.SHADOW_BG_COLOR, dtype=np.uint8)
            fg_color = (0, 0, 0)          # 白底 → 黑影
            after_color = (60, 60, 80)    # 残影偏暗
        else:
            canvas = np.zeros((h, w, 3), dtype=np.uint8)
            fg_color = cfg.SHADOW_FG_COLOR      # 黑底 → 白影
            after_color = cfg.SHADOW_AFTERIMAGE_COLOR

        # 根据 analyzer 输出决定残影强度倍率
        intensity = 1.0
        if memory_state:
            intensity = memory_state.get("intensity", 1.0)

        # 1. 绘制历史残影（越早越透明）
        n = len(self.afterimage_buffer)
        for idx, hist_mask in enumerate(self.afterimage_buffer):
            # 越靠前的历史（idx 小）透明度越低
            alpha = (idx + 1) / max(n, 1)
            alpha = alpha ** (1.0 / max(intensity, 0.1))  # 强度越高残影越明显
            alpha = float(np.clip(alpha * cfg.AFTERIMAGE_DECAY, 0.05, 1.0))
            sil = self._mask_to_silhouette(hist_mask, after_color, cfg.AFTERIMAGE_BLUR)
            canvas = cv2.addWeighted(canvas, 1.0, sil, alpha, 0)

        # 2. 绘制当前人体剪影（最亮、不透明）
        if has_body:
            cur_sil = self._mask_to_silhouette(body_mask, fg_color, 0)
            canvas = cv2.addWeighted(canvas, 1.0, cur_sil, 1.0, 0)

            # 3. 采样保存历史残影（用当前 mask）
            self.frame_count += 1
            if self.frame_count % cfg.AFTERIMAGE_INTERVAL == 0:
                self.afterimage_buffer.append(body_mask.copy())
                if len(self.afterimage_buffer) > cfg.AFTERIMAGE_COUNT:
                    self.afterimage_buffer.pop(0)

        return canvas