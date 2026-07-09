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
import time
import random

import config as cfg


class ShadowRenderer:
    """数字影子渲染器：当前人体剪影 + 历史残影（透明度递减）。"""

    def __init__(self):
        # 残影列表：每个元素为 {"mask": 预模糊归一化 float(0~1), "alpha": 独立透明度}
        # 性能优化：模糊只在采样时做一次；每个残影拥有独立 alpha，随时间连续衰减
        self.afterimages = []
        self.frame_count = 0
        self.recall_state = None
        self.recall_active = False
        self.recall_timer = 0.0
        self.show_pose_debug = False
        # 记录上次 spawn 的历史 target time，避免短时间内重复生成相同的历史 ghost
        self._last_spawned_target_time = 0.0
        self._last_spawned_time = 0.0
        self._spawn_cooldown = 4.0

    def _is_valid_landmarks(self, landmarks):
        """检查历史/实时 Pose 是否完整且坐标合法，避免错误姿态覆盖当前影子。"""
        if not landmarks or len(landmarks) < 8:
            return False
        valid_points = 0
        for lm in landmarks:
            if not isinstance(lm, dict):
                continue
            x = lm.get("x")
            y = lm.get("y")
            if x is None or y is None:
                continue
            try:
                x = float(x)
                y = float(y)
            except (TypeError, ValueError):
                continue
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            if abs(x) < 1e-6 and abs(y) < 1e-6:
                continue
            # 兼容旧版/跨尺度姿态坐标：只要是有限且不是全零的点，就视为有效。
            if abs(x) > 10.0 or abs(y) > 10.0:
                continue
            valid_points += 1
        return valid_points >= 8

    def _landmarks_to_points(self, landmarks, w, h):
        """把姿态坐标映射到画布像素坐标，兼容 0..1 与旧版跨尺度坐标。"""
        if not landmarks:
            return []

        values = []
        for lm in landmarks:
            if not isinstance(lm, dict):
                continue
            x = lm.get("x")
            y = lm.get("y")
            if x is None or y is None:
                continue
            try:
                x = float(x)
                y = float(y)
            except (TypeError, ValueError):
                continue
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            if abs(x) < 1e-6 and abs(y) < 1e-6:
                continue
            values.append((x, y))

        if not values:
            return []

        xs = [v[0] for v in values]
        ys = [v[1] for v in values]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        def _normalize(v, v_min, v_max):
            if v_max - v_min < 1e-6:
                return 0.5
            return (v - v_min) / (v_max - v_min)

        points = []
        for x, y in values:
            if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
                nx, ny = x, y
            else:
                nx = _normalize(x, x_min, x_max)
                ny = _normalize(y, y_min, y_max)
            px = int(np.clip(nx, 0.0, 1.0) * (w - 1))
            py = int(np.clip(ny, 0.0, 1.0) * (h - 1))
            points.append((px, py))
        return points

    def _draw_pose_recall(self, canvas, landmarks, alpha, fg_color, offset=(0, 0)):
        """用少量骨架线条绘制一个轻量的“过去姿态”幽灵，保持低开销。"""
        if not landmarks:
            return canvas

        h, w = canvas.shape[:2]
        overlay = np.zeros((h, w, 3), dtype=np.float32)
        if isinstance(fg_color, np.ndarray):
            fg_color = tuple(np.round(fg_color).astype(np.uint8).tolist())
        else:
            fg_color = tuple(int(v) for v in fg_color)
        debug_mode = bool(self.show_pose_debug)
        line_width = 2 if debug_mode else 1
        circle_radius = 3 if debug_mode else 1
        color = (255, 255, 255) if debug_mode else (220, 220, 220)
        points = self._landmarks_to_points(landmarks, w, h)
        # 将历史姿态做像素级偏移以制造“残像”错位感
        offset_x, offset_y = offset if offset is not None else (0, 0)
        shifted = []
        for (px, py) in points:
            sx = int(np.clip(px + int(offset_x), 0, w - 1))
            sy = int(np.clip(py + int(offset_y), 0, h - 1))
            shifted.append((sx, sy))
        points = shifted

        if len(points) >= 13:
            pairs = [
                (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6),
                (1, 7), (2, 8), (7, 9), (8, 10), (9, 11), (10, 12)
            ]
            for a, b in pairs:
                if a < len(points) and b < len(points):
                    cv2.line(overlay, points[a], points[b], color, line_width, cv2.LINE_AA)
            for pt in points:
                cv2.circle(overlay, pt, circle_radius, color, -1, cv2.LINE_AA)
        # 提升残像基底亮度以增强可见性
        overlay = overlay * (0.35 if not debug_mode else 0.7)
        return canvas * (1.0 - alpha) + overlay * alpha

    def _draw_history_ghost(self, canvas, landmarks, alpha, fg_color, offset=(0, 0)):
        """将历史关键点转换为半透明人体掩码并叠加到画布上。
        使用凸包填充关键点并高斯模糊边缘，产生与 realtime `body_mask` 风格相似的幽灵影子。
        """
        if not landmarks:
            return canvas

        h, w = canvas.shape[:2]
        points = self._landmarks_to_points(landmarks, w, h)
        if not points:
            return canvas

        # 应用偏移
        ox, oy = offset if offset is not None else (0, 0)
        pts = []
        for (px, py) in points:
            sx = int(np.clip(px + int(ox), 0, w - 1))
            sy = int(np.clip(py + int(oy), 0, h - 1))
            pts.append([sx, sy])

        if len(pts) < 3:
            return canvas

        pts_arr = np.array(pts, dtype=np.int32)
        try:
            hull = cv2.convexHull(pts_arr)
        except Exception:
            hull = pts_arr

        # 生成二值 mask 并做羽化
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, hull, 255)
        # 使用配置的羽化核大小
        k = getattr(cfg, 'SHADOW_EDGE_FEATHER', 21)
        if k % 2 == 0:
            k = k + 1
        soft = cv2.GaussianBlur(mask, (k, k), 0).astype(np.float32) / 255.0

        # 生成彩色叠加层（采用 fg_color）并按 alpha 混合
        fg = np.array(fg_color, dtype=np.float32)
        overlay = soft[:, :, np.newaxis] * fg
        return canvas * (1.0 - alpha) + overlay * alpha

    def _spawn_history_ghosts(self, target, after_color, w, h, layers=4):
        """基于 recall target 一次性生成若干历史 ghost 层（低分辨率合成）。
        每层作为一个独立的 afterimage 元素加入 self.afterimages，具有不同 start alpha。
        """
        now = time.time()
        # 如果已有 recall 来源的残影且仍在生命周期内，则避免重复生成
        recall_active_recent = False
        recall_linger = 4.0
        for ai in self.afterimages:
            if ai.get("source") == "recall":
                if now - ai.get("birth", now) < recall_linger:
                    recall_active_recent = True
                    break
        if recall_active_recent:
            print("[renderer] skip spawning history ghosts: recent recall ghosts present")
            return
        scale = getattr(cfg, 'SHADOW_RENDER_SCALE', 1.0)
        if scale <= 0.0:
            scale = 1.0
        sw = max(8, int(round(w * scale)))
        sh = max(8, int(round(h * scale)))

        # 获取小尺寸二值 mask：优先使用 target 中保存的 mask，否则由 landmarks 生成凸包 mask并缩小
        target_mask_small = target.get("mask") if isinstance(target, dict) else None
        if target_mask_small:
            small = np.array(target_mask_small, dtype=np.uint8)
            small = (small > 0).astype(np.uint8) * 255
            # 如果存储的是已经是小尺寸，尝试直接使用；若尺寸不同则 resize
            try:
                if small.shape[0] != sh or small.shape[1] != sw:
                    small = cv2.resize(small, (sw, sh), interpolation=cv2.INTER_NEAREST)
            except Exception:
                small = cv2.resize(small, (sw, sh), interpolation=cv2.INTER_NEAREST)
        else:
            # 从关键点构建 mask（低分辨率）
            landmarks = target.get("landmarks", []) if isinstance(target, dict) else []
            small = np.zeros((sh, sw), dtype=np.uint8)
            pts = []
            points = self._landmarks_to_points(landmarks, sw, sh)
            for (px, py) in points:
                pts.append([px, py])
            if len(pts) >= 3:
                try:
                    hull = cv2.convexHull(np.array(pts, dtype=np.int32))
                    cv2.fillConvexPoly(small, hull, 255)
                except Exception:
                    pass

        # 生成多层历史残影，每层 alpha 递减并带小偏移
        base_alpha = 0.55
        for i in range(layers):
            frac = float(i) / max(1, layers - 1) if layers > 1 else 0.0
            start_alpha = base_alpha * (1.0 - 0.6 * frac)
            # 偏移以像素为单位（低分辨率下的偏移更小）
            max_off = max(2, int(round(12 * scale)))
            ox = random.randint(-max_off, max_off)
            oy = random.randint(-max_off, max_off)
            # 应用偏移
            if ox != 0 or oy != 0:
                shifted = np.roll(small, shift=(oy, ox), axis=(0, 1))
                if oy > 0:
                    shifted[:oy, :] = 0
                elif oy < 0:
                    shifted[oy:, :] = 0
                if ox > 0:
                    shifted[:, :ox] = 0
                elif ox < 0:
                    shifted[:, ox:] = 0
            else:
                shifted = small

            # 羽化核随 scale 缩放
            k = max(1, int(round(cfg.SHADOW_EDGE_FEATHER * max(scale, 0.3))))
            if k % 2 == 0:
                k += 1
            soft = cv2.GaussianBlur(shifted, (k, k), 0).astype(np.float32) / 255.0
            cmask_small = soft[:, :, np.newaxis] * np.array(after_color, dtype=np.float32)
            # 将每层作为 afterimage 添加，birth 时间可微调以制造拖尾效果
            self.afterimages.append({
                "cmask_small": cmask_small,
                "birth": now + i * 0.02,
                "start": float(np.clip(start_alpha, 0.02, 0.8)),
                "source": "recall",
            })
            # 控制最大残影数量
            if len(self.afterimages) > cfg.AFTERIMAGE_COUNT * 3:
                self.afterimages.pop(0)

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
        landmarks = pose_data.get("landmarks", []) or []
        current_pose_valid = self._is_valid_landmarks(landmarks)
        has_body = body_mask is not None and np.any(body_mask)

        # 背景画布（黑底或白底）
        if cfg.SHADOW_BG_MODE == "white":
            canvas = np.full((h, w, 3), cfg.SHADOW_BG_COLOR, dtype=np.float32)
            fg_color = (0, 0, 0)          # 白底 → 黑影
            after_color = (60, 60, 80)    # 残影偏暗
        else:
            canvas = np.zeros((h, w, 3), dtype=np.float32)
            fg_color = cfg.SHADOW_FG_COLOR      # 黑底 → 白影
            after_color = cfg.SHADOW_AFTERIMAGE_COLOR

        # 根据 analyzer 输出决定残影强度倍率与行为学习习惯
        intensity = 1.0
        is_frequent = False
        is_long_held = False
        is_known = False
        action_changed = False
        habit_strength = 0.0
        memory_echo = False
        if memory_state:
            intensity = memory_state.get("intensity", 1.0)
            is_frequent = memory_state.get("is_frequent", False)
            is_long_held = memory_state.get("is_long_held", False)
            is_known = memory_state.get("is_known", False)
            action_changed = memory_state.get("action_changed", False)
            habit_strength = memory_state.get("habit_strength", 0.0)
            memory_echo = memory_state.get("memory_echo", False)

        # 1. 基于时间的连续衰减：每个残影按"存活时长"指数衰减，
        #    与帧率无关，保证 5~10 秒内完全消失。
        #    强度越高 → 时间常数越大（残像更持久）；
        #    长时间保持的动作 → 停留更久（tau 进一步放大）。
        now = time.time()
        tau_mult = cfg.LONG_HOLD_TAU_MULT if is_long_held else 1.0
        tau = cfg.AFTERIMAGE_TAU * float(np.clip(intensity, 0.7, 1.5)) * tau_mult

        # 2. 所有残影一次性累加到单个图层（在低分辨率上合成以提升性能）
        #    使用 cfg.SHADOW_RENDER_SCALE 将高昂的高斯模糊与色彩乘法降到小尺寸上，
        #    合成后再上采样回原始分辨率叠加。
        scale = getattr(cfg, 'SHADOW_RENDER_SCALE', 1.0)
        if scale <= 0.0:
            scale = 1.0
        if scale < 1.0:
            sw = max(8, int(round(w * scale)))
            sh = max(8, int(round(h * scale)))
        else:
            sw, sh = w, h

        if self.afterimages:
            acc_small = np.zeros((sh, sw, 3), dtype=np.float32)
            for ai in self.afterimages:
                age = now - ai["birth"]
                a = ai["start"] * np.exp(-age / tau)
                if a <= 0.01:
                    continue
                a *= 0.7
                # 支持旧条目与新条目兼容：优先使用小尺寸缓存 cmask_small
                if ai.get("cmask_small") is not None:
                    acc_small += ai["cmask_small"] * a
                else:
                    # 如果只有全尺寸 cmask，则先缩小再累加（不常见）
                    cm = ai.get("cmask")
                    if cm is not None:
                        small = cv2.resize(cm, (sw, sh), interpolation=cv2.INTER_AREA)
                        acc_small += small * a
            # 上采样回全分辨率并叠加
            if scale < 1.0:
                acc = cv2.resize(acc_small, (w, h), interpolation=cv2.INTER_LINEAR)
            else:
                acc = acc_small
            canvas += acc * 0.9

        # 3. 仅移除已衰减到几乎不可见的残影（平滑消失，无突兀边界）
        self.afterimages = [
            ai for ai in self.afterimages
            if ai["start"] * np.exp(-(now - ai["birth"]) / tau) > 0.01
        ]

        # 4. 绘制当前人体剪影（最亮、边缘羽化柔和）
        if has_body:
            # 边缘羽化：对二值 mask 做高斯模糊得到 0~1 软掩码，
            # 以其为权重做 alpha 混合，使剪影边缘柔和渐变而非生硬
            soft = cv2.GaussianBlur(
                body_mask, (cfg.SHADOW_EDGE_FEATHER, cfg.SHADOW_EDGE_FEATHER), 0
            ).astype(np.float32) / 255.0
            soft3 = soft[:, :, np.newaxis]
            fg = np.array(fg_color, dtype=np.float32)
            canvas = canvas * (1.0 - soft3) + fg * soft3

            # 5. 采样保存新残影（带独立起始 alpha，最新影子最亮）
            self.frame_count += 1

            # 高频动作 / 习惯回响 → 影子更容易出现：轻微缩短采样间隔、提高起始透明度
            interval = max(1, cfg.AFTERIMAGE_INTERVAL // 2) if is_frequent else cfg.AFTERIMAGE_INTERVAL
            habit_boost = cfg.HABIT_BOOST if is_frequent else 1.0
            if habit_strength > 0.5:
                habit_boost = max(habit_boost, 1.0 + 0.2 * habit_strength)
                interval = max(1, int(round(interval * (1.0 - 0.15 * habit_strength))))

            def _spawn(fixed_alpha=None):
                # 在低分辨率上预模糊并归一化为 float（0~1），后续渲染直接复用，减少每帧运算
                if scale < 1.0:
                    small_mask = cv2.resize(body_mask, (sw, sh), interpolation=cv2.INTER_NEAREST)
                else:
                    small_mask = body_mask

                # 根据缩放比例自适应模糊核，保证视觉一致性
                blur_k = max(1, int(round(cfg.AFTERIMAGE_BLUR * max(scale, 0.3))))
                if blur_k % 2 == 0:
                    blur_k += 1
                blurred_small = cv2.GaussianBlur(small_mask, (blur_k, blur_k), 0).astype(np.float32) / 255.0

                if fixed_alpha is not None:
                    start_alpha = float(fixed_alpha)
                else:
                    start_alpha = float(np.clip(
                        cfg.AFTERIMAGE_START_ALPHA * habit_boost
                        * (0.6 + 0.4 * min(intensity, 2.0) / 2.0),
                        0.05, 0.7
                    ))

                cmask_small = blurred_small[:, :, np.newaxis] * np.array(after_color, dtype=np.float32)
                self.afterimages.append({
                    "cmask_small": cmask_small,
                    "birth": time.time(),
                    "start": start_alpha,
                })
                # 安全上限，防止极端情况下无限增长
                if len(self.afterimages) > cfg.AFTERIMAGE_COUNT * 3:
                    self.afterimages.pop(0)

            # 常规采样
            if self.frame_count % interval == 0:
                _spawn()

            # 已知动作重现 → 提前出现的残影（预回声）：
            # 当切换到一个"曾经出现过"的动作时，立即生成一帧淡残影，
            # 仿佛数字影子"预判"了用户的习惯动作。
            if action_changed and (is_known or memory_echo):
                _spawn(fixed_alpha=cfg.ANTICIPATION_ALPHA * (0.8 + 0.4 * habit_strength))

        # 6. 过去动作召回：仅作为当前实时影子的“附加幽灵”，不能替代当前影子
        recall_duration = 4.0
        if self.recall_active and now >= self.recall_timer:
            self.recall_active = False
            self.recall_state = None

        if not has_body:
            self.recall_active = False
            self.recall_state = None
            print("[renderer] recall skipped: no current body mask")
        elif memory_state and memory_state.get("recall_triggered") and memory_state.get("recall_target"):
            target = memory_state["recall_target"]
            target_landmarks = target.get("landmarks", []) or []
            if self._is_valid_landmarks(target_landmarks):
                # 触发 recall：一次性生成多个历史 ghost 图层（不再每帧实时更新），
                # 使其表现为运动残影而非播放卡顿的人形动画。
                try:
                    self._spawn_history_ghosts(target, after_color, w, h)
                    print("[renderer] recall accepted: spawned history ghosts")
                except Exception:
                    print("[renderer] recall accepted but failed to spawn ghosts")
                # 不保持 recall_state，避免后续帧继续以历史 target 实时绘制
                self.recall_active = False
                self.recall_state = None
            else:
                self.recall_active = False
                self.recall_state = None
                print("[renderer] recall skipped: history pose invalid")
        elif self.recall_active and self.recall_state is not None:
            print("[renderer] recall holding history pose")
        else:
            self.recall_active = False
            self.recall_state = None

        if self.recall_state is not None:
            target = self.recall_state.get("target", {})
            landmarks = target.get("landmarks", []) or []
            if self._is_valid_landmarks(landmarks):
                elapsed = now - self.recall_state.get("start", now)
                duration = max(0.01, self.recall_state.get("duration", 2.2))
                t = min(1.0, elapsed / duration)
                t = t * t * (3.0 - 2.0 * t)
                # 增强 alpha 基准并根据 recall_timer 做淡出
                base_alpha = max(0.25, 0.35 + 0.5 * t)
                remaining = max(0.0, self.recall_timer - now) if hasattr(self, 'recall_timer') else max(0.0, duration - elapsed)
                fade = (remaining / duration) if duration > 0 else 0.0
                final_alpha = base_alpha * fade
                # 保证在回忆期内不至于完全透明
                if fade > 0.0:
                    final_alpha = max(final_alpha, 0.12)

                offset = self.recall_state.get("offset", (0, 0))
                # 优先使用历史保存的小尺寸 mask（如果存在），以还原为与 realtime 相同的剪影
                target_mask_small = target.get("mask")
                if target_mask_small:
                    try:
                        small = np.array(target_mask_small, dtype=np.uint8)
                        small = (small > 0).astype(np.uint8) * 255
                        mask_full = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

                        # 应用像素偏移（使用 np.roll 然后清空包裹部分）
                        ox, oy = int(offset[0]), int(offset[1])
                        if ox != 0 or oy != 0:
                            mask_shifted = np.roll(mask_full, shift=(oy, ox), axis=(0, 1))
                            # 清理 wrap-around 区域
                            if oy > 0:
                                mask_shifted[:oy, :] = 0
                            elif oy < 0:
                                mask_shifted[oy:, :] = 0
                            if ox > 0:
                                mask_shifted[:, :ox] = 0
                            elif ox < 0:
                                mask_shifted[:, ox:] = 0
                        else:
                            mask_shifted = mask_full

                        # 羽化并混合（与实时绘制方式一致）
                        k = getattr(cfg, 'SHADOW_EDGE_FEATHER', 21)
                        if k % 2 == 0:
                            k = k + 1
                        soft = cv2.GaussianBlur(mask_shifted, (k, k), 0).astype(np.float32) / 255.0
                        soft3 = soft[:, :, np.newaxis]
                        fg = np.array(fg_color, dtype=np.float32)
                        overlay = fg * soft3
                        canvas = canvas * (1.0 - final_alpha) + overlay * final_alpha
                        print("[renderer] using history pose (mask)")
                    except Exception:
                        canvas = self._draw_history_ghost(canvas, landmarks, final_alpha, np.array(fg_color, dtype=np.float32), offset=offset)
                        print("[renderer] using history pose (fallback hull)")
                else:
                    # 没有保存的 mask，回退到由关键点生成的凸包掩码方法
                    canvas = self._draw_history_ghost(canvas, landmarks, final_alpha, np.array(fg_color, dtype=np.float32), offset=offset)
                    print("[renderer] using history pose")

                if elapsed >= duration:
                    self.recall_state = None
            else:
                print("[renderer] history pose invalid, fallback to realtime")
                self.recall_state = None
        else:
            print("[renderer] using realtime pose")

        if memory_state is not None and (self.recall_active or self.recall_state is not None):
            memory_state["recall_triggered"] = True

        # 裁剪到合法范围并转回 uint8 显示
        return np.clip(canvas, 0, 255).astype(np.uint8)
