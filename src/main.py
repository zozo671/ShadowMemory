"""
main.py
《存在的残像：数字影子记忆系统》程序入口。
串联：PoseTracker → MemoryStore → BehaviorAnalyzer → ShadowRenderer
"""

import time

import cv2

import config as cfg
from pose_tracker import PoseTracker
from memory_store import MemoryStore
from analyzer import BehaviorAnalyzer
from shadow_renderer import ShadowRenderer


def main():
    # 1. 初始化各模块
    tracker = PoseTracker()
    memory = MemoryStore()
    analyzer = BehaviorAnalyzer()
    renderer = ShadowRenderer()

    print("系统启动：按 'q' 退出")

    frame_counter = 0
    try:
        while True:
            # 2. 打开摄像头并获取人体姿态
            ret, frame = tracker.read_frame()
            if not ret:
                break

            # 摄像头原始输出为镜像，水平翻转一次得到正向画面
            frame = cv2.flip(frame, 1)

            frame_counter += 1

            # 性能优化：一次转换同时获取姿态与人体 mask
            pose_data, body_mask = tracker.get_pose_and_mask(frame)

            # 4. 分析历史行为（先分析，拿到动作持续时间）
            memory_state = analyzer.analyze_frame(pose_data, memory, body_mask=body_mask)

            # 3. 保存动作记忆（仅当检测到人体时，duration 来自 analyzer）
            landmarks = pose_data.get("landmarks", [])
            if landmarks:
                memory.save_action(
                    pose_data["timestamp"],
                    landmarks,
                    duration=memory_state["duration"],
                    action_category=memory_state["action_category"],
                    move_direction=memory_state.get("move_dir", "idle"),
                )

            # 5. 根据分析结果生成数字影子（黑底白影 + 残影）
            display = renderer.update(frame, pose_data, body_mask, memory_state)
            print(
                f"memory={memory.count()} recall={'yes' if memory_state.get('recall_triggered') else 'no'} "
                f"source={'history' if memory_state.get('recall_triggered') else 'realtime'}"
            )

            # 6. 实时显示
            cv2.imshow(cfg.WINDOW_NAME, display)

            # 每 30 帧将内存记录落盘一次（避免每帧写文件导致卡顿）
            if frame_counter % 30 == 0:
                memory.flush()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # 释放资源并保存记忆
        memory.flush()
        tracker.release()
        print(f"记忆已保存至 {cfg.MEMORY_FILE}，共 {memory.count()} 条记录")


if __name__ == "__main__":
    main()