"""
main.py
《存在的残像：数字影子记忆系统》程序入口。
串联：PoseTracker → MemoryStore → BehaviorAnalyzer → ShadowRenderer
"""

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

            frame_counter += 1

            pose_data = tracker.get_pose_data(frame)

            # 提取人体分割 mask（用于数字剪影）
            body_mask = tracker.get_body_mask(frame)

            # 4. 分析历史行为（先分析，拿到动作持续时间）
            memory_state = analyzer.analyze_frame(pose_data, memory)

            # 3. 保存动作记忆（仅当检测到人体时，duration 来自 analyzer）
            landmarks = pose_data.get("landmarks", [])
            if landmarks:
                memory.save_action(
                    pose_data["timestamp"],
                    landmarks,
                    duration=memory_state["duration"],
                )

            # 5. 根据分析结果生成数字影子（黑底白影 + 残影）
            display = renderer.update(frame, pose_data, body_mask, memory_state)

            # 叠加文字信息
            info = (
                f"动作总数:{memory_state['total_actions']} "
                f"近期:{memory_state['recent_count']} "
                f"移动频率:{memory_state['move_freq']:.3f} "
                f"强度:{memory_state['intensity']:.1f}"
            )
            cv2.putText(
                display, info, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2
            )
            if memory_state["high_freq"]:
                cv2.putText(
                    display, "高频动作残像增强",
                    (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    cfg.COLOR_PAST, 2
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