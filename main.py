import cv2
import mediapipe as mp
import json
import time


# 初始化 MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_draw = mp.solutions.drawing_utils


# 保存动作历史
motion_history = []


# 打开摄像头
cap = cv2.VideoCapture(0)


while True:

    ret, frame = cap.read()

    if not ret:
        break


    # BGR 转 RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # 人体姿态检测
    results = pose.process(rgb_frame)


    landmarks_data = []


    if results.pose_landmarks:

        for landmark in results.pose_landmarks.landmark:

            landmarks_data.append({
                "x": landmark.x,
                "y": landmark.y,
                "z": landmark.z
            })


        # 保存当前人体状态
        motion_data = {
            "time": time.time(),
            "landmarks": landmarks_data
        }


        motion_history.append(motion_data)


        # 限制历史长度
        if len(motion_history) > 100:
            motion_history.pop(0)


        # 绘制人体骨架
        mp_draw.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )


    # 显示结果

    cv2.imshow(
        "Python Shadow Tracking",
        frame
    )


    # 按 q 退出

    if cv2.waitKey(1) & 0xFF == ord('q'):

        break



# 保存数据

with open(
    "motion_history.json",
    "w"
) as f:

    json.dump(
        motion_history,
        f,
        indent=4
    )


cap.release()

cv2.destroyAllWindows()