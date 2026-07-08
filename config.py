"""
config.py
全局参数配置文件
保存整个《存在的残像》数字影子记忆系统的可调参数。
所有参数均带中文注释，方便一天内快速调参。
"""

# ===================== 摄像头与窗口 =====================
CAMERA_INDEX = 0          # 摄像头编号（默认 0 为内置摄像头，外接触发可改为 1）
WINDOW_WIDTH = 1280       # 视频窗口宽度（像素）
WINDOW_HEIGHT = 720       # 视频窗口高度（像素）
WINDOW_NAME = "存在的残像 · 数字影子记忆系统"  # 显示窗口标题

# ===================== MediaPipe 检测参数 =====================
MIN_DETECTION_CONFIDENCE = 0.5   # 人体检测最低置信度（0~1，越高越严格）
MIN_TRACKING_CONFIDENCE = 0.5    # 关键点跟踪最低置信度（0~1）
POSE_MODEL_COMPLEXITY = 1        # 模型复杂度（0=轻量，1=完整，2=重型）
SEGMENTATION_MODEL = 1           # Selfie Segmentation 模型（0=轻量，1=完整）
SEGMENTATION_THRESHOLD = 0.6     # 人体分割阈值（0~1，高于该值视为人体像素）

# ===================== 历史记忆保存 =====================
MEMORY_DIR = "data"                      # 记忆数据存放目录
MEMORY_FILE = "data/memory.json"         # 历史行为记忆 JSON 文件路径
MAX_MEMORY_RECORDS = 2000                # 内存中保留的最大行为记录数（防止无限增长）

# ===================== 残影（视觉拖尾）参数 =====================
AFTERIMAGE_COUNT = 12            # 残影数量（同时叠加的历史影子层数）
AFTERIMAGE_DECAY = 0.82          # 残影透明度衰减参数（0~1，越接近 1 残影消失越慢）
AFTERIMAGE_INTERVAL = 5          # 每隔多少帧采样一次残影（降低采样密度，拉长拖尾时间）

# ===================== 行为分析阈值 =====================
STAY_THRESHOLD = 0.02            # 重心位移小于该值视为"停留"（归一化坐标）
MOVE_FREQ_WINDOW = 30            # 计算移动频率的滑动窗口帧数
REPEAT_ACTION_WINDOW = 300       # 统计动作重复次数的时间窗口（秒）
HIGH_FREQ_COUNT = 5              # 某动作出现次数超过该值视为"高频动作"

# ===================== 颜色配置（BGR 格式） =====================
COLOR_CURRENT = (0, 255, 255)    # 当前影子颜色（青黄）
COLOR_PAST = (255, 0, 0)         # 过去影子/高频残影颜色（蓝）
COLOR_NEW = (0, 255, 0)          # 新动作反馈颜色（绿）

# ===================== 数字影子视觉风格 =====================
# 背景模式："black"（黑底白影）或 "white"（白底黑影）
SHADOW_BG_MODE = "black"
SHADOW_FG_COLOR = (255, 255, 255)   # 人体剪影前景色（白）
SHADOW_BG_COLOR = (0, 0, 0)         # 背景色（黑）
# 残影（历史剪影）颜色：默认偏冷蓝，与当前白影区分
SHADOW_AFTERIMAGE_COLOR = (180, 180, 200)
# 残影模糊半径（像素），让残像更柔和
AFTERIMAGE_BLUR = 7
