# 存在的残像：基于 Python 与人体姿态识别的数字影子记忆系统

## 项目简介

《存在的残像》是一个基于 Python 计算机视觉的交互艺术项目。用户进入摄像头范围后，系统生成一个"数字影子"。影子不是简单跟随用户，而是会**保存用户过去的动作**，并根据历史行为产生不同反馈（高频动作产生更明显残影、长时间停留位置留下更久影子、新动作产生不同反馈）。

本项目的"学习"不是训练 AI 模型，而是基于历史数据的**记忆系统**：用 JSON 保存行为数据，用简单数学算法分析趋势并改变影子表现。

## 核心理念

现实中的影子随人移动而变化，但无法留下过去的痕迹。本项目尝试构建一个能感知人体行为、记录动作轨迹的数字影子，使用户过去的动作以**数字残像**的形式被保存与重现。

## 技术栈

- **Python 3.11**（⚠️ 见下方"环境要求"）
- **OpenCV** — 读取摄像头、图像处理
- **MediaPipe Pose** (`mp.solutions.pose`) — 人体 33 个关键点检测
- **NumPy** — 关键点数据处理、向量计算
- **JSON** — 历史行为数据持久化（无数据库）

## 环境要求（重要）

本项目使用 MediaPipe 的旧版 `mp.solutions.pose` API，该 API **仅存在于 MediaPipe ≤ 0.10.14**，且**不支持 Python 3.13 / 3.14**。

你的系统默认是 Python 3.13，因此**必须**使用 Python 3.11 虚拟环境运行。项目内已创建好 `venv/`（基于 Python 3.11.9）。

> 若需自行重建环境：
> ```bash
> winget install Python.Python.3.11
> py -3.11 -m venv venv
> venv\Scripts\python.exe -m pip install -r requirements.txt
> ```

## 项目结构

```
shadowMemory/
├── README.md            # 项目说明（本文件）
├── requirements.txt     # 依赖（mediapipe==0.10.14, opencv-python, numpy）
├── config.py            # 全局配置参数（阈值、颜色、衰减系数等）
├── pose_tracker.py      # PoseTracker 类：摄像头 + MediaPipe 关键点提取
├── memory_store.py      # MemoryStore 类：JSON 行为记忆读写
├── analyzer.py          # BehaviorAnalyzer 类：历史趋势分析
├── shadow_renderer.py   # ShadowRenderer 类：当前影子 + 残影叠加渲染
├── main.py              # 主循环：串联以上模块
├── memory.json          # 运行时自动生成的历史行为数据
└── venv/                # Python 3.11 虚拟环境（已装好依赖）
```

## 各文件作用

| 文件 | 类 / 职责 |
|------|-----------|
| `config.py` | 集中管理所有可调参数：摄像头索引、关键点置信度阈值、动作判定阈值、残影数量与衰减、颜色映射等 |
| `pose_tracker.py` | `PoseTracker`：打开摄像头，调用 MediaPipe Pose 提取 33 个关键点（用于行为记忆/分析），同时调用 MediaPipe **Selfie Segmentation** 生成人体二值 mask（用于剪影）；提供 `get_pose_data()` / `get_body_mask()` / `release()` |
| `memory_store.py` | `MemoryStore`：将每次"动作片段"（时间、关键点序列、持续时间、变化趋势）追加写入 `memory.json`；提供加载、统计、按动作类型聚合的接口 |
| `analyzer.py` | `BehaviorAnalyzer`：读取记忆，计算重复次数、停留时长、移动频率变化，输出"高频动作 / 长停留位置 / 新动作"等反馈信号 |
| `shadow_renderer.py` | `ShadowRenderer`：用人体 mask 在纯色背景上绘制**白色（或黑色）人体剪影**，并把历史 mask 以透明度递减的残影叠加，形成"数字残像"；不再显示 MediaPipe 骨架 |
| `main.py` | 初始化各模块，进入 `while` 循环：取帧 → 姿态+分割 → 记忆 → 分析 → 渲染剪影 → 显示；按 `q` 退出并保存记忆 |

## 数据流流程

```
摄像头 (OpenCV)
   │  BGR 帧
   ▼
PoseTracker.get_pose_data(frame)  →  33 个关键点 (x, y, z)   [用于行为记忆/分析]
PoseTracker.get_body_mask(frame)  →  人体二值 mask (0/255)    [用于剪影]
   │
   ▼
[记忆] 关键点 → MemoryStore（时间/关键点/时长/趋势）→ memory.json
   │
   ▼
BehaviorAnalyzer.analyze()  →  重复次数↑ / 停留时长↑ / 移动频率变化 → 反馈信号
   │
   ▼
ShadowRenderer.update(frame, pose_data, body_mask, memory_state)
   │  ① 历史 mask 残影：透明度递减叠加（越早越淡）
   │  ② 当前人体 mask → 纯色背景上的白色（或黑色）剪影
   ▼
画面 = 黑底（或白底）+ 当前人体剪影 + 过去残影（透明度递减）
   │
   ▼
cv2.imshow() 显示（不再显示摄像头原画面与骨架）
```

## 运行方式

使用项目内已配置好的 Python 3.11 虚拟环境：

```bash
# 1. 激活虚拟环境（Windows PowerShell）
.\venv\Scripts\Activate.ps1

# 2. 运行主程序
python main.py
```

程序启动后自动调用摄像头（默认索引 0）。检测到人体时显示蓝色"当前影子"；历史动作会以递减透明度的残影重现。按 `q` 退出，记忆自动保存到 `memory.json`。

## 功能对照（需求 → 实现）

1. **实时检测人体** → `PoseTracker` 每帧调用 MediaPipe Pose + Selfie Segmentation
2. **提取关键点** → 33 个关键点 (x, y, z) 用于行为记忆；人体 mask 用于剪影
3. **行为记忆模块** → `MemoryStore` 保存：时间 / 关键点 / 动作持续时间 / 变化趋势
4. **分析历史行为** → `BehaviorAnalyzer`：重复次数、停留时长、移动频率
5. **改变影子** → 高频动作残影更明显、长停留位置影子更久、新动作不同反馈
6. **视觉效果** → **黑底白色人体剪影**（或白底黑影）+ 过去残影（透明度递减），不显示摄像头原画面与 MediaPipe 骨架

## 视觉风格说明

- 默认 `SHADOW_BG_MODE = "black"`：**黑底 + 白色人体剪影**（最符合"数字残像"氛围）。
- 可在 `config.py` 改为 `"white"` 得到**白底 + 黑色剪影**。
- 残影使用 `Selfie Segmentation` 的 mask 历史帧，按时间透明度递减叠加，形成拖尾残像。
- 所有视觉参数集中在 `config.py` 的"数字影子视觉风格"段落，便于演示时快速调参。

## 开发说明

- 不训练模型、不使用深度学习、不使用数据库——纯 JSON + NumPy 数学算法实现"记忆"。
- 所有可调参数集中在 `config.py`，便于演示时快速调参。
- 历史数据存于 `memory.json`，删除该文件即可重置记忆。