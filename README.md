\# 存在的残像：基于 Python 与人体姿态识别的数字影子记忆系统



\## 项目背景



《存在的残像》是一项基于人工智能与计算机视觉技术的交互艺术项目。



项目希望探索人与数字系统之间关于“存在”和“记忆”的关系。



现实中的影子会随着人的移动产生变化，但无法留下过去的痕迹。本项目尝试通过 Python 构建一个能够感知人体行为并记录动作轨迹的数字影子系统，使用户过去的动作以数字残像的形式被保存。



本项目以 Python 编程为核心，通过计算机视觉和人体姿态识别技术实现智能感知，并结合 TouchDesigner 完成交互视觉展示。



\## 项目目标



\* 学习并应用 Python 进行计算机视觉开发

\* 实现实时人体姿态检测

\* 分析人体动作变化

\* 建立用户运动轨迹记录系统

\* 将计算结果转化为数字影子视觉效果



\## 系统功能



当前版本实现：



1\. 摄像头实时视频采集



2\. 基于 OpenCV 的图像处理



3\. 基于 MediaPipe Pose 的人体关键点检测



4\. 人体运动数据提取



5\. 用户动作历史数据保存



后续版本计划：



\* 将 Python 数据传输至 TouchDesigner

\* 实现实时数字影子生成

\* 添加残像、粒子、流体等视觉效果

\* 根据用户动作产生不同视觉反馈



\## 技术架构



```

Camera

&#x20;  |

&#x20;  v

OpenCV

&#x20;  |

&#x20;  v

MediaPipe Pose

&#x20;  |

&#x20;  v

Python Motion Analysis

&#x20;  |

&#x20;  v

Movement History Data

&#x20;  |

&#x20;  v

TouchDesigner Visualization

```



\## 技术栈



\* Python

\* OpenCV

\* MediaPipe

\* NumPy

\* TouchDesigner



\## 运行方式



\### 1. 安装依赖



```

pip install -r requirements.txt

```



\### 2. 运行程序



```

python main.py

```



\### 3. 使用摄像头



程序启动后，将自动调用摄像头并检测人体姿态。



\## 项目结构



```

ShadowMemory



├── README.md



├── main.py



└── requirements.txt

```



\## 开发方向



未来将继续完善：



\* 更准确的人体动作识别算法

\* Python 与 TouchDesigner 数据通信

\* 实时影子生成系统

\* 基于AI的动作记忆和视觉生成



