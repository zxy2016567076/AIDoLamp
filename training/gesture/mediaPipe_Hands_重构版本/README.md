项目说明
gesture_recognition/
├── config/                 # 配置管理
│   ├── model_config.yaml   # 模型超参数配置
│   └── thresholds.yaml     # 手势判断阈值
├── data/                   # 数据管理
│   ├── raw/                # 原始手势图片（按类别存放）
│   │   ├── thumbs_up/
│   │   └── open_palm/
│   └── processed/          # 预处理后的特征数据
├── data_preparation/       # 数据预处理模块
│   ├── preprocessor.py     # 传统特征提取器
│   └── dl_preprocessor.py  # 深度学习数据增强
├── models/                 # 模型相关
│   ├── train_model.py      # 传统模型训练
│   └── nn_model.py         # 神经网络模型
├── src/                    # 核心功能实现
│   ├── gesture_detector.py # 实时检测逻辑
│   └── utils.py            # 几何计算工具
├── utils/                  # 辅助工具
│   ├── visualizer.py       # 可视化绘制
│   └── camera.py           # 摄像头工具类
└── main.py                 # 主程序入口

config/	集中管理所有可配置参数，便于快速调整算法行为	被所有功能模块引用
data/raw	原始数据采集区（需手动创建子文件夹存放不同手势类别的图片）	手动维护
data_preparation/	数据处理流水线：包含媒体pipe特征提取和图像增强两种预处理策略	在训练前执行
models/	训练和保存分类模型：传统机器学习模型或深度学习模型	依赖预处理后的数据
src/gesture_detector.py	实时检测中枢：集成MediaPipe检测、手势判断逻辑	主程序运行时核心组件
utils/visualizer.py	可视化输出工具：关键点绘制、状态显示、FPS计算	被主程序调用

config/: 配置文件目录，包含模型超参数配置和手势判断阈值等。
data/: 数据管理目录，包含原始手势图片和预处理后的特征数据。
data_preparation/: 数据预处理模块，包含传统特征提取器和深度学习数据增强。
models/: 模型相关目录，包含传统机器学习模型和神经网络模型的训练脚本。
src/: 核心功能实现目录，包含实时检测逻辑和几何计算工具。
utils/: 辅助工具目录，包含可视化绘制和摄像头工具类。
main.py: 主程序入口，整合所有功能，支持语音反馈和连续手势检测。


配置文件：

app_config.yaml: 主程序控制逻辑（手势序列定义）
thresholds.yaml: 手势判断具体参数（关节点阈值）
model_config.yaml: 模型训练参数
gesture_mappings.yaml: 语音反馈映射
核心功能模块：

gesture_detector.py: 实时手势检测核心逻辑
gesture_sequence.py: 连续手势序列识别器
audio_feedback.py: 语音反馈系统
data_collector.py: 手势数据采集工具
数据处理模块：

preprocessor.py: MediaPipe特征提取器
dl_preprocessor.py: 深度学习数据增强工具
模型相关：

nn_model.py: CNN神经网络模型
traditional_model.py: 传统机器学习模型
主程序：

main_1.py: 程序主入口