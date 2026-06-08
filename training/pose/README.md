# 🧘 Pose Detection System (姿态检测系统)

本项目包含两套姿态检测方案，适用于不同的硬件性能和应用场景。

## 📂 文件说明

### 1. `main_lstm.py` (推荐：高精度版)
*   **核心算法**：MediaPipe Pose + LSTM 深度学习模型。
*   **适用场景**：PC、高性能笔记本、Jetson Nano。
*   **特点**：
    *   通过 LSTM 时序模型分析动作序列，抗干扰能力强。
    *   依赖 `posture_lstm.h5` 模型文件。
    *   包含平滑滤波和完整的 UI 交互。

### 2. `main_rule.py` (推荐：轻量版)
*   **核心算法**：MediaPipe Pose + 几何规则计算。
*   **适用场景**：树莓派、旧电脑、CPU 较弱的设备。
*   **特点**：
    *   纯数学计算（通过计算耳肩距离、脊柱角度等判断）。
    *   响应速度极快，无需加载额外模型。
    *   可通过调整 `CONFIG` 中的阈值来适配不同用户。

### 3. 辅助文件
*   `posture_lstm.h5`: 训练好的 LSTM 模型文件（`main_lstm.py` 必须依赖此文件）。
*   `train.py`: 用于重新训练 LSTM 模型的脚本。
*   `collect.py`: 用于采集训练数据的工具（自动保存为CSV）。
*   `data_process.py`: 数据预处理脚本（将CSV转换为LSTM所需的NPY序列数据）。

## 🚀 如何运行

**运行高精度版 (LSTM):**
```bash
python main_lstm.py
```

**运行轻量版 (规则):**
```bash
python main_rule.py
```
