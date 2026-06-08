import numpy as np
import pandas as pd
from scipy.signal import savgol_filter  # 平滑处理

# 读取CSV数据
data = pd.read_csv('posture_data.csv')
labels = data['label'].values
keypoints = data.drop('label', axis=1).values

# 定义关键点索引（MediaPipe Pose的33个关键点）
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_EAR = 7
RIGHT_EAR = 8

def calculate_features(frame_keypoints):
    """从单帧关键点中提取特征"""
    features = []
    
    # 获取关键点索引（确保与MediaPipe定义一致）
    LEFT_SHOULDER = 11  # MediaPipe Pose中左肩的索引是11
    RIGHT_SHOULDER = 12
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_EAR = 7
    RIGHT_EAR = 8

    # 1. 计算肩部中心点（x和y分别取平均）
    left_shoulder_x = frame_keypoints[LEFT_SHOULDER * 2]
    left_shoulder_y = frame_keypoints[LEFT_SHOULDER * 2 + 1]
    right_shoulder_x = frame_keypoints[RIGHT_SHOULDER * 2]
    right_shoulder_y = frame_keypoints[RIGHT_SHOULDER * 2 + 1]
    shoulder_center_x = (left_shoulder_x + right_shoulder_x) / 2
    shoulder_center_y = (left_shoulder_y + right_shoulder_y) / 2
    shoulder_center = [shoulder_center_x, shoulder_center_y]

    # 2. 计算髋部中心点
    left_hip_x = frame_keypoints[LEFT_HIP * 2]
    left_hip_y = frame_keypoints[LEFT_HIP * 2 + 1]
    right_hip_x = frame_keypoints[RIGHT_HIP * 2]
    right_hip_y = frame_keypoints[RIGHT_HIP * 2 + 1]
    hip_center_x = (left_hip_x + right_hip_x) / 2
    hip_center_y = (left_hip_y + right_hip_y) / 2
    hip_center = [hip_center_x, hip_center_y]

    # 3. 计算脊柱弯曲角度（肩膀中心到髋部中心的连线与垂直方向的夹角）
    spine_vector = np.array([hip_center[0] - shoulder_center[0], hip_center[1] - shoulder_center[1]])
    vertical_vector = np.array([0, 1])  # 垂直方向参考向量
    angle = np.degrees(np.arccos(
        np.dot(spine_vector, vertical_vector) /
        (np.linalg.norm(spine_vector) * np.linalg.norm(vertical_vector))
    ))
    features.append(angle)

    # 4. 头部前倾距离（耳朵中心与肩膀中心的水平偏移）
    left_ear_x = frame_keypoints[LEFT_EAR * 2]
    left_ear_y = frame_keypoints[LEFT_EAR * 2 + 1]
    right_ear_x = frame_keypoints[RIGHT_EAR * 2]
    right_ear_y = frame_keypoints[RIGHT_EAR * 2 + 1]
    ear_center_x = (left_ear_x + right_ear_x) / 2
    head_forward = ear_center_x - shoulder_center[0]  # x方向的距离
    features.append(head_forward)

    # 5. 肩膀倾斜角度（左右肩高度差）
    shoulder_tilt = right_shoulder_y - left_shoulder_y
    features.append(shoulder_tilt)
    
    return np.array(features)

# 平滑处理（Savitzky-Golay滤波器）
features_seq = []
window_size = 30  # 时间窗口大小
for i in range(len(keypoints) - window_size + 1):
    window = keypoints[i:i+window_size]
    window_features = []
    for frame in window:
        frame_features = calculate_features(frame)
        window_features.append(frame_features)
    window_features = savgol_filter(window_features, window_length=5, polyorder=2, axis=0)  # 平滑
    features_seq.append(window_features)

features_seq = np.array(features_seq)
labels_seq = labels[window_size-1:]  # 对齐标签

# 保存为训练数据
np.save('features_seq.npy', features_seq)
np.save('labels_seq.npy', labels_seq)