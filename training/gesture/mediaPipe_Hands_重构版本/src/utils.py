# 几何计算工具

import numpy as np

def landmark_to_array(hand_landmarks, image_shape):
    """
    将MediaPipe的手部关键点转换为坐标数组
    参数:
        hand_landmarks: MediaPipe输出的手部关键点对象
        image_shape: 图像的形状 (height, width, channels)
    返回:
        List[Tuple[int, int]]: 包含(x,y)坐标的列表
    """
    h, w = image_shape[:2]
    return [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]

def calculate_angle(a, b, c):
    """
    计算三点之间的夹角(以b为顶点)
    参数:
        a,b,c: 三点坐标，每个点格式为(x,y)
    返回:
        float: 角度值(单位：度)
    """
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosine_angle, -1, 1)))
