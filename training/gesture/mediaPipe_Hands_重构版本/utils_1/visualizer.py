import cv2
from mediapipe import solutions as mp
def draw_landmarks(image, landmarks, connections=True):
    """
    在图像上绘制手部关键点和连线
    参数:
        image: 要绘制的图像
        landmarks: 关键点坐标列表
        connections: 是否绘制连接线
    """
    if landmarks is None:
        return image
    
    # 绘制关键点
    for idx, (x, y) in enumerate(landmarks):
        cv2.circle(image, (x, y), 3, (0, 255, 0), -1)
        cv2.putText(image, str(idx), (x+5, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,0,0), 1)
    
    # 绘制连接线
    if connections:
        # MediaPipe定义的连接关系
        HAND_CONNECTIONS = mp.hands.HAND_CONNECTIONS
        for connection in HAND_CONNECTIONS:
            start = landmarks[connection[0]]
            end = landmarks[connection[1]]
            cv2.line(image, start, end, (255,0,0), 1)
    
    return image

def display_status(image, gestures, fps=None):
    """
    在图像上显示识别结果
    参数:
        image: 要显示的图像
        gestures: 手势名称列表
        fps: 可选，显示帧率
    """
    y_pos = 30
    # 显示帧率
    if fps is not None:
        cv2.putText(image, f"FPS: {fps:.1f}", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        y_pos += 30
    
    # 显示手势识别结果
    if gestures:
        for i, gesture in enumerate(gestures):
            cv2.putText(image, gesture, (10, y_pos+30*i),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    else:
        cv2.putText(image, "No Gesture Detected", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
    return image
