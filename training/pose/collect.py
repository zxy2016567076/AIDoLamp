import cv2
import mediapipe as mp
import numpy as np
import csv

# 初始化MediaPipe Pose和绘图工具
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils  # 关键点绘图工具
pose = mp_pose.Pose()

# 打开摄像头
cap = cv2.VideoCapture(0)

# 创建CSV文件保存关键点和标签
csv_file = open('posture_data.csv', 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['label'] + [f'x{i}' for i in range(33)] + [f'y{i}' for i in range(33)])

label = 0  # 0=正常，1=异常（手动标注）

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # 将帧转换为RGB格式（MediaPipe需要RGB输入）
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 检测关键点
    results = pose.process(image_rgb)
    
    if results.pose_landmarks:
        # 在图像上绘制关键点和骨架连线
        mp_drawing.draw_landmarks(
            frame,  # 绘制在BGR格式的帧上
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,  # 预定义的骨架连接关系
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),  # 关键点颜色
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2))  # 连线颜色

        # 提取所有关键点的x,y坐标（归一化到0~1）
        landmarks = results.pose_landmarks.landmark
        row = [label] + [lmk.x for lmk in landmarks] + [lmk.y for lmk in landmarks]
        csv_writer.writerow(row)

    # 显示标签状态
    cv2.putText(frame, f"Label: {label} (0=Normal, 1=Unusual)", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # 显示画面
    cv2.imshow('Data Collection - 按0/1标注，Q退出', frame)
    
    # 键盘操作
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('0'):  # 按0键标记为正常
        label = 0
    elif key == ord('1'):  # 按1键标记为异常
        label = 1

# 释放资源
cap.release()
csv_file.close()
cv2.destroyAllWindows()