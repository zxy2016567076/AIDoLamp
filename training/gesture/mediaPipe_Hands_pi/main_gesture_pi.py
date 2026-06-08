# 导入所需库
import cv2
import joblib
import numpy as np
import mediapipe as mp
import time
import os
from picamera2 import Picamera2  # 树莓派专用摄像头库
from libcamera import Transform  # 用于摄像头硬件翻转配置

class GestureRecognizer:
    def __init__(self):
        # 加载手势识别模型（确保模型路径正确）
        self.model = joblib.load(os.path.join('gesture_model.pkl'))
        # 初始化MediaPipe手部检测模型
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,   # 非静态图像模式（适合实时视频）
            max_num_hands=1,           # 最多检测1只手
            min_detection_confidence=0.7,  # 检测置信度阈值
            min_tracking_confidence=0.5    # 跟踪置信度阈值
        )
        
        # 状态控制变量
        self.activated = False         # 系统是否激活
        self.activation_start_time = 0  # 激活开始时间戳
        self.current_mode = None        # 当前操作模式
        self.last_action_time = 0       # 上次操作时间戳
        self.gesture_timers = {}        # 手势计时器（记录各手势开始时间）
        self.confirmed_label = None     # 已确认的手势标签

    def process_frame(self, frame):
        """处理帧图像并返回预测结果"""
        # 将图像从BGR转换为RGB（MediaPipe需要RGB输入）
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 使用MediaPipe检测手部关键点
        results = self.hands.process(frame_rgb)
        
        if not results.multi_hand_landmarks:  # 未检测到手部
            return frame, None
        
        # 提取第一个检测到的手部关键点坐标
        hand_landmarks = results.multi_hand_landmarks[0]
        keypoints = []
        for lm in hand_landmarks.landmark:
            # 将归一化坐标转为实际坐标（x, y, z）
            keypoints.extend([lm.x, lm.y, lm.z])
        
        if len(keypoints) != 63:  # 关键点数量校验（21个点×3坐标）
            return frame, None
        
        # 使用模型预测手势标签
        label = self.model.predict([keypoints])[0]
        return frame, label

    def check_gesture_duration(self, current_label):
        """检查手势持续时间是否达标"""
        # 手势1需要3秒，其他手势需要1秒
        required_time = 3 if current_label == 1 else 1
        if current_label not in self.gesture_timers:
            self.gesture_timers[current_label] = time.time()
        return time.time() - self.gesture_timers[current_label] >= required_time

    def handle_mode_operations(self, label):
        """处理模式内的操作逻辑"""
        if self.current_mode and self.check_gesture_duration(label):
            if label == 10:   # 上
                print("1")
            elif label == 2:  # 下
                print("2")
            elif label == 9:  # 右
                print("3")
            elif label == 4:  # 左
                print("4")
            elif label == 3: # 退出模式
                print(f"退出模式{self.current_mode}")
                self.current_mode = None
            self.last_action_time = time.time()
            self.gesture_timers.clear()

    def real_time_recognition(self):
        """实时手势识别主函数"""
        # 初始化Picamera2并配置摄像头参数
        picam2 = Picamera2()
        config = picam2.create_video_configuration(
            main={"size": (320, 240)},  # 设置分辨率320x240以降低计算负载
            transform=Transform(hflip=True)  # 硬件水平翻转（无需软件处理）
        )
        picam2.configure(config)
        picam2.start()  # 启动摄像头

        while True:
            # 从摄像头捕获帧并转换为OpenCV兼容格式
            frame = picam2.capture_array()          # 获取RGB格式图像
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # 转为BGR供OpenCV使用
            
            # 处理帧图像并获取手势标签
            processed_frame, label = self.process_frame(frame)
            display_text = ""  # 初始化状态显示文本

            if label is not None:
                # 清除非当前手势的计时器
                for l in list(self.gesture_timers.keys()):
                    if l != label:
                        del self.gesture_timers[l]

                # 状态机逻辑
                if not self.activated:
                    if label == 1 and self.check_gesture_duration(label):
                        self.activated = True
                        self.gesture_timers.clear()
                        print("系统已激活！")
                    else:
                        self.activation_start_time = 0
                else:
                    if label == 3 and self.check_gesture_duration(label):
                        self.activated = False
                        self.current_mode = None
                        self.gesture_timers.clear()
                        print("系统已关闭")
                    elif label in [5, 6, 7, 8] and not self.current_mode:
                        if self.check_gesture_duration(label):
                            self.current_mode = label - 4  # 映射模式编号
                            print(f"进入模式{self.current_mode}")
                            self.gesture_timers.clear()
                    elif self.current_mode:
                        self.handle_mode_operations(label)

                # 更新显示文本
                display_text = f"Label: {label}"
                if self.activated:
                    display_text += " | activated"
                if self.current_mode:
                    display_text += f" | mode{self.current_mode}"

            # 绘制进度条
            if label in self.gesture_timers:
                duration = 3 if label == 1 else 1
                elapsed = time.time() - self.gesture_timers[label]
                progress = min(elapsed / duration, 1.0)
                cv2.rectangle(processed_frame, (20, 80), (220, 100), (255, 255, 255), 2)
                cv2.rectangle(processed_frame, (20, 80), 
                              (20 + int(200 * progress), 100), 
                              (0, 255, 0), -1)

            # 在图像上叠加状态文本
            cv2.putText(processed_frame, display_text, (20, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # 显示处理后的帧
            cv2.imshow('Gesture Recognition', processed_frame)
            
            # 按ESC退出
            if cv2.waitKey(1) & 0xFF == 27:
                break

        # 释放资源
        picam2.stop()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    recognizer = GestureRecognizer()
    recognizer.real_time_recognition()