import cv2
import joblib
import numpy as np
import mediapipe as mp
import time
import os

class GestureRecognizer:
    def __init__(self):
        self.model = joblib.load(os.path.join('data', 'processed', 'gesture_model.pkl'))
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # 状态变量
        self.activated = False
        self.activation_start_time = 0
        self.current_mode = None
        self.last_action_time = 0
        self.last_label = None

    def process_frame(self, frame):
        """处理单个帧并返回预测结果和标签"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        
        if not results.multi_hand_landmarks:
            return frame, None
        
        # 提取特征
        hand_landmarks = results.multi_hand_landmarks[0]
        keypoints = []
        for lm in hand_landmarks.landmark:
            keypoints.extend([lm.x, lm.y, lm.z])
        
        if len(keypoints) != 63:
            return frame, None
        
        label = self.model.predict([keypoints])[0]
        return frame, label

    def handle_mode_operations(self, label):
        """处理模式内的操作逻辑"""
        if self.current_mode and time.time() - self.last_action_time > 1:  # 防抖1秒
            if label == 10:
                print("1")
            elif label == 2:
                print("2")
            elif label == 9:
                print("3")
            elif label == 4:
                print("4")
            elif label == 3:  # 退出当前模式
                print(f"退出模式{self.current_mode}")
                self.current_mode = None
            self.last_action_time = time.time()

    def real_time_recognition(self):
        """实时摄像头识别"""
        cap = cv2.VideoCapture(0)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # 镜像显示
            frame = cv2.flip(frame, 1)
            
            # 处理帧
            processed_frame, label = self.process_frame(frame)
            display_text = ""
            
            if label is not None:
                # 状态机逻辑
                if not self.activated:
                    if label == 1:
                        if self.activation_start_time == 0:
                            self.activation_start_time = time.time()
                        elif time.time() - self.activation_start_time > 3:
                            self.activated = True
                            self.activation_start_time = 0
                            print("系统已激活！")
                    else:
                        self.activation_start_time = 0
                else:
                    if label == 3:  # 退出整个系统
                        self.activated = False
                        self.current_mode = None
                        print("系统已关闭")
                    elif label in [5,6,7,8] and not self.current_mode:
                        self.current_mode = label - 4  # 5->1, 6->2等
                        print(f"进入模式{self.current_mode}")
                    elif self.current_mode:
                        self.handle_mode_operations(label)
                
                # 更新显示文本
                display_text = f"Label: {label}"
                if self.activated:
                    display_text += " | 已激活"
                if self.current_mode:
                    display_text += f" | 模式{self.current_mode}"

            # 绘制激活进度条
            if not self.activated and label == 1:
                elapsed = time.time() - self.activation_start_time
                cv2.rectangle(processed_frame, (20,80), (220,100), (255,255,255), 2)
                cv2.rectangle(processed_frame, (20,80), (20 + int(200*(elapsed/3)),100), (0,255,0), -1)

            # 显示文本
            cv2.putText(processed_frame, display_text, (20, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            
            cv2.imshow('Gesture Recognition', processed_frame)
            
            if cv2.waitKey(1) & 0xFF == 27:  # ESC退出
                break
                
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    recognizer = GestureRecognizer()
    recognizer.real_time_recognition()