import cv2
import joblib
import numpy as np
import mediapipe as mp
#from datetime import datetime
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
        
    def process_frame(self, frame):
        """处理单个帧并返回预测结果"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        
        if not results.multi_hand_landmarks:
            return frame, None
        
        # 提取特征
        hand_landmarks = results.multi_hand_landmarks[0]
        keypoints = []
        for lm in hand_landmarks.landmark:
            keypoints.extend([lm.x, lm.y, lm.z])
        
        # 预测
        if len(keypoints) != 63:
            return frame, None
        
        label = self.model.predict([keypoints])[0]
        return frame, label

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
            
            # 显示结果
            if label is not None:
                cv2.putText(processed_frame, f"Label: {label}", (20, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                
            cv2.imshow('Gesture Recognition', processed_frame)
            
            if cv2.waitKey(1) & 0xFF == 27:  # ESC退出
                break
                
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    recognizer = GestureRecognizer()
    recognizer.real_time_recognition()
