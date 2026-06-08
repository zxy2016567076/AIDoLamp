import cv2
import joblib
import os
import time
import mediapipe as mp

class GestureRecognizer:
    def __init__(self):
        # 模型初始化
        self.model = joblib.load(os.path.join('data', 'processed', 'gesture_model.pkl'))
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # 状态控制变量
        self.system_active = False     # 系统总开关
        self.activation_counter = 0    # 激活计时器
        self.current_mode = None       # 当前模式（1-4）
        self.last_print_time = 0       # 防重复输出计时

    def process_frame(self, frame):
        """处理视频帧并返回带标注的帧和预测标签"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        
        if not results.multi_hand_landmarks:
            return frame, None
        
        # 提取手部关键点特征
        hand_landmarks = results.multi_hand_landmarks[0]
        keypoints = []
        for lm in hand_landmarks.landmark:
            keypoints.extend([lm.x, lm.y, lm.z])
        
        # 预测手势标签
        return frame, self.model.predict([keypoints])[0] if len(keypoints) == 63 else None

    def real_time_recognition(self):
        """实时识别主循环"""
        cap = cv2.VideoCapture(0)
        gesture_buffer = []  # 手势稳定性缓冲区
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            # 预处理帧
            frame = cv2.flip(frame, 1)
            processed_frame, label = self.process_frame(frame)
            current_time = time.time()
            
            # ==================== 状态机逻辑 ====================
            # 状态 0: 系统未激活
            if not self.system_active:
                if label == 1:  # 检测到激活手势
                    self.activation_counter += 1
                    
                    # 持续检测到激活手势3秒（约30帧/秒 * 3秒 = 90帧）
                    if self.activation_counter > 90:
                        self.system_active = True
                        self.activation_counter = 0
                        cv2.putText(processed_frame, "SYSTEM ACTIVATED!", (50, 150), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    self.activation_counter = 0  # 重置计数器
                
                # 显示激活进度条
                cv2.rectangle(processed_frame, (20, 400), (220, 440), (255, 255, 255), 2)
                cv2.rectangle(processed_frame, (20, 400), 
                             (20 + int(200 * (self.activation_counter/90)), 440), 
                             (0, 255, 0), -1)
                
            # 状态 1: 系统已激活
            else:
                # 检测退出手势
                if label == 3:
                    self.system_active = False
                    self.current_mode = None
                    cv2.putText(processed_frame, "SYSTEM DEACTIVATED", (50, 150), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    continue
                
                # 模式选择逻辑
                if self.current_mode is None:
                    mode_dict = {5:1, 6:2, 7:3, 8:4}
                    if label in mode_dict:
                        self.current_mode = mode_dict[label]
                        cv2.putText(processed_frame, f"MODE {self.current_mode} ACTIVATED!", 
                                  (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                
                # 模式控制逻辑        
                else:
                    control_map = {
                        10: ("UP", 1),
                        2: ("DOWN", 2), 
                        9: ("RIGHT",3),
                        4: ("LEFT",4)
                    }
                    
                    if label in control_map:
                        # 防重复输出（0.5秒间隔）
                        if current_time - self.last_print_time > 0.5:
                            print(control_map[label][1])
                            self.last_print_time = current_time
                            
                        # 在画面上显示控制方向    
                        cv2.putText(processed_frame, control_map[label][0], 
                                  (300, 150), cv2.FONT_HERSHEY_SIMPLEX, 
                                  2, (0, 255, 255), 3)
                    
                    # 退出当前模式
                    if label == 3:
                        self.current_mode = None

            # ==================== 界面显示 ====================
            status_text = f"ACTIVE: {self.system_active}  MODE: {self.current_mode or 'None'}"
            cv2.putText(processed_frame, status_text, (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            if label is not None:
                cv2.putText(processed_frame, f"Gesture: {label}", (20, 80), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('Gesture Control System', processed_frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC退出
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    recognizer = GestureRecognizer()
    recognizer.real_time_recognition()
