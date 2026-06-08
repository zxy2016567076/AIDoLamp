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
        self.current_mode = None       # 当前模式（1-4）
        
        # 计时器系统
        self.activation_progress = 0   # 激活进度（0-100%）
        self.gesture_requirements = {
            1:  {"duration": 3.0, "progress": 0},  # 特殊处理激活手势
            "default": {"duration": 1.0, "progress": 0}
        }
        self.last_valid_gesture = None # 最后一个有效手势
        self.gesture_start_time = 0    # 手势开始时间

    def update_gesture_progress(self, label):
        """统一更新所有手势的识别进度"""
        current_time = time.time()
        
        # 首次识别或手势切换时重置计时
        if label != self.last_valid_gesture:
            self.last_valid_gesture = label
            self.gesture_start_time = current_time
            return 0
        
        # 计算所有手势的持续时间要求
        required_time = self.gesture_requirements.get(label, self.gesture_requirements["default"])["duration"]
        elapsed = current_time - self.gesture_start_time
        progress = min(int((elapsed / required_time) * 100), 100)
        
        return progress

    def real_time_recognition(self):
        """实时识别主循环"""
        cap = cv2.VideoCapture(0)
        last_execute_time = 0  # 最后一次操作执行时间
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            # 预处理帧
            frame = cv2.flip(frame, 1)
            processed_frame, label = self.process_frame(frame)
            current_progress = 0
            
            # ==================== 状态机逻辑 ====================
            # 实时更新进度条
            if label is not None:
                current_progress = self.update_gesture_progress(label)
                
            # 状态 0: 系统未激活
            if not self.system_active:
                if label == 1:
                    self.activation_progress = current_progress
                    
                    # 完整进度后激活系统
                    if current_progress >= 100:
                        self.system_active = True
                        self.activation_progress = 0
                        cv2.putText(processed_frame, "SYSTEM ACTIVATED!", (50, 150), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    self.activation_progress = 0  # 重置激活进度
            
            # 状态 1: 系统已激活
            else:
                # 退出系统检测（需要持续1秒）
                if label == 3 and current_progress >= 100:
                    self.system_active = False
                    self.current_mode = None
                    cv2.putText(processed_frame, "SYSTEM DEACTIVATED", (50, 150), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    self.last_valid_gesture = None
                    continue
                
                # 模式选择逻辑
                if self.current_mode is None:
                    mode_dict = {5:1, 6:2, 7:3, 8:4}
                    if label in mode_dict and current_progress >= 100:
                        self.current_mode = mode_dict[label]
                        cv2.putText(processed_frame, f"MODE {self.current_mode} ACTIVATED!", 
                                  (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                        self.last_valid_gesture = None  # 重置状态
                
                # 模式控制逻辑        
                else:
                    control_map = {
                        10: ("UP", 1),
                        2: ("DOWN", 2), 
                        9: ("RIGHT",3),
                        4: ("LEFT",4),
                        3: ("EXIT", "exit")
                    }
                    
                    if label in control_map and current_progress >= 100:
                        # 退出当前模式
                        if control_map[label][1] == "exit":
                            self.current_mode = None
                            cv2.putText(processed_frame, f"MODE EXITED", (50, 150), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                            self.last_valid_gesture = None
                        else:
                            # 防止高频重复操作（0.5秒间隔）
                            if time.time() - last_execute_time > 0.5:
                                print(control_map[label][1])
                                last_execute_time = time.time()
                            
                            # 在画面上显示控制方向    
                            cv2.putText(processed_frame, control_map[label][0], 
                                      (300, 150), cv2.FONT_HERSHEY_SIMPLEX, 
                                      2, (0, 255, 255), 3)
                        # 重置手势状态
                        self.last_valid_gesture = None

            # ==================== 可视化反馈 ====================
            # 绘制全局进度条
            if label is not None and label != 1:
                bar_width = 200
                cv2.rectangle(processed_frame, 
                            (300, 400), 
                            (300 + bar_width, 430), 
                            (100, 100, 100), 2)
                cv2.rectangle(processed_frame, 
                            (300, 400), 
                            (300 + int(bar_width*(current_progress/100)), 430), 
                            (0, 200, 200), -1)
                cv2.putText(processed_frame, f"{current_progress}%", 
                          (310, 425), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            
            # 系统激活专用进度条
            if not self.system_active:
                cv2.rectangle(processed_frame, (20, 400), (220, 440), (255, 255, 255), 2)
                cv2.rectangle(processed_frame, (20, 400), 
                            (20 + int(200 * (self.activation_progress/100)), 440), 
                            (0, 255, 0), -1)
            
            # 状态显示
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
