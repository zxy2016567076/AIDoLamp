# main.py
import cv2
import time
import joblib
import numpy as np
from collections import deque
from utils_1.camera import Camera
from utils_1.visualizer import draw_landmarks, display_status

class GestureControlSystem:
    def __init__(self):
        # 硬件组件
        self.camera = Camera()
        self.hands = self._init_mediapipe()
        
        # 模型组件
        self.model, self.label_encoder = self._load_model()
        self.scaler = joblib.load("models/scaler.pkl")
        
        # 系统状态
        self.control_active = False    # 手动控制总开关
        self.current_mode = None       # 当前模式（1-4）
        self.activation_counter = 0    # 激活手势计数器
        self.gesture_buffer = deque(maxlen=30)  # 缓存最近手势
        self.last_operation_time = time.time()  # 防误触计时
        
        # 常量配置
        self.ACTIVATION_THRESHOLD = 25     # 需要连续25帧激活手势（约1秒）
        self.MODE_HOLD_DURATION = 15       # 模式切换需保持15帧
        self.STABLE_FRAME_THRESHOLD = 5    # 手势稳定检测阈值

    def _init_mediapipe(self):
        import mediapipe as mp
        return mp.solutions.hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )

    def _load_model(self):
        model_data = joblib.load("models/advanced_model.pkl")
        return model_data['model'], model_data['label_encoder']

    def _preprocess_landmarks(self, landmarks):
        """将关键点数据转换为模型输入格式"""
        base_features = [lm for hand in landmarks for lm in 
                        [hand.landmark[i].x for i in range(21)] +
                        [hand.landmark[i].y for i in range(21)]]
        geometry_features = self._calculate_geometry_features(landmarks[0].landmark)
        combined = base_features + geometry_features
        return self.scaler.transform([combined])

    def _calculate_geometry_features(self, landmarks):
        """计算几何特征（与preprocessor保持同步）"""
        wrist = landmarks[0]
        middle_base = landmarks[9]
        return [
            abs(wrist.x - middle_base.x),   # palm_width
            abs(wrist.y - middle_base.y),   # palm_height
            # 各手指平均角度计算（示例，需与训练时保持一致）
            0.0, 0.0, 0.0, 0.0, 0.0  # 此处应替换实际计算
        ]

    def _predict_gesture(self, frame):
        """核心检测流程"""
        results = self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not results.multi_hand_landmarks:
            return None, None

        # 获取主要手部信息
        hand_landmarks = results.multi_hand_landmarks[0]
        processed_data = self._preprocess_landmarks([hand_landmarks])
        
        # 模型预测
        pred = self.model.predict(processed_data)[0]
        return self.label_encoder.inverse_transform([pred])[0], hand_landmarks

    def _handle_system_control(self, gesture):
        """处理系统级控制手势"""
        # 激活/关闭控制模式
        if gesture == 'activate':
            self.activation_counter += 1
            if self.activation_counter >= self.ACTIVATION_THRESHOLD:
                self.control_active = not self.control_active
                print(f"手动控制 {'已启动' if self.control_active else '已关闭'}")
                self.activation_counter = 0
        else:
            self.activation_counter = 0

        # 退出手势处理
        if gesture == 'exit' and self.control_active:
            self.current_mode = None
            self.control_active = False
            print("完全退出控制模式")

    def _handle_mode_control(self, gesture):
        """处理模式切换"""
        if not self.control_active or time.time() - self.last_operation_time < 1:
            return

        if gesture in ['mode1', 'mode2', 'mode3', 'mode4']:
            self.gesture_buffer.append(gesture)
            
            # 检查持续识别到同一模式
            if len(self.gesture_buffer) >= self.MODE_HOLD_DURATION and \
                len(set(self.gesture_buffer)) == 1:
                self.current_mode = self.gesture_buffer[-1]
                print(f"切换到{self.current_mode}模式")
                self.last_operation_time = time.time()
                self.gesture_buffer.clear()

    def _handle_direction_control(self, gesture):
        """处理方向指令"""
        if self.current_mode and time.time() - self.last_operation_time > 0.5:
            controls = {
                'up': 1, 'down': 2, 'left': 3, 'right': 4
            }
            if output := controls.get(gesture):
                print(f"[{self.current_mode}] 执行操作: {output}")
                self.last_operation_time = time.time()

    def _update_display(self, frame, gesture, landmarks):
        """更新显示界面"""
        # 绘制关键点
        if landmarks:
            frame = draw_landmarks(frame, [
                (int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0]))
                for lm in landmarks.landmark
            ])
        
        # 状态信息排版
        status = []
        if self.control_active:
            status.append(f"控制模式: {self.current_mode or '待选择'}")
        else:
            status.append("等待激活...")
        
        return display_status(frame, status)

    def run(self):
        try:
            while True:
                frame = self.camera.get_frame()
                if frame is None: continue

                gesture, landmarks = self._predict_gesture(frame)
                frame = self._update_display(frame, gesture, landmarks)

                if gesture:
                    self._handle_system_control(gesture)
                    self._handle_mode_control(gesture)
                    self._handle_direction_control(gesture)

                cv2.imshow('Gesture Control', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            self.camera.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    GestureControlSystem().run()
