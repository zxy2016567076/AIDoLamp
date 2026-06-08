"""
坐姿检测系统稳定版（修复所有警告+优化性能）
版本：v2.1.1
最后更新：2024年5月
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 完全屏蔽TensorFlow日志
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import logging
import absl.logging

# ------------------------- 日志配置 -------------------------
absl.logging.set_verbosity(absl.logging.ERROR)  # 阻断ABSL日志
logging.basicConfig(level=logging.ERROR)
mp_logger = logging.getLogger('mediapipe')
mp_logger.setLevel(logging.ERROR)

# ------------------------- 全局配置 -------------------------
CONFIG = {
    # 硬件相关
    "camera_id": 0,                      # 摄像头设备ID
    "camera_resolution": (1280, 720),    # 推荐分辨率
    
    # 模型参数
    "pose_complexity": 0,               # 模型复杂度（0-2，建议0减少警告）
    "min_detection_confidence": 0.7,    # 初始检测置信度
    "min_tracking_confidence": 0.5,     # 持续跟踪置信度
    
    # 算法参数  
    "window_size": 30,                  # LSTM时间窗口
    "smoothing_window": 5,              # 预测结果平滑
    
    # 界面显示
    "alert_text": ("NORMAL", "BAD POSTURE"),
    "alert_color": ((86, 228, 129), (57, 64, 234)),
    "info_color": (255, 255, 255)
}

# ------------------------- 核心算法类 -------------------------
class PoseAnalyzer:
    LANDMARK_INDEX = {
        'LEFT_SHOULDER': 11,
        'RIGHT_SHOULDER': 12,
        'LEFT_HIP': 23,
        'RIGHT_HIP': 24,
        'LEFT_EAR': 7,
        'RIGHT_EAR': 8
    }
    
    def __init__(self):
        self._pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=CONFIG["pose_complexity"],
            min_detection_confidence=CONFIG["min_detection_confidence"],
            min_tracking_confidence=CONFIG["min_tracking_confidence"]
        )
        self._features_buffer = []
        self._prediction_buffer = []
        
    def process_frame(self, frame):
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_size = (frame.shape[1], frame.shape[0])  # 获取实际分辨率
        
        # 显式传递图像尺寸以消除投影警告
        results = self._pose.process(image_rgb, image_size=frame_size)
        
        if results.pose_landmarks:
            features = self._extract_features(
                results.pose_landmarks.landmark, 
                frame_size
            )
            if features is not None:
                self._update_buffer(features)
                
                if len(self._features_buffer) == CONFIG["window_size"]:
                    prediction = self._predict_posture()
                    return self._smooth_prediction(prediction)
        return None
    
    def _extract_features(self, landmarks, image_size):
        idx = self.LANDMARK_INDEX
        
        try:
            # 向量化坐标获取（提升性能）
            get_xy = lambda i: (landmarks[i].x * image_size[0], 
                              landmarks[i].y * image_size[1])
            
            shoulder_l = get_xy(idx['LEFT_SHOULDER'])
            shoulder_r = get_xy(idx['RIGHT_SHOULDER'])
            hip_l = get_xy(idx['LEFT_HIP'])
            hip_r = get_xy(idx['RIGHT_HIP'])
            ear_l = get_xy(idx['LEFT_EAR'])
            ear_r = get_xy(idx['RIGHT_EAR'])
            
            # 脊柱角度计算
            shoulder_center = np.mean([shoulder_l, shoulder_r], axis=0)
            hip_center = np.mean([hip_l, hip_r], axis=0)
            spine_vector = hip_center - shoulder_center
            spine_angle = np.degrees(
                np.arccos(spine_vector[1] / np.linalg.norm(spine_vector))
            )
            
            # 头部前倾距离
            ear_center = np.mean([ear_l, ear_r], axis=0)
            head_forward = ear_center[0] - shoulder_center[0]
            
            # 肩部倾斜度
            shoulder_tilt = shoulder_r[1] - shoulder_l[1]
            
            return np.array([spine_angle, head_forward, shoulder_tilt])
        
        except (IndexError, AttributeError) as e:
            logging.error(f"特征提取失败: {str(e)}")
            return None
    
    def _update_buffer(self, features):
        self._features_buffer.append(features)
        if len(self._features_buffer) > CONFIG["window_size"]:
            self._features_buffer.pop(0)
            
    def _predict_posture(self):
        model = load_model('posture_lstm.h5')
        input_data = np.array(self._features_buffer).reshape(1, CONFIG["window_size"], -1)
        return model.predict(input_data, verbose=0)[0]
    
    def _smooth_prediction(self, prediction):
        self._prediction_buffer.append(np.argmax(prediction))
        if len(self._prediction_buffer) > CONFIG["smoothing_window"]:
            self._prediction_buffer.pop(0)
        return np.round(np.mean(self._prediction_buffer)).astype(int)

# ------------------------- 主控程序 -------------------------
class PostureMonitor:
    def __init__(self):
        self.cap = self._init_camera()
        self.analyzer = PoseAnalyzer()
        self.frame_count = 0
        self.last_fps = 0.0
        
    def _init_camera(self):
        cap = cv2.VideoCapture(CONFIG["camera_id"])
        if not cap.isOpened():
            raise RuntimeError("摄像头初始化失败，请检查："
                              f"\n- 设备ID是否正确（当前：{CONFIG['camera_id']})"
                              "\n- 摄像头权限设置"
                              "\n- 硬件连接状态")
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG["camera_resolution"][0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG["camera_resolution"][1])
        return cap
    
    def run(self):
        last_time = cv2.getTickCount()
        
        while True:
            success, frame = self.cap.read()
            if not success:
                logging.error("视频流中断，请检查摄像头")
                break
            
            # 姿态分析
            posture_state = self.analyzer.process_frame(frame)
            
            # 更新FPS
            self.frame_count += 1
            current_time = cv2.getTickCount()
            if (current_time - last_time) / cv2.getTickFrequency() > 0.5:
                self.last_fps = self.frame_count / ((current_time - last_time)/cv2.getTickFrequency())
                last_time = current_time
                self.frame_count = 0
            
            # 界面显示
            self._draw_interface(frame, posture_state)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cap.release()
        cv2.destroyAllWindows()
    
    def _draw_interface(self, frame, posture_state):
        # 主检测结果
        if posture_state in [0, 1]:
            text = CONFIG["alert_text"][posture_state]
            color = CONFIG["alert_color"][posture_state]
            cv2.putText(frame, text, (20, 60), 
                       cv2.FONT_HERSHEY_DUPLEX, 1.5, color, 2, cv2.LINE_AA)
            
        # 信息面板
        status = {
            "FPS": f"{self.last_fps:.1f}",
            "Resolution": f"{frame.shape[1]}x{frame.shape[0]}",
            "Tracking": 'Working' if posture_state is not None else 'Not found'
        }
        
        y_pos = 100
        for k, v in status.items():
            cv2.putText(frame, f"{k}: {v}", (20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                       CONFIG["info_color"], 1, cv2.LINE_AA)
            y_pos += 30
            
        cv2.imshow("PostureGuard Monitor", frame)

if __name__ == "__main__":
    monitor = PostureMonitor()
    monitor.run()
