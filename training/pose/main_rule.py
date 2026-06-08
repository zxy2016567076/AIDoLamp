import cv2
import mediapipe as mp
import numpy as np

# ------------------------- 全局配置 -------------------------
CONFIG = {
    "camera_id": 0,                  # 摄像头设备ID
    "camera_resolution": (1280, 720),# 分辨率
    "pose_complexity": 1,            # 模型复杂度
    "min_detection_confidence": 0.8, # 检测置信度（调高以提高准确性）
    "min_tracking_confidence": 0.5,  # 跟踪置信度
    
    # 姿势判断阈值
    "head_angle_threshold": 75,       # 头部前倾角度阈值（度） 
    "head_vertical_threshold": 0.15,  # 耳肩垂直距离阈值（基于图像高度）
    "spine_angle_threshold": 28,      # 脊柱弯曲角度
    "shoulder_tilt_threshold": 0.12,  # 肩部倾斜比例
    "ear_shoulder_distance": 0.22    # 颈部弯曲阈值
}

# ------------------------- 核心检测类 -------------------------
class PostureDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=CONFIG["pose_complexity"],
            min_detection_confidence=CONFIG["min_detection_confidence"],
            min_tracking_confidence=CONFIG["min_tracking_confidence"]
        )
        
        self.landmark_index = mp.solutions.pose.PoseLandmark
        self._setup_visuals()

    def _setup_visuals(self):
        """可视化参数"""
        self.alert_style = {
            "font": cv2.FONT_HERSHEY_SIMPLEX,
            "color_good": (100, 255, 100),  # 绿色
            "color_bad": (100, 100, 255),   # 红色
            "thickness": 2,
            "line_type": cv2.LINE_AA
        }

    def _get_landmark_coords(self, landmarks, img_shape):
        """转换坐标"""
        h, w = img_shape[:2]
        return {
            "LEFT_SHOULDER": (int(landmarks[self.landmark_index.LEFT_SHOULDER].x * w),
                            int(landmarks[self.landmark_index.LEFT_SHOULDER].y * h)),
            "RIGHT_SHOULDER": (int(landmarks[self.landmark_index.RIGHT_SHOULDER].x * w),
                             int(landmarks[self.landmark_index.RIGHT_SHOULDER].y * h)),
            "LEFT_HIP": (int(landmarks[self.landmark_index.LEFT_HIP].x * w),
                        int(landmarks[self.landmark_index.LEFT_HIP].y * h)),
            "RIGHT_HIP": (int(landmarks[self.landmark_index.RIGHT_HIP].x * w),
                         int(landmarks[self.landmark_index.RIGHT_HIP].y * h)),
            "LEFT_EAR": (int(landmarks[self.landmark_index.LEFT_EAR].x * w),
                        int(landmarks[self.landmark_index.LEFT_EAR].y * h)),
            "RIGHT_EAR": (int(landmarks[self.landmark_index.RIGHT_EAR].x * w),
                         int(landmarks[self.landmark_index.RIGHT_EAR].y * h))
        }

    def analyze_posture(self, frame):
        """姿势分析主函数"""
        alerts = []
        debug_info = []
        img_h, img_w = frame.shape[:2]
        
        results = self.mp_pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        if results.pose_landmarks:
            coords = self._get_landmark_coords(results.pose_landmarks.landmark, frame.shape)
            
            # 计算身体中线关键点
            shoulder_mid = (
                (coords["LEFT_SHOULDER"][0] + coords["RIGHT_SHOULDER"][0]) // 2,
                (coords["LEFT_SHOULDER"][1] + coords["RIGHT_SHOULDER"][1]) // 2
            )
            hip_mid = (
                (coords["LEFT_HIP"][0] + coords["RIGHT_HIP"][0]) // 2,
                (coords["LEFT_HIP"][1] + coords["RIGHT_HIP"][1]) // 2
            )
            ear_mid = (
                (coords["LEFT_EAR"][0] + coords["RIGHT_EAR"][0]) // 2,
                (coords["LEFT_EAR"][1] + coords["RIGHT_EAR"][1]) // 2
            )

            # ------------------------- 头部检测（优化部分） -------------------------
            # 检测方法1：耳肩垂直距离
            head_vertical_ratio = abs(ear_mid[1] - shoulder_mid[1]) / img_h
            debug_info.append(f"Vertical: {head_vertical_ratio:.2f}")
            
            # 检测方法2：头部前倾角度
            dx = ear_mid[0] - shoulder_mid[0]
            dy = ear_mid[1] - shoulder_mid[1]
            head_angle = np.degrees(np.arctan2(dy, dx))  # 计算相对于垂直轴的角度
            debug_info.append(f"Angle: {head_angle:.1f}°")
            
            # 综合判断
            if head_vertical_ratio > CONFIG["head_vertical_threshold"] and head_angle < CONFIG["head_angle_threshold"]:
                alerts.append("Head Forward")
                self._draw_head_alert(frame, ear_mid, shoulder_mid)

            # ------------------------- 其他检测 -------------------------
            # 脊柱弯曲检测
            spine_vector = (hip_mid[0] - shoulder_mid[0], hip_mid[1] - shoulder_mid[1])
            spine_angle = np.degrees(np.arctan2(abs(spine_vector[0]), spine_vector[1]))
            if spine_angle > CONFIG["spine_angle_threshold"]:
                alerts.append(f"Spine Bend: {spine_angle:.1f}°")
                self._draw_spine_line(frame, shoulder_mid, hip_mid)

            # 肩部倾斜检测
            shoulder_tilt = abs(coords["LEFT_SHOULDER"][1] - coords["RIGHT_SHOULDER"][1]) / img_h
            if shoulder_tilt > CONFIG["shoulder_tilt_threshold"]:
                alerts.append("Shoulder Tilt")
                self._draw_shoulder_line(frame, coords)

            # 颈部弯曲检测
            neck_bend = abs(ear_mid[1] - shoulder_mid[1]) / img_h
            if neck_bend > CONFIG["ear_shoulder_distance"]:
                alerts.append("Neck Bend")
                self._draw_neck_line(frame, ear_mid, shoulder_mid)

        return alerts, debug_info, frame

    # ------------------------- 可视化方法 -------------------------
    def _draw_spine_line(self, frame, shoulder, hip):
        cv2.line(frame, shoulder, hip, self.alert_style["color_bad"], 2)
        
    def _draw_head_alert(self, frame, ear, shoulder):
        # 绘制垂直线条提示
        cv2.arrowedLine(frame, ear, (ear[0], shoulder[1]), 
                       self.alert_style["color_bad"], 2, tipLength=0.3)
        
    def _draw_shoulder_line(self, frame, coords):
        cv2.line(frame, coords["LEFT_SHOULDER"], coords["RIGHT_SHOULDER"],
                self.alert_style["color_bad"], 2)
        
    def _draw_neck_line(self, frame, ear, shoulder):
        cv2.line(frame, ear, (ear[0], shoulder[1]), 
                self.alert_style["color_bad"], 2)

# ------------------------- 主程序 -------------------------
def main():
    cap = cv2.VideoCapture(CONFIG["camera_id"])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG["camera_resolution"][0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG["camera_resolution"][1])
    
    detector = PostureDetector()
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        
        frame = cv2.flip(frame, 1)  # 镜像显示
        
        # 执行检测
        alerts, debug_info, processed_frame = detector.analyze_posture(frame)
        
        # ------------------------- 显示界面 -------------------------
        # 标题
        status_y = 40
        cv2.putText(processed_frame, "Posture Monitor", (20, status_y),
                   detector.alert_style["font"], 1.2, (200, 200, 250), 2)
        
        # 显示警报（红色）
        for alert in alerts:
            status_y += 40
            cv2.putText(processed_frame, f"! {alert}", (30, status_y),
                       detector.alert_style["font"], 0.9, 
                       detector.alert_style["color_bad"], 2)
        
        # 显示调试信息（白色）
        debug_y = processed_frame.shape[0] - 50
        for info in debug_info:
            cv2.putText(processed_frame, info, (30, debug_y),
                       detector.alert_style["font"], 0.7, 
                       (255, 255, 255), 1)
            debug_y -= 30
        
        cv2.imshow('Smart Posture Monitor', processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
