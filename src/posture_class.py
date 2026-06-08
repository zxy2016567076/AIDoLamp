import cv2
import mediapipe as mp
import numpy as np
from picamera2 import Picamera2

# ------------------------- 全局配置 -------------------------
CONFIG = {
    "camera_resolution": (320, 240),  # CSI摄像头支持的分辨率
    "min_detection_confidence": 0.7, # 检测置信度
    "min_tracking_confidence": 0.5, # 跟踪置信度
    "chin_shoulder_threshold": 0.05, # 下巴与肩膀的距离阈值
    "shoulder_tilt_threshold": 0.1, # 肩膀倾斜度阈值
    "head_forward_threshold": 25 # 头部前倾角度阈值
}

# ------------------------- 核心检测类 -------------------------
class PostureDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,  # 树莓派性能有限可改为0
            min_detection_confidence=CONFIG["min_detection_confidence"],
            min_tracking_confidence=CONFIG["min_tracking_confidence"]
        )
        self.mp_face = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5
        )
        self.landmark_index = mp.solutions.pose.PoseLandmark
        self._setup_visuals()

    def _setup_visuals(self):
        self.alert_style = {
            "font": cv2.FONT_HERSHEY_SIMPLEX,
            "color_good": (100, 255, 100),
            "color_bad": (100, 100, 255),
            "thickness": 2,
            "line_type": cv2.LINE_AA
        }

    def _get_landmark_coords(self, pose_landmarks, face_landmarks, img_shape):
        h, w = img_shape[:2]
        coords = {}
        
        if pose_landmarks:
            coords.update({
                "LEFT_SHOULDER": (int(pose_landmarks[self.landmark_index.LEFT_SHOULDER].x * w)),
                "RIGHT_SHOULDER": (int(pose_landmarks[self.landmark_index.RIGHT_SHOULDER].x * w)),
                "LEFT_SHOULDER_Y": int(pose_landmarks[self.landmark_index.LEFT_SHOULDER].y * h),
                "RIGHT_SHOULDER_Y": int(pose_landmarks[self.landmark_index.RIGHT_SHOULDER].y * h),
            })
        
        if face_landmarks:
            chin = face_landmarks.landmark[152]
            coords["CHIN"] = (int(chin.x * w), int(chin.y * h))
            
        return coords

    def analyze_posture(self, frame):
        alerts = []
        debug_info = []
        img_h, img_w = frame.shape[:2]
        
        pose_results = self.mp_pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        face_results = self.mp_face.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        if pose_results.pose_landmarks and face_results.multi_face_landmarks:
            coords = self._get_landmark_coords(
                pose_results.pose_landmarks.landmark,
                face_results.multi_face_landmarks[0],
                frame.shape
            )
            
            shoulder_mid_y = (coords["LEFT_SHOULDER_Y"] + coords["RIGHT_SHOULDER_Y"]) // 2
            chin = coords["CHIN"]
            
            head_vertical = abs(chin[1] - shoulder_mid_y) / img_h
            is_chin_lower = chin[1] > shoulder_mid_y
            
            debug_info.append(f"Chin-Shoulder: {head_vertical:.3f}")
            debug_info.append(f"Chin Lower: {'Yes' if is_chin_lower else 'No'}")

            if head_vertical < CONFIG["chin_shoulder_threshold"] or is_chin_lower:
                alerts.append("Head Too Close")
                self._draw_distance_line(frame, chin, (chin[0], shoulder_mid_y))
                
            shoulder_tilt = abs(coords["LEFT_SHOULDER_Y"] - coords["RIGHT_SHOULDER_Y"]) / img_h
            debug_info.append(f"Shoulder Tilt: {shoulder_tilt:.3f}")
            
            if shoulder_tilt > CONFIG["shoulder_tilt_threshold"]:
                alerts.append("Shoulder Tilt")
                self._draw_shoulder_line(
                    frame, 
                    (coords["LEFT_SHOULDER"], coords["LEFT_SHOULDER_Y"]),
                    (coords["RIGHT_SHOULDER"], coords["RIGHT_SHOULDER_Y"])
                )

        return alerts, debug_info, frame

    def _draw_distance_line(self, frame, start, end):
        cv2.line(frame, start, end, self.alert_style["color_bad"], 2)
        cv2.circle(frame, start, 5, self.alert_style["color_bad"], -1)
        
    def _draw_shoulder_line(self, frame, left_shoulder, right_shoulder):
        cv2.line(frame, 
                (left_shoulder[0], left_shoulder[1]),
                (right_shoulder[0], right_shoulder[1]),
                self.alert_style["color_bad"], 2)

# ------------------------- 主程序 -------------------------
def main():
    # 初始化CSI摄像头
    picam2 = Picamera2()
    config = picam2.create_video_configuration(
        main={"size": CONFIG["camera_resolution"]},
    )
    picam2.configure(config)
    picam2.start()

    detector = PostureDetector()

    while True:
        frame = picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # 转换颜色空间
        frame = cv2.flip(frame, 1)  # 镜像翻转

        # 执行检测
        alerts, debug_info, processed_frame = detector.analyze_posture(frame)

        # 显示界面
        cv2.putText(processed_frame, "Posture Monitor - Raspberry Pi", (20, 40),
                   detector.alert_style["font"], 1.1, (200, 200, 250), 2)
        
        status_y = 80
        for alert in alerts:
            cv2.putText(processed_frame, f"! {alert}", (30, status_y),
                       detector.alert_style["font"], 0.9, 
                       detector.alert_style["color_bad"], 2)
            status_y += 40
            
        debug_y = processed_frame.shape[0] - 50
        for info in debug_info:
            cv2.putText(processed_frame, info, (30, debug_y),
                       detector.alert_style["font"], 0.6, (200, 200, 200), 1)
            debug_y -= 25
        
        cv2.imshow('Smart Posture Monitor', processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    picam2.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()