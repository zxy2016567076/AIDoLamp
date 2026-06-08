import cv2  # OpenCV库，用于图像处理
import joblib  # 用于加载机器学习模型
import numpy as np  # 数值计算库
import mediapipe as mp  # Mediapipe库，用于手势和姿势检测
import time  # 时间相关操作
import os  # 文件和路径操作
from picamera2 import Picamera2  # 用于控制树莓派摄像头
from libcamera import Transform  # 用于摄像头图像变换
import threading  # 新增此行
from serial_comm import SerialCommunicator
from voice_class import VoiceAssistant

# ------------------------- 全局配置 -------------------------
CONFIG = {
    "camera_resolution": (320, 240),  # 摄像头分辨率
    "chin_shoulder_threshold": 0.05,  # 下巴与肩膀距离的阈值
    "shoulder_tilt_threshold": 0.1,  # 肩膀倾斜的阈值
    "head_forward_threshold": 25  # 头部前倾的阈值
}

# ------------------------- 手势识别类 -------------------------
class GestureRecognizer:
    def __init__(self):
        # 加载手势识别模型
        self.model = joblib.load(os.path.join('gesture_model.pkl'))
        # 初始化Mediapipe的手部检测模块
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,  # 是否使用静态图像模式
            max_num_hands=1,  # 最大检测手数
            min_detection_confidence=0.7,  # 最小检测置信度
            min_tracking_confidence=0.5  # 最小跟踪置信度
        )
        self.activated = False  # 系统是否激活
        self.activation_start_time = 0  # 激活开始时间
        self.current_mode = None  # 当前模式
        self.last_action_time = 0  # 上次动作时间
        self.gesture_timers = {}  # 手势计时器
        self.confirmed_label = None  # 确认的手势标签
        self.state = "standby"          # 新增：系统状态
        self.should_exit = False        # 新增：退出标志位
        self.serial_comm = SerialCommunicator() # 串口通信类

    def process_frame(self, frame):
        # 将帧从BGR转换为RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 使用Mediapipe处理帧，检测手部
        results = self.hands.process(frame_rgb)
        if not results.multi_hand_landmarks:  # 如果没有检测到手部
            return None
        
        # 提取手部关键点
        hand_landmarks = results.multi_hand_landmarks[0]
        keypoints = []
        for lm in hand_landmarks.landmark:
            keypoints.extend([lm.x, lm.y, lm.z])  # 将每个关键点的x, y, z坐标添加到列表
        
        # 如果关键点数量为63（21个关键点，每个3个坐标），则进行预测
        return self.model.predict([keypoints])[0] if len(keypoints) == 63 else None

    def check_gesture_duration(self, current_label):
        # 根据手势标签设置所需持续时间
        required_time = 3 if current_label == 1 else 1
        if current_label not in self.gesture_timers:  # 如果手势计时器中没有当前手势
            self.gesture_timers[current_label] = time.time()  # 初始化计时器
        # 检查手势持续时间是否达到要求
        return time.time() - self.gesture_timers[current_label] >= required_time

    def handle_mode_operations(self, label, mode):
        """处理模式内的操作逻辑（根据模式区分标签行为）"""
        if self.check_gesture_duration(label):
            # 退出逻辑（所有模式通用）
            if label == 3:
                print(f"退出模式{mode}")
                self.current_mode = None
            else:
                # 模式1：模式转换
                if mode == 1:
                    if label == 5:
                        print("进入待机状态")
                        self.state = "standby"
                        self.serial_comm.send_mode(1) # 发送模式1指令
                        self.should_exit = True  # 设置退出标志
                        
                    elif label == 6:                          
                        print("进入普通状态")
                        self.state = "normal"
                        self.serial_comm.send_mode(2) # 发送模式2指令
                        self.should_exit = True
                        
                    elif label == 7:  
                        print("进入互动状态")
                        self.state = "interactive"
                        self.serial_comm.send_mode(3) # 发送模式3指令
                        self.should_exit = True
                        
                    elif label == 8:                           
                        print("进入工作状态")
                        self.state = "work"
                        self.serial_comm.send_mode(4) # 发送模式4指令
                        self.should_exit = True
                
                
                # 模式2：光强和色温调节
                elif mode == 2:
                    if label == 10: # 上
                        
                        print("增强光照强度")
                        
                    elif label == 2:# 下
                        
                        print("减弱光照强度")
                        
                    elif label == 9:# 右
                        
                        print("调节色温")
                        
                    elif label == 4:# 左
                        
                        print("调节色温")
                        
                
                # 模式3：调节灯的方向
                elif mode == 3:
                    if label == 10:
                        
                        print("灯罩向上")
                        
                    elif label == 2:
                        
                        print("灯罩向下")
                        
                    elif label == 9:
                        
                        print("灯身向右")
                        
                    elif label == 4:
                        
                        print("灯身向左")
                        
                
                # 模式4：调节灯的高度
                elif mode == 4:
                    if label == 10:
                        
                        print("中间舵机向上")
                        
                    elif label == 2:
                        
                        print("中间舵机向下")
                        
                    elif label == 9:
                        
                        print("下面舵机向前")
                        
                    elif label == 4:
                        
                        print("下面舵机向后")
                        
            
            # 重置计时器和操作时间
            self.last_action_time = time.time()
            self.gesture_timers.clear()

# ------------------------- 坐姿检测类 -------------------------
class PostureDetector:
    def __init__(self):
        # 初始化Mediapipe的姿势检测模块
        self.mp_pose = mp.solutions.pose.Pose(
            static_image_mode=False,  # 是否使用静态图像模式
            model_complexity=1,  # 模型复杂度
            min_detection_confidence=0.7,  # 最小检测置信度
            min_tracking_confidence=0.5  # 最小跟踪置信度
        )
        # 初始化Mediapipe的面部检测模块
        self.mp_face = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,  # 是否使用静态图像模式
            max_num_faces=1,  # 最大检测人脸数
            min_detection_confidence=0.5  # 最小检测置信度
        )
        self.landmark_index = mp.solutions.pose.PoseLandmark  # 姿势关键点索引

    def analyze_posture(self, frame):
        alerts = []  # 用于存储警告信息
        # 使用Mediapipe处理帧，检测姿势和面部
        pose_results = self.mp_pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        face_results = self.mp_face.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        if pose_results.pose_landmarks and face_results.multi_face_landmarks:  # 如果检测到姿势和面部
            h, w = frame.shape[:2]  # 获取帧的高度和宽度
            pose_landmarks = pose_results.pose_landmarks.landmark  # 姿势关键点
            chin = face_results.multi_face_landmarks[0].landmark[152]  # 下巴关键点
            
            # 头部检测
            left_shoulder_y = int(pose_landmarks[self.landmark_index.LEFT_SHOULDER].y * h)
            right_shoulder_y = int(pose_landmarks[self.landmark_index.RIGHT_SHOULDER].y * h)
            shoulder_mid_y = (left_shoulder_y + right_shoulder_y) // 2  # 计算肩膀中点的y坐标
            chin_y = int(chin.y * h)  # 下巴的y坐标
            
            # 检查下巴与肩膀的距离是否过近
            if abs(chin_y - shoulder_mid_y)/h < CONFIG["chin_shoulder_threshold"] or chin_y > shoulder_mid_y:
                alerts.append("Head Too Close")
                
                
            # 检查肩膀是否倾斜
            if abs(left_shoulder_y - right_shoulder_y)/h > CONFIG["shoulder_tilt_threshold"]:
                alerts.append("Shoulder Tilt")
                
        return alerts  # 返回警告信息

# ------------------------- 主程序 -------------------------
class CombinedSystem:
    def __init__(self, shared_camera=None):
        # 初始化摄像头
        self.camera_owned = shared_camera is None  # 是否由本类创建摄像头
        
        if shared_camera is None:
            # 自己创建摄像头
            try:
                self.picam2 = Picamera2(1)
                config = self.picam2.create_video_configuration(
                    main={"size": CONFIG["camera_resolution"]},  # 设置摄像头分辨率
                    transform=Transform(hflip=True))  # 水平翻转图像
                self.picam2.configure(config)  # 配置摄像头
                self.picam2.start()  # 启动摄像头
                print("CombinedSystem创建并启动了自己的摄像头")
            except Exception as e:
                print(f"CombinedSystem摄像头初始化错误: {e}")
                self.picam2 = None
        else:
            # 使用共享摄像头
            self.picam2 = shared_camera
            print("CombinedSystem使用共享摄像头")
        
        # 初始化手势识别和坐姿检测模块
        self.gesture_recognizer = GestureRecognizer()
        self.posture_detector = PostureDetector()
        self.prev_time = time.time()  # 上一帧的时间
        
        # 初始化状态变量
        self.posture_alerts = []  # 新增：存储坐姿警报
        self.lock = threading.Lock()  # 新增：线程锁

    def run(self):
        final_state = "standby"  # 初始化最终状态
        while True:
            # 获取帧并转换颜色空间
            frame = self.picam2.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # 计算FPS
            current_time = time.time()
            fps = 1 / (current_time - self.prev_time)
            self.prev_time = current_time
            try:
                # 并行处理检测任务
                gesture_label = self.gesture_recognizer.process_frame(frame)
                posture_alerts = self.posture_detector.analyze_posture(frame)
                
                # 带锁更新姿势警报
                with self.lock:
                    self.posture_alerts = posture_alerts.copy()  # 线程安全更新
                
                # 更新手势状态机
                self._update_gesture_state(gesture_label)
                
                # 显示UI
                self._draw_ui(frame, fps, gesture_label, posture_alerts)
                
                # 显示图像窗口
                cv2.imshow('Combined System', frame)
                if cv2.waitKey(1) in [27, ord('q')]:  # 按下ESC或q退出
                    break
                
                # 退出检测
                if cv2.waitKey(1) in [27, ord('q')] or self.gesture_recognizer.should_exit:
                    final_state = self.gesture_recognizer.state
                    break
                    
            except Exception as e:
                print(f"检测流程异常: {e}")
                break

        # 只有当摄像头是由当前类创建的时候才关闭它
        if self.camera_owned and self.picam2 is not None:
            try:
                if hasattr(self.picam2, 'is_running') and self.picam2.is_running:
                    self.picam2.stop()
                    print("CombinedSystem停止了自己的摄像头")
            except Exception as e:
                print(f"CombinedSystem停止摄像头错误: {e}")
                
        cv2.destroyAllWindows()  # 销毁所有窗口
        return final_state  # 返回最终状态

    def _update_gesture_state(self, label):
        if label is None: return
        
        # 清除非当前手势计时器
        for l in list(self.gesture_recognizer.gesture_timers.keys()):
            if l != label:
                del self.gesture_recognizer.gesture_timers[l]

        if not self.gesture_recognizer.activated:
            if label == 1 and self.gesture_recognizer.check_gesture_duration(label):
                self.gesture_recognizer.activated = True
                self.gesture_recognizer.gesture_timers.clear()
        else:
            if label == 3 and self.gesture_recognizer.check_gesture_duration(label):
                self.gesture_recognizer.activated = False
                self.gesture_recognizer.current_mode = None
            elif label in [5,6,7,8] and not self.gesture_recognizer.current_mode:
                if self.gesture_recognizer.check_gesture_duration(label):
                    self.gesture_recognizer.current_mode = label - 4
            elif self.gesture_recognizer.current_mode:
                # 修改调用方式，传入当前模式参数
                self.gesture_recognizer.handle_mode_operations(label, self.gesture_recognizer.current_mode)

    def _draw_ui(self, frame, fps, gesture_label, posture_alerts):
        # 显示FPS
        cv2.putText(frame, f"FPS: {int(fps)}", (frame.shape[1]-120, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        
        # 显示手势状态
        status_text = []
        if gesture_label is not None:
            status_text.append(f"Gesture: {gesture_label}")
        if self.gesture_recognizer.activated:
            status_text.append("Activated")
        if self.gesture_recognizer.current_mode:
            status_text.append(f"Mode {self.gesture_recognizer.current_mode}")
        
        # 显示坐姿警告
        if posture_alerts:
            status_text.extend([f"! {alert}" for alert in posture_alerts])
        
        # 绘制所有状态信息
        y_pos = 40
        for text in status_text:
            color = (0,255,0) if not text.startswith("!") else (0,0,255)  # 警告信息用红色
            cv2.putText(frame, text, (20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
            y_pos += 25

if __name__ == "__main__":
    system = CombinedSystem()  # 初始化系统
    final_state = system.run()  # 运行系统
    print("系统最终状态:", final_state)
