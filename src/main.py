import cv2
import time
import os
import numpy as np
from picamera2 import Picamera2
from libcamera import Transform  # 用于摄像头图像变换
import threading

# 导入自定义模块
from gesture_class import GestureRecognizer
from posture_class import PostureDetector
from csi1 import PostureDetector, CombinedSystem, GestureRecognizer
from serial_comm import SerialCommunicator
from voice_class import VoiceAssistant

# 导入新的YOLO和ObjectTracker模块
from yolo_class import YOLODetector
from object_tracking import ObjectTracker


class SmartLampSystem:
    def __init__(self):
        # 硬件初始化
        self.serial_comm = SerialCommunicator()  # 串口通信类
        self.serial_comm.set_mode_change_callback(self.switch_to_normal)  # 设置串口回调

        # 视觉模块初始化为None（按需创建）
        self.cam1 = None  # 底座摄像头（编号1，用于手势识别和坐姿检测）
        self.cam2 = None  # 顶部摄像头（编号0，用于YOLO检测）

        # 算法模块
        self.gesture = GestureRecognizer()  # 手势识别
        self.posture = PostureDetector()  # 坐姿检测

        # 创建一个简单的IR传感器模拟
        self.ir_sensor = type("IRSensor", (), {"get_distance": lambda self: 0})()

        # 创建一个简单的LCD显示模拟
        self.lcd = type(
            "LCDDisplay", (), {"update_display": lambda self, state, distance: None}
        )()

        # 新的YOLO和ObjectTracker实例
        # 先不传递camera_instance，稍后在使用时再设置
        self.yolo_detector = YOLODetector(
            camera_instance=None,  # 暂不共享摄像头
            model_path="best.pt",
        )
        self.object_tracker = ObjectTracker()

        # 配置语音助手（API 密钥从环境变量读取，参考 .env.example）
        voice_config = {
            "BAIDU_APP_ID": os.environ.get("BAIDU_APP_ID", ""),
            "BAIDU_API_KEY": os.environ.get("BAIDU_API_KEY", ""),
            "BAIDU_SECRET_KEY": os.environ.get("BAIDU_SECRET_KEY", ""),
            "DEEPSEEK_API_KEY": os.environ.get("DEEPSEEK_API_KEY", ""),
            "WEATHER_API_KEY": os.environ.get("WEATHER_API_KEY", ""),
        }
        self.voice = VoiceAssistant(voice_config)  # 语音助手

        # 状态管理
        self.state = "standby"  # 待机状态
        self.last_state_change = time.time()  # 上次状态切换时间
        self.user_distance = 0  # 用户距离（测距模块）

        # 坐姿检测语音提醒间隔时间
        self.last_posture_alert = 0  # 最后坐姿提醒时间
        self.alert_interval = 8  # 提醒间隔(秒)

        # 确保csi_system初始化，避免后续引用错误
        self.csi_system = None

    def switch_to_normal(self):
        """串口回调函数"""
        print("检测到距离变化≥5cm，触发模式切换")
        self.state = "normal"

    def _setup_cameras(self):
        """按需初始化摄像头"""
        try:
            # 先检查摄像头是否已经初始化
            if self.cam1 is None:
                self.cam1 = Picamera2(1)
                cam1_config = self.cam1.create_video_configuration(
                    main={"size": (320, 240)}, transform=Transform(hflip=True)
                )
                self.cam1.configure(cam1_config)
                print("cam1 (摄像头1) 初始化成功")

            if self.cam2 is None:
                self.cam2 = Picamera2(0)
                cam2_config = self.cam2.create_video_configuration(
                    main={"size": (320, 240)}
                )
                self.cam2.configure(cam2_config)
                print("cam2 (摄像头0) 初始化成功")

            # 更新YOLO检测器的摄像头实例
            self.yolo_detector.picam2 = self.cam2
        except Exception as e:
            print(f"摄像头初始化错误: {e}")

    def _start_cameras(self):
        """启动摄像头"""
        # 先确保摄像头已初始化
        self._setup_cameras()

        try:
            # 安全启动摄像头1
            if (
                self.cam1 is not None
                and hasattr(self.cam1, "is_running")
                and not self.cam1.is_running
            ):
                self.cam1.start()
                print("摄像头1已启动")

            # 安全启动摄像头2
            if (
                self.cam2 is not None
                and hasattr(self.cam2, "is_running")
                and not self.cam2.is_running
            ):
                self.cam2.start()
                print("摄像头2已启动")
        except Exception as e:
            print(f"启动摄像头错误: {e}")

    def _stop_cameras(self):
        """停止摄像头"""
        try:
            # 安全停止摄像头1
            if (
                self.cam1 is not None
                and hasattr(self.cam1, "is_running")
                and self.cam1.is_running
            ):
                self.cam1.stop()
                print("摄像头1已停止")

            # 安全停止摄像头2
            if (
                self.cam2 is not None
                and hasattr(self.cam2, "is_running")
                and self.cam2.is_running
            ):
                self.cam2.stop()
                print("摄像头2已停止")
        except Exception as e:
            print(f"停止摄像头错误: {e}")

    def state_machine(self):
        try:
            while True:
                if self.state != "standby":
                    self._setup_cameras()  # 进入活跃状态时初始化
                    self._start_cameras()
                else:
                    self._stop_cameras()
                # 更新环境数据
                self.user_distance = self.ir_sensor.get_distance()

                if self.state == "standby":  # 如果当前状态是待机状态
                    self._standby_state()  # 调用待机状态处理函数
                elif self.state == "normal":  # 如果当前状态是普通模式
                    self._normal_state()  # 调用普通模式处理函数
                elif self.state == "interactive":  # 如果当前状态是互动模式
                    self._interactive_state()  # 调用互动模式处理函数
                elif self.state == "work":  # 如果当前状态是工作模式
                    self._work_state()  # 调用工作模式处理函数

                # 通过串口更新LCD显示
                self.lcd.update_display(
                    self.state, self.user_distance
                )  # 在LCD上显示当前状态和用户距离
        finally:
            # 确保摄像头资源被释放
            self._stop_cameras()

    def _standby_state(self):
        """待机状态处理"""
        self._stop_cameras()  # 停止摄像头
        self.serial_comm.start_mode_one()  # 启动串口模式一（持续监测距离）

    def _normal_state(self):
        """普通模式处理"""
        self._start_cameras()  # 启动摄像头
        self.serial_comm.stop_mode_one()  # 停止串口模式一
        self.voice.voice_feedback("欢迎主人回家已经进入普通模式")  # 可能会堵塞程序运行
        # 传递摄像头实例给手势识别模块
        self.state = self.gesture.real_time_recognition(
            shared_cam=self.cam1
        )  # 实时手势检测

    def _interactive_state(self):
        """互动模式主函数（多线程版本）"""

        # 确保摄像头已初始化并启动
        self._start_cameras()  # 这个方法会安全地启动摄像头，不需要重复调用start

        # ----------------- 共享变量定义 -----------------
        self.gesture_state = "interactive"  # 初始状态强制设置为互动模式
        self.yolo_results = None  # 存储YOLO检测结果
        self.voice_activated = False  # 语音激活标志
        self.stop_threads = False  # 线程停止标志
        self.lock = threading.Lock()  # 线程锁保证共享变量安全

        # ----------------- 线程定义 -----------------
        def gesture_monitor():
            """手势检测线程（直接调用原有逻辑）"""
            try:
                # 调用原有手势识别主函数（会更新self.gesture.state）
                # 传递共享摄像头避免资源冲突
                current_state = self.gesture.real_time_recognition(shared_cam=self.cam1)
                with self.lock:
                    self.gesture_state = current_state  # 同步最新状态
            except Exception as e:
                print(f"手势线程异常: {e}")

        def yolo_tracker():
            """人脸跟踪线程"""
            while not self.stop_threads:
                try:
                    # 确保摄像头存在并正在运行
                    if (
                        self.cam2 is None
                        or not hasattr(self.cam2, "is_running")
                        or not self.cam2.is_running
                    ):
                        time.sleep(0.1)  # 轻微延迟避免CPU过载
                        continue

                    # 捕获摄像头帧
                    frame = self.cam2.capture_array()

                    # 使用更新后的YOLO检测器仅检测人脸
                    face_coordinates = self.yolo_detector.detect_face(frame)

                    # 如果检测到人脸，计算机械臂角度
                    if face_coordinates:
                        # 使用object_tracker计算机械臂角度
                        base_angle, wrist_angle = self.object_tracker.track_face(
                            face_coordinates
                        )

                        # 设置舵机角度（只需更新底座和腕部）
                        # 将角度转换为度数（0-180度范围） #可能为负数
                        base_deg = int(np.degrees(base_angle))
                        wrist_deg = int(np.degrees(wrist_angle))

                        # 发送舵机控制命令
                        # self.serial_comm.send_servo_action('Alldro', base_deg, 60, 30, wrist_deg)
                        self.serial_comm.send_servos_smooth(
                            [base_deg, 60, 30, wrist_deg], duration=3.0
                        )
                        time.sleep(3)  # 休眠3s

                        with self.lock:
                            self.yolo_results = {"face": face_coordinates}

                    time.sleep(0.03)  # 降低CPU占用
                except Exception as e:
                    print(f"人脸跟踪异常: {e}")
                    time.sleep(0.5)  # 出错时延长等待时间

        def voice_processor():
            """语音处理线程"""
            while not self.stop_threads:
                try:
                    self.voice.record(duration=3)  # 录制3秒音频
                    text = self.voice.speech_to_text()  # 语音转文本

                    # 激活阶段处理
                    if not self.voice_activated:
                        if "你好悠悠" in text or "你好,悠悠" in text:
                            with self.lock:
                                self.voice_activated = True
                            self.voice.voice_feedback("在呢")  # 直接语音反馈

                    # 已激活指令处理
                    else:
                        response = self.voice.process_command(text)
                        if response == "[EXIT]":  # 退出指令
                            with self.lock:
                                self.voice_activated = False
                            self.voice.voice_feedback("已退出，需要悠悠时请说你好悠悠")
                        else:
                            self.voice.voice_feedback(response)  # 直接语音输出

                except Exception as e:
                    print(f"语音处理异常: {e}")
                    time.sleep(0.5)  # 出错时延长等待时间

        # ----------------- 线程启动 -----------------
        threads = [
            threading.Thread(
                target=gesture_monitor, daemon=True
            ),  # 手势线程设为守护线程
            threading.Thread(target=yolo_tracker),
            threading.Thread(target=voice_processor),
        ]
        for t in threads:  # 启动所有线程
            t.start()

        # ----------------- 主控制循环 -----------------
        try:
            while True:
                # 实时获取手势状态（带锁保护）
                with self.lock:
                    current_gesture_state = self.gesture_state  # 获取当前手势状态
                    yolo_data = self.yolo_results  # 获取YOLO检测结果

                # ----------------- 模式切换检测 -----------------
                if current_gesture_state != "interactive":
                    print(f"手势触发模式切换 -> {current_gesture_state}")
                    self.state = current_gesture_state  # 更新系统主状态
                    break  # 退出互动模式循环

                # 降低CPU占用（关键！避免主循环卡死线程）
                time.sleep(0.05)  # 20Hz刷新率

        # ----------------- 资源清理 -----------------
        finally:
            # 设置停止标志
            self.stop_threads = True
            print("正在停止所有线程...")

            # 等待非守护线程结束（最大等待1秒）
            for t in threads[1:]:  # 跳过守护线程
                t.join(timeout=1)

            # 强制关闭摄像头
            self._stop_cameras()

            # 清理OpenCV窗口
            try:
                if (
                    cv2.getWindowProperty("Gesture Recognition", cv2.WND_PROP_VISIBLE)
                    >= 1
                ):
                    cv2.destroyAllWindows()
            except:
                cv2.destroyAllWindows()  # 尝试关闭所有窗口

            print("互动模式资源已释放")

    def _work_state(self):
        """工作模式主函数（多线程）"""

        # 确保摄像头已初始化并启动
        self._start_cameras()

        # ----------------- 共享变量定义 -----------------
        self.gesture_state = "work"
        self.yolo_results = None
        self.posture_alerts = []
        self.stop_threads = False
        self.lock = threading.Lock()

        # ----------------- 创建组合系统实例 -----------------
        # 创建组合系统并直接传递共享摄像头
        self.csi_system = CombinedSystem(shared_camera=self.cam1)

        # ----------------- 线程定义 -----------------
        def csi_monitor():
            """综合检测线程（手势+坐姿）"""
            try:
                # 运行组合系统的主循环
                final_state = self.csi_system.run()
                with self.lock:
                    self.gesture_state = final_state
            except Exception as e:
                print(f"[综合检测异常] {str(e)}")

        def yolo_tracker():
            """书本跟踪线程"""
            while not self.stop_threads:
                try:
                    # 确保摄像头正在运行
                    if (
                        self.cam2 is None
                        or not hasattr(self.cam2, "is_running")
                        or not self.cam2.is_running
                    ):
                        time.sleep(0.1)
                        continue

                    frame = self.cam2.capture_array()

                    # 使用更新后的YOLO检测器仅检测书本
                    book_coordinates = self.yolo_detector.detect_book(frame)

                    # 如果检测到书本，计算机械臂角度
                    if book_coordinates:
                        # 使用object_tracker计算四轴机械臂角度
                        base_angle, shoulder_angle, elbow_angle, wrist_angle = (
                            self.object_tracker.track_book(book_coordinates)
                        )

                        # 将角度转换为度数（0-180度范围）#度数范围
                        base_deg = int(np.degrees(base_angle))
                        shoulder_deg = int(np.degrees(shoulder_angle))
                        elbow_deg = int(np.degrees(elbow_angle))
                        wrist_deg = int(np.degrees(wrist_angle))

                        # 发送舵机控制命令（四个舵机都需控制）
                        # self.serial_comm.send_servo_action('Alldro', base_deg, shoulder_deg, elbow_deg, wrist_deg)
                        self.serial_comm.send_servos_smooth(
                            [base_deg, shoulder_deg, elbow_deg, wrist_deg], duration=3.0
                        )
                        time.sleep(3)  # 休眠3s

                        with self.lock:
                            self.yolo_results = {"book": book_coordinates}

                    time.sleep(0.03)
                except Exception as e:
                    print(f"[YOLO异常] {str(e)}")
                    time.sleep(0.5)  # 异常时增加等待时间

        # ----------------- 线程启动 -----------------
        threads = [
            threading.Thread(target=csi_monitor, daemon=True),
            threading.Thread(target=yolo_tracker),
        ]
        for t in threads:
            t.start()

        # ----------------- 主控制循环 -----------------
        try:
            while True:
                with self.lock:
                    current_state = self.gesture_state  # 获取当前状态
                    book_data = self.yolo_results  # 获取书本检测结果
                    # 从组合系统获取坐姿警报
                    posture_alerts = []
                    if self.csi_system and hasattr(self.csi_system, "posture_alerts"):
                        posture_alerts = (
                            self.csi_system.posture_alerts
                        )  # 在类里面写一个坐姿警报的语音

                # 模式切换检测
                if current_state != "work":
                    print(f"状态变更 -> {current_state}")
                    self.state = current_state
                    break

                if time.time() - self.last_posture_alert > self.alert_interval:
                    alert_messages = []

                    # 检测警报类型
                    has_head_low = any("Head Too Close" in a for a in posture_alerts)
                    has_shoulder_tilt = any(
                        "Shoulder Tilt" in a for a in posture_alerts
                    )

                    # 组合提醒内容
                    if has_head_low and has_shoulder_tilt:
                        alert_messages.append("主人要注意头部和坐姿哦")
                    elif has_head_low:
                        alert_messages.append("主人要注意不要将头低的太低哦")
                    elif has_shoulder_tilt:
                        alert_messages.append("主人要注意坐姿哦")

                    # 执行语音提醒
                    if alert_messages:
                        # 随机选择一条提醒（避免重复播报相同内容）
                        self.voice.voice_feedback(alert_messages[0])
                        self.last_posture_alert = time.time()  # 更新最后提醒时间

                time.sleep(0.05)

        finally:
            # 设置停止标志
            self.stop_threads = True
            print("正在停止所有线程...")

            # 等待线程结束（最大等待1秒）
            for t in threads[1:]:  # 跳过守护线程
                t.join(timeout=1)

            # 关闭摄像头
            self._stop_cameras()

            # 清空OpenCV窗口
            try:
                if (
                    cv2.getWindowProperty("Gesture Recognition", cv2.WND_PROP_VISIBLE)
                    >= 1
                ):
                    cv2.destroyAllWindows()
            except:
                cv2.destroyAllWindows()  # 尝试关闭所有窗口

            print("【工作模式】资源已释放")


# 主程序入口（这部分保持不变）
if __name__ == "__main__":
    lamp = SmartLampSystem()
    lamp.state_machine()
