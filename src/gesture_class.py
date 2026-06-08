# 导入所需库
import cv2
import joblib
import mediapipe as mp
import time
import os
from picamera2 import Picamera2  # 树莓派专用摄像头库
from libcamera import Transform  # 用于摄像头硬件翻转配置
from serial_comm import SerialCommunicator
from voice_class import VoiceAssistant


class GestureRecognizer:
    def __init__(self):
        # 加载手势识别模型（确保模型路径正确）
        self.model = joblib.load(os.path.join("gesture_model.pkl"))
        # 初始化MediaPipe手部检测模型
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,  # 非静态图像模式（适合实时视频）
            max_num_hands=1,  # 最多检测1只手
            min_detection_confidence=0.7,  # 检测置信度阈值
            min_tracking_confidence=0.5,  # 跟踪置信度阈值
        )

        # 状态控制变量
        self.activated = False  # 系统是否激活
        self.activation_start_time = 0  # 激活开始时间戳
        self.current_mode = None  # 当前操作模式
        self.last_action_time = 0  # 上次操作时间戳
        self.gesture_timers = {}  # 手势计时器（记录各手势开始时间）
        self.confirmed_label = None  # 已确认的手势标签
        self.state = "standby"  # 系统状态（待机 普通 互动 工作）
        self.should_exit = False  # 新增退出标志位

        self.serial_comm = SerialCommunicator()  # 串口通信类

        # 配置语音助手（API 密钥请在 .env 中配置，参考 .env.example）
        voice_config = {
            "BAIDU_APP_ID": os.environ.get("BAIDU_APP_ID", ""),
            "BAIDU_API_KEY": os.environ.get("BAIDU_API_KEY", ""),
            "BAIDU_SECRET_KEY": os.environ.get("BAIDU_SECRET_KEY", ""),
            "DEEPSEEK_API_KEY": os.environ.get("DEEPSEEK_API_KEY", ""),
            "WEATHER_API_KEY": os.environ.get("WEATHER_API_KEY", ""),
        }
        self.voice = VoiceAssistant(voice_config)  # 语音助手

    def process_frame(self, frame):
        """处理帧图像并返回预测结果"""
        # 将图像从BGR转换为RGB（MediaPipe需要RGB输入）
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 使用MediaPipe检测手部关键点
        results = self.hands.process(frame_rgb)

        if not results.multi_hand_landmarks:  # 未检测到手部
            return frame, None

        # 提取第一个检测到的手部关键点坐标
        hand_landmarks = results.multi_hand_landmarks[0]
        keypoints = []
        for lm in hand_landmarks.landmark:
            # 将归一化坐标转为实际坐标（x, y, z）
            keypoints.extend([lm.x, lm.y, lm.z])

        if len(keypoints) != 63:  # 关键点数量校验（21个点×3坐标）
            return frame, None

        # 使用模型预测手势标签
        label = self.model.predict([keypoints])[0]
        return frame, label

    def check_gesture_duration(self, current_label):
        """检查手势持续时间是否达标"""
        # 手势1需要3秒，其他手势需要1秒
        required_time = 3 if current_label == 1 else 1
        if current_label not in self.gesture_timers:
            self.gesture_timers[current_label] = time.time()
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
                        self.serial_comm.send_mode(1)  # 发送模式1指令
                        self.voice.voice_feedback(
                            "已经进入待机模式"
                        )  # 可能会堵塞程序运行
                        self.should_exit = True  # 设置退出标志
                    elif label == 6:
                        print("进入普通状态")
                        self.state = "normal"
                        self.serial_comm.send_mode(2)  # 发送模式2指令
                        self.voice.voice_feedback("已经进入普通模式")
                        self.should_exit = True  # 设置退出标志
                    elif label == 7:
                        print("进入互动状态")
                        self.state = "interactive"
                        self.serial_comm.send_mode(3)  # 发送模式3指令
                        self.voice.voice_feedback("已经进入互动模式")
                        self.should_exit = True  # 设置退出标志
                    elif label == 8:
                        print("进入工作状态")
                        self.state = "work"
                        self.serial_comm.send_mode(4)  # 发送模式4指令
                        self.voice.voice_feedback("已经进入工作模式")
                        self.should_exit = True  # 设置退出标志

                # 模式2：光强和色温调节
                elif mode == 2:
                    if label == 10:  # 上
                        print("增强光照强度")
                        self.serial_comm.send_light_command("LIGHT_ZJ")  # 光强增加
                    elif label == 2:  # 下
                        print("减弱光照强度")
                        self.serial_comm.send_light_command("LIGHT_JX")  # 光强减少
                    elif label == 9:  # 右
                        print("调节色温")
                        self.serial_comm.send_light_command("light_zj")  # 色温增加
                    elif label == 4:  # 左
                        self.serial_comm.send_light_command("light_jx")  # 色温减少

                # 模式3：调节灯的方向
                elif mode == 3:
                    if label == 10:
                        print("灯罩向上")
                        self.serial_comm.send_servo_action("Adroup_3")
                    elif label == 2:
                        print("灯罩向下")
                        self.serial_comm.send_servo_action("Adrodown_3")
                    elif label == 9:
                        print("灯身向右")
                        self.serial_comm.send_servo_action("Adroleft")
                    elif label == 4:
                        print("灯身向左")
                        self.serial_comm.send_servo_action("Adroright")

                # 模式4：调节灯的高度
                elif mode == 4:
                    if label == 10:
                        print("中间舵机向上")
                        self.serial_comm.send_servo_action("Adroup_2")
                    elif label == 2:
                        print("中间舵机向下")
                        self.serial_comm.send_servo_action("Adrodown_2")
                    elif label == 9:
                        print("下面舵机向前")
                        self.serial_comm.send_servo_action("Adroup_1")
                    elif label == 4:
                        print("下面舵机向后")
                        self.serial_comm.send_servo_action("Adrodown_1")

            # 重置计时器和操作时间
            self.last_action_time = time.time()
            self.gesture_timers.clear()
            return self.state  # 返回当前状态

    def real_time_recognition(self, shared_cam=None):
        """实时手势识别主函数"""
        # 初始化Picamera2并配置摄像头参数
        using_shared_camera = shared_cam is not None

        if using_shared_camera:
            # 使用共享的摄像头实例
            picam2 = shared_cam
            should_close_camera = False
            print("手势识别使用共享摄像头实例")
        else:
            # 创建自己的摄像头实例
            try:
                picam2 = Picamera2(1)
                config = picam2.create_video_configuration(
                    main={"size": (320, 240)}, transform=Transform(hflip=True)
                )
                picam2.configure(config)
                picam2.start()
                should_close_camera = True
                print("手势识别创建独立摄像头实例")
            except Exception as e:
                print(f"手势识别摄像头初始化错误: {e}")
                return "standby"  # 出错时返回待机状态

        prev_time = time.time()  # 初始化FPS计时器
        while True:
            # 计算FPS
            current_time = time.time()
            fps = 1 / (current_time - prev_time)
            prev_time = current_time

            # 捕获并处理帧
            frame = picam2.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            processed_frame, label = self.process_frame(frame)
            display_text = ""

            # FPS显示（右上角）
            cv2.putText(
                processed_frame,
                f"FPS: {int(fps)}",
                (processed_frame.shape[1] - 120, 20),  # 右上角位置
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

            if label is not None:
                # 清除非当前手势的计时器
                for l in list(self.gesture_timers.keys()):
                    if l != label:
                        del self.gesture_timers[l]

                # 状态机逻辑
                if not self.activated:
                    if label == 1 and self.check_gesture_duration(label):
                        self.activated = True
                        self.gesture_timers.clear()
                        print("系统已激活！")
                        self.serial_comm.send_oled_command("OLED_1")
                    else:
                        self.activation_start_time = 0
                else:
                    if label == 3 and self.check_gesture_duration(label):
                        self.activated = False
                        self.current_mode = None
                        self.gesture_timers.clear()
                        print("系统已关闭")
                    elif label in [5, 6, 7, 8] and not self.current_mode:
                        if self.check_gesture_duration(label):
                            self.current_mode = label - 4  # 映射模式编号
                            print(f"进入模式{self.current_mode}")
                            self.serial_comm.send_oled_command("OLED_2")
                            self.gesture_timers.clear()
                    elif self.current_mode:
                        self.state = self.handle_mode_operations(
                            label, self.current_mode
                        )  # 传入当前模式

                # 更新后的显示文本（左下角）
                display_text = f"Label: {label}"
                if self.activated:
                    display_text += " | activated"
                if self.current_mode:
                    display_text += f" | mode{self.current_mode}"

            # 调整后的进度条位置（Y坐标上移）
            if label in self.gesture_timers:
                duration = 3 if label == 1 else 1
                elapsed = time.time() - self.gesture_timers[label]
                progress = min(elapsed / duration, 1.0)
                # 进度条位置调整为(20, 40)到(220, 60)
                cv2.rectangle(processed_frame, (20, 40), (220, 60), (255, 255, 255), 2)
                cv2.rectangle(
                    processed_frame,
                    (20, 40),
                    (20 + int(200 * progress), 60),
                    (0, 255, 0),
                    -1,
                )

            # 状态文本显示（左下角）
            cv2.putText(
                processed_frame,
                display_text,
                (20, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

            cv2.imshow("Gesture Recognition", processed_frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break
            if self.should_exit:
                break

        # 在函数结束时安全关闭资源
        if not using_shared_camera and should_close_camera:
            try:
                picam2.stop()  # 只关闭自己创建的摄像头
                print("手势识别释放摄像头资源")
            except Exception as e:
                print(f"手势识别关闭摄像头错误: {e}")

        cv2.destroyAllWindows()  # 关闭窗口
        return self.state  # 返回当前状态


if __name__ == "__main__":
    recognizer = GestureRecognizer()
    state = recognizer.real_time_recognition()
    print("当前状态", state)
