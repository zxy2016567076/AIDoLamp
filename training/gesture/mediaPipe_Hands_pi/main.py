import cv2  # 导入OpenCV模块
import joblib  # 导入Joblib模块
import numpy as np  # 导入NumPy模块
import mediapipe as mp  # 导入MediaPipe模块
import time  # 导入时间模块
import os  # 导入操作系统模块

class GestureRecognizer:
    def __init__(self):
        self.model = joblib.load(os.path.join('data', 'processed', 'gesture_model.pkl'))  # 加载手势识别模型
        self.mp_hands = mp.solutions.hands  # 导入手部模型
        self.hands = self.mp_hands.Hands(
            static_image_mode=False, # 非静态图像模式
            max_num_hands=1, #最大检测手数
            min_detection_confidence=0.7, # 最小检测置信度
            min_tracking_confidence=0.5 # 最小跟踪置信度
        )
        
        # 状态变量
        self.activated = False  # 系统激活状态
        self.activation_start_time = 0  # 激活开始时间
        self.current_mode = None  # 当前模式
        self.last_action_time = 0  # 上次操作时间
        self.gesture_timers = {}  # 存储各手势开始时间
        self.confirmed_label = None  # 已确认的手势

    def process_frame(self, frame):
        """处理单个帧并返回预测结果和标签"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 转换图像为RGB格式
        results = self.hands.process(frame_rgb)  # 处理图像，提取手部关键点
        
        if not results.multi_hand_landmarks:  # 如果没有检测到手部关键点
            return frame, None  # 返回原始帧和None
        
        # 提取特征
        hand_landmarks = results.multi_hand_landmarks[0]  # 获取第一个手部关键点
        keypoints = []  # 初始化关键点列表
        for lm in hand_landmarks.landmark:  # 遍历所有关键点
            keypoints.extend([lm.x, lm.y, lm.z])  # 添加关键点的x, y, z坐标
        
        if len(keypoints) != 63:  # 如果关键点长度不为63
            return frame, None  # 返回原始帧和None
        
        label = self.model.predict([keypoints])[0]  # 预测手势标签
        return frame, label  # 返回处理后的帧和标签

    def check_gesture_duration(self, current_label):
        """验证手势持续时间，返回是否达到要求"""
        # 1号手势特殊处理（需要3秒）
        if current_label == 1:
            if current_label not in self.gesture_timers:  # 如果手势计时器不存在
                self.gesture_timers[current_label] = time.time()  # 初始化计时器
            return time.time() - self.gesture_timers[current_label] >= 3  # 检查是否达到3秒
        
        # 其他手势需要1秒
        if current_label not in self.gesture_timers:  # 如果手势计时器不存在
            self.gesture_timers[current_label] = time.time()  # 初始化计时器
        return time.time() - self.gesture_timers[current_label] >= 1  # 检查是否达到1秒

    def handle_mode_operations(self, label):
        """处理模式内的操作逻辑"""
        if self.current_mode and self.check_gesture_duration(label):  # 如果当前模式存在且手势持续时间达到要求
            if label == 10: #上
                print("1")
            elif label == 2:#下
                print("2")
            elif label == 9:#右
                print("3")
            elif label == 4:#左
                print("4")
            elif label == 3:  # 退出当前模式
                print(f"退出模式{self.current_mode}")
                self.current_mode = None  # 清空当前模式
            self.last_action_time = time.time()  # 更新上次操作时间
            self.gesture_timers.clear()  # 清空手势计时器

    def real_time_recognition(self):
        """实时摄像头识别"""
        cap = cv2.VideoCapture(0)  # 打开摄像头
        
        while cap.isOpened():  # 如果摄像头打开
            ret, frame = cap.read()  # 读取摄像头数据
            if not ret:  # 如果没有读取到数据
                break  # 退出循环
                
            # 镜像显示
            frame = cv2.flip(frame, 1)  # 水平翻转图像
            
            # 处理帧
            processed_frame, label = self.process_frame(frame)  # 处理帧并获取标签
            display_text = ""  # 初始化显示文本
            current_time = time.time()  # 获得当前时间
            
            if label is not None:  # 如果有手势
                # 清除非当前手势的计时器
                for l in list(self.gesture_timers.keys()):  # 遍历所有手势
                    if l != label:  # 如果不是当前手势
                        del self.gesture_timers[l]  # 删除非当前手势的计时器

                # 状态机逻辑
                if not self.activated:  # 如果没有激活
                    if label == 1 and self.check_gesture_duration(label):  # 如果是1号手势且时间达到要求
                        self.activated = True  # 激活
                        self.gesture_timers.clear()  # 清空计时器
                        print("系统已激活！")
                    else:
                        self.activation_start_time = 0  # 激活开始时间清零
                else:#激活了
                    #激活后应该再判断当前模式有没有激活，不然识别到退出手势就直接退出了
                    if label == 3 and self.check_gesture_duration(label):  # 如果是3号手势(退出)且时间达到要求
                        self.activated = False  # 关闭激活
                        self.current_mode = None  # 退出模式
                        self.gesture_timers.clear()  # 清空计时器
                        print("系统已关闭")
                    elif label in [5, 6, 7, 8] and not self.current_mode:  # 如果是模式手势且当前没有模式
                        if self.check_gesture_duration(label):  # 检查手势持续时间
                            self.current_mode = label - 4  # 5->1, 6->2, 7->3, 8->4
                            print(f"进入模式{self.current_mode}")
                            self.gesture_timers.clear()  # 清空计时器
                    elif self.current_mode:  # 如果当前模式存在
                        self.handle_mode_operations(label)  # 处理模式内的操作逻辑
                
                # 更新显示文本
                display_text = f"Label: {label}"  # 显示手势标签
                if self.activated:
                    display_text += " | 已激活"  # 显示激活状态
                if self.current_mode:
                    display_text += f" | 模式{self.current_mode}"  # 显示当前模式

            # 绘制进度条
            if label in self.gesture_timers:  # 如果手势在计时器中
                duration = 3 if label == 1 else 1  # 设置持续时间
                elapsed = current_time - self.gesture_timers[label]  # 计算已持续时间
                progress = min(elapsed / duration, 1.0)  # 计算进度
                cv2.rectangle(processed_frame, (20, 80), (220, 100), (255, 255, 255), 2)  # 绘制进度条边框
                cv2.rectangle(processed_frame, (20, 80), 
                              (20 + int(200 * progress), 100), 
                              (0, 255, 0), -1)  # 绘制进度条

            # 显示文本
            cv2.putText(processed_frame, display_text, (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  # 在图像上绘制文本
            
            cv2.imshow('Gesture Recognition', processed_frame)  # 显示处理后的图像
            
            if cv2.waitKey(1) & 0xFF == 27:  # ESC退出
                break  # 退出循环
                
        cap.release()  # 释放摄像头
        cv2.destroyAllWindows()  # 关闭所有OpenCV窗口

if __name__ == '__main__':
    recognizer = GestureRecognizer()  # 创建手势识别器实例
    recognizer.real_time_recognition()  # 开始实时手势识别