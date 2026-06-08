"""
手势识别程序主模块：实时摄像头手势识别和控制系统
"""
import cv2
import joblib
import numpy as np
import mediapipe as mp
import time
import os
import sys
import argparse
import logging

# 导入配置和日志
from config import (
    CLASS_MAPPING, REVERSE_CLASS_MAPPING, MODEL_PATH,
    PROCESSED_DATA_DIR, MIN_DETECTION_CONFIDENCE, MIN_TRACKING_CONFIDENCE,
    MAX_NUM_HANDS, ACTIVATION_GESTURE_DURATION, NORMAL_GESTURE_DURATION,
    UI_TEXT_COLOR, UI_TEXT_FONT_SIZE, UI_TEXT_THICKNESS,
    UI_PROGRESS_BAR_COLOR, UI_PROGRESS_BAR_BG_COLOR,
    UI_PROGRESS_BAR_WIDTH, UI_PROGRESS_BAR_HEIGHT,
    UI_PROGRESS_BAR_X, UI_PROGRESS_BAR_Y,
    UI_TEXT_X, UI_TEXT_Y, CAMERA_ID
)
from logger import setup_logger

# 1_feature_extractor模块用于数据收集
from typing import Optional, Tuple, Dict, List, Any, Union

# 设置日志记录器
logger = setup_logger("gesture_recognizer", "gesture_recognizer")

class GestureRecognizer:
    """手势识别器类"""
    def __init__(self, model_path: str = MODEL_PATH, camera_id: int = CAMERA_ID):
        """
        初始化手势识别器
        
        参数:
            model_path: 模型文件路径
            camera_id: 摄像头ID
        """
        logger.info("初始化手势识别器...")
        self.camera_id = camera_id
        
        # 加载模型
        try:
            self.model = joblib.load(model_path)
            logger.info(f"成功加载模型: {model_path}")
            
            # 尝试加载特征缩放器
            scaler_path = os.path.join(PROCESSED_DATA_DIR, 'feature_scaler.pkl')
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                logger.info("成功加载特征缩放器")
            else:
                self.scaler = None
                logger.warning("未找到特征缩放器，将使用原始特征")
        except Exception as e:
            logger.error(f"加载模型失败: {str(e)}")
            raise RuntimeError(f"无法加载模型: {str(e)}")
        
        # 初始化MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=MAX_NUM_HANDS,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE
        )
        
        # 状态变量
        self.activated = False
        self.activation_start_time = 0
        self.current_mode = None
        self.last_action_time = 0
        self.gesture_timers: Dict[int, float] = {}  # 存储各手势开始时间
        self.confirmed_label = None  # 已确认的手势
        
        # 性能监控变量
        self.frame_count = 0
        self.fps = 0
        self.fps_time = time.time()
        
        # 手势提示图
        self.gesture_reference_images: Dict[int, Optional[np.ndarray]] = {}
        self.load_gesture_reference_images()
        
        # 功能映射
        self.mode_functions = {
            1: self.mode1_functions,
            2: self.mode2_functions,
            3: self.mode3_functions,
            4: self.mode4_functions
        }
        
        logger.info("手势识别器初始化完成")

    def load_gesture_reference_images(self) -> None:
        """加载手势参考图像"""
        vis_dir = os.path.join(PROCESSED_DATA_DIR, 'visualizations')
        if os.path.exists(vis_dir):
            for class_id in CLASS_MAPPING:
                img_path = os.path.join(vis_dir, f'class_{class_id}.jpg')
                if os.path.exists(img_path):
                    try:
                        self.gesture_reference_images[class_id] = cv2.imread(img_path)
                    except Exception as e:
                        logger.warning(f"无法加载手势参考图 {class_id}: {str(e)}")
        else:
            logger.warning(f"手势参考图目录不存在: {vis_dir}")

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[int], Optional[List[Any]]]:
        """
        处理单个帧并返回预测结果和标签
        
        参数:
            frame: 输入图像帧
            
        返回:
            processed_frame: 处理后的帧
            label: 预测的手势类别
            hand_landmarks: 检测到的手部关键点
        """
        if frame is None:
            logger.error("输入帧为空")
            return None, None, None
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        
        # 创建处理后的帧副本
        processed_frame = frame.copy()
        
        if not results.multi_hand_landmarks:
            return processed_frame, None, None
        
        # 绘制手部关键点
        for hand_landmarks in results.multi_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                processed_frame,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style()
            )
        
        # 提取特征
        hand_landmarks = results.multi_hand_landmarks[0]  # 只处理第一只手
        keypoints = []
        for lm in hand_landmarks.landmark:
            keypoints.extend([lm.x, lm.y, lm.z])
        
        if len(keypoints) != 63:
            logger.warning(f"关键点数量异常: {len(keypoints)}")
            return processed_frame, None, hand_landmarks
        
        # 应用特征缩放（如果有）
        if self.scaler is not None:
            keypoints = self.scaler.transform([keypoints])[0]
        
        # 预测手势
        try:
            label = self.model.predict([keypoints])[0]
            
            # 可选：获取预测概率（如果模型支持）
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba([keypoints])[0]
                max_prob = np.max(probabilities)
                
                # 如果最高概率低于阈值，认为是不确定的预测
                if max_prob < 0.6:
                    logger.debug(f"低置信度预测: {label} ({max_prob:.2f})")
                    # 可以选择忽略低置信度预测
                    # return processed_frame, None, hand_landmarks
            
            return processed_frame, label, hand_landmarks
            
        except Exception as e:
            logger.error(f"预测出错: {str(e)}")
            return processed_frame, None, hand_landmarks

    def check_gesture_duration(self, current_label: int) -> bool:
        """
        验证手势持续时间，返回是否达到要求
        
        参数:
            current_label: 当前检测到的手势标签
            
        返回:
            bool: 是否达到所需持续时间
        """
        # 激活手势(1号)特殊处理（需要3秒）
        if current_label == REVERSE_CLASS_MAPPING.get('activate', 1):
            required_duration = ACTIVATION_GESTURE_DURATION
        else:
            required_duration = NORMAL_GESTURE_DURATION
        
        current_time = time.time()
        if current_label not in self.gesture_timers:
            self.gesture_timers[current_label] = current_time
            return False
        
        elapsed_time = current_time - self.gesture_timers[current_label]
        return elapsed_time >= required_duration

    def mode1_functions(self, label: int) -> None:
        """模式1的功能实现"""
        if label == REVERSE_CLASS_MAPPING.get('up', 10):
            logger.info("模式1-向上操作：增加音量") 
            # 这里实现实际的功能，如音量控制
            # 例如：os.system("amixer -D pulse sset Master 5%+")
            print("1")
        elif label == REVERSE_CLASS_MAPPING.get('down', 2):
            logger.info("模式1-向下操作：减小音量")
            # 例如：os.system("amixer -D pulse sset Master 5%-")
            print("2")
        elif label == REVERSE_CLASS_MAPPING.get('left', 4):
            logger.info("模式1-向左操作：上一曲")
            # 实现上一曲功能
            print("3")
        elif label == REVERSE_CLASS_MAPPING.get('right', 9):
            logger.info("模式1-向右操作：下一曲")
            # 实现下一曲功能
            print("4")

    def mode2_functions(self, label: int) -> None:
        """模式2的功能实现"""
        if label == REVERSE_CLASS_MAPPING.get('up', 10):
            logger.info("模式2-向上操作：屏幕亮度增加")
            # 亮度增加功能
            print("1")
        elif label == REVERSE_CLASS_MAPPING.get('down', 2):
            logger.info("模式2-向下操作：屏幕亮度减小")
            # 亮度减小功能
            print("2")
        elif label == REVERSE_CLASS_MAPPING.get('left', 4):
            logger.info("模式2-向左操作：切换到上一个应用")
            # 切换应用功能
            print("3")
        elif label == REVERSE_CLASS_MAPPING.get('right', 9):
            logger.info("模式2-向右操作：切换到下一个应用")
            # 切换应用功能
            print("4")

    def mode3_functions(self, label: int) -> None:
        """模式3的功能实现"""
        if label == REVERSE_CLASS_MAPPING.get('up', 10):
            logger.info("模式3-向上操作：页面向上滚动")
            # 页面向上滚动功能
        elif label == REVERSE_CLASS_MAPPING.get('down', 2):
            logger.info("模式3-向下操作：页面向下滚动")
            # 页面向下滚动功能
        elif label == REVERSE_CLASS_MAPPING.get('left', 4):
            logger.info("模式3-向左操作：后退")
            # 浏览器后退功能
        elif label == REVERSE_CLASS_MAPPING.get('right', 9):
            logger.info("模式3-向右操作：前进")
            # 浏览器前进功能

    def mode4_functions(self, label: int) -> None:
        """模式4的功能实现"""
        if label == REVERSE_CLASS_MAPPING.get('up', 10):
            logger.info("模式4-向上操作：鼠标向上移动")
            # 鼠标向上移动功能
        elif label == REVERSE_CLASS_MAPPING.get('down', 2):
            logger.info("模式4-向下操作：鼠标向下移动")
            # 鼠标向下移动功能
        elif label == REVERSE_CLASS_MAPPING.get('left', 4):
            logger.info("模式4-向左操作：鼠标向左移动")
            # 鼠标向左移动功能
        elif label == REVERSE_CLASS_MAPPING.get('right', 9):
            logger.info("模式4-向右操作：鼠标向右移动")
            # 鼠标向右移动功能

    def handle_mode_operations(self, label: int) -> None:
        """
        处理模式内的操作逻辑
        
        参数:
            label: 当前检测到的手势标签
        """
        if not self.current_mode:
            return
            
        exit_label = REVERSE_CLASS_MAPPING.get('exit', 3)
        
        if self.check_gesture_duration(label):
            if label == exit_label:  # 退出当前模式
                logger.info(f"退出模式{self.current_mode}")
                self.current_mode = None
            elif self.current_mode in self.mode_functions:
                # 调用相应模式的功能处理函数
                self.mode_functions[self.current_mode](label)
                
            self.last_action_time = time.time()
            self.gesture_timers.clear()

    def draw_ui(self, frame: np.ndarray, label: Optional[int], display_fps: bool = True) -> np.ndarray:
        """
        绘制用户界面元素
        
        参数:
            frame: 输入图像帧
            label: 当前检测到的手势标签
            display_fps: 是否显示FPS
            
        返回:
            带有UI元素的帧
        """
        # 显示FPS
        if display_fps:
            self.frame_count += 1
            if (time.time() - self.fps_time) > 1:
                self.fps = self.frame_count / (time.time() - self.fps_time)
                self.frame_count = 0
                self.fps_time = time.time()
            
            # 注释掉显示FPS的代码
            # cv2.putText(
            #     frame, f"FPS: {self.fps:.1f}", 
            #     (frame.shape[1] - 120, 30), 
            #     cv2.FONT_HERSHEY_SIMPLEX, 0.8, UI_TEXT_COLOR, 2
            # )
        
        # 状态信息
        display_text = ""
        if label is not None:
            gesture_name = CLASS_MAPPING.get(label, f"未知({label})")
            display_text = f"gesture: {gesture_name}"
        
        if self.activated:
            display_text += " | activated"
        else:
            display_text += " | Shut"
            
        if self.current_mode:
            display_text += f" | mode{self.current_mode}"
        
        cv2.putText(
            frame, display_text, 
            (UI_TEXT_X, UI_TEXT_Y), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            UI_TEXT_FONT_SIZE, 
            UI_TEXT_COLOR, 
            UI_TEXT_THICKNESS
        )
        
        # 显示手势持续时间进度条
        if label in self.gesture_timers:
            required_duration = ACTIVATION_GESTURE_DURATION if label == REVERSE_CLASS_MAPPING.get('activate', 1) else NORMAL_GESTURE_DURATION
            elapsed = time.time() - self.gesture_timers[label]
            progress = min(elapsed / required_duration, 1.0)
            
            # 绘制进度条背景
            cv2.rectangle(
                frame, 
                (UI_PROGRESS_BAR_X, UI_PROGRESS_BAR_Y), 
                (UI_PROGRESS_BAR_X + UI_PROGRESS_BAR_WIDTH, UI_PROGRESS_BAR_Y + UI_PROGRESS_BAR_HEIGHT), 
                UI_PROGRESS_BAR_BG_COLOR, 
                2
            )
            
            # 绘制进度条填充
            cv2.rectangle(
                frame, 
                (UI_PROGRESS_BAR_X, UI_PROGRESS_BAR_Y), 
                (UI_PROGRESS_BAR_X + int(UI_PROGRESS_BAR_WIDTH * progress), UI_PROGRESS_BAR_Y + UI_PROGRESS_BAR_HEIGHT), 
                UI_PROGRESS_BAR_COLOR, 
                -1
            )
        
        # 显示当前模式可用的手势提示
        if self.activated:
            reference_y = UI_PROGRESS_BAR_Y + UI_PROGRESS_BAR_HEIGHT + 20
            
            # 如果在某个模式中，显示该模式的操作手势
            if self.current_mode:
                gestures_to_show = [
                    REVERSE_CLASS_MAPPING.get('up', 10),
                    REVERSE_CLASS_MAPPING.get('down', 2),
                    REVERSE_CLASS_MAPPING.get('left', 4),
                    REVERSE_CLASS_MAPPING.get('right', 9),
                    REVERSE_CLASS_MAPPING.get('exit', 3)
                ]
                
                cv2.putText(
                    frame, 
                    f"mode {self.current_mode} gesture:", 
                    (20, reference_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, 
                    (255, 255, 255), 
                    1
                )
                reference_y += 30
            
            # 否则显示模式选择手势
            else:
                gestures_to_show = [
                    REVERSE_CLASS_MAPPING.get('mode1', 5),
                    REVERSE_CLASS_MAPPING.get('mode2', 6),
                    REVERSE_CLASS_MAPPING.get('mode3', 7),
                    REVERSE_CLASS_MAPPING.get('mode4', 8),
                    REVERSE_CLASS_MAPPING.get('exit', 3)
                ]
                
                cv2.putText(
                    frame, 
                    "mode:", 
                    (20, reference_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, 
                    (255, 255, 255), 
                    1
                )
                reference_y += 30
            
            # 显示参考图像
            ref_img_width = 80
            ref_img_height = 60
            for i, gesture_id in enumerate(gestures_to_show):
                if gesture_id in self.gesture_reference_images:
                    ref_img = self.gesture_reference_images[gesture_id]
                    if ref_img is not None:
                        # 调整参考图像大小
                        ref_img_resized = cv2.resize(ref_img, (ref_img_width, ref_img_height))
                        
                        # 计算位置
                        x_offset = 20 + i * (ref_img_width + 10)
                        
                        # 放置图像
                        frame[reference_y:reference_y+ref_img_height, x_offset:x_offset+ref_img_width] = ref_img_resized
                        
                        # 添加标签
                        gesture_name = CLASS_MAPPING.get(gesture_id, f"未知({gesture_id})")
                        cv2.putText(
                            frame, 
                            gesture_name, 
                            (x_offset, reference_y + ref_img_height + 15), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.4, 
                            (200, 200, 200), 
                            1
                        )
        
        # 如果未激活，显示激活提示
        elif not self.activated:
            activate_id = REVERSE_CLASS_MAPPING.get('activate', 1)
            if activate_id in self.gesture_reference_images:
                ref_img = self.gesture_reference_images[activate_id]
                if ref_img is not None:
                    # 在画面中央显示激活手势提示
                    ref_img_width = 120
                    ref_img_height = 90
                    ref_img_resized = cv2.resize(ref_img, (ref_img_width, ref_img_height))
                    
                    x_offset = (frame.shape[1] - ref_img_width) // 2
                    y_offset = (frame.shape[0] - ref_img_height) // 2 - 50
                    
                    # 创建半透明覆盖层
                    overlay = frame.copy()
                    cv2.rectangle(
                        overlay, 
                        (x_offset-10, y_offset-40), 
                        (x_offset+ref_img_width+10, y_offset+ref_img_height+40), 
                        (0, 0, 0), 
                        -1
                    )
                    
                    # 应用半透明效果
                    alpha = 0.7
                    cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)
                    
                    # 添加手势图像
                    frame[y_offset:y_offset+ref_img_height, x_offset:x_offset+ref_img_width] = ref_img_resized
                    
                    # 添加提示文字
                    cv2.putText(
                        frame, 
                        "need activate", 
                        (x_offset - 30, y_offset - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.8, 
                        (255, 255, 255), 
                        2
                    )
        
        return frame

    def real_time_recognition(self, show_ui: bool = True, mirror: bool = True) -> None:
        """
        实时摄像头识别主循环
        
        参数:
            show_ui: 是否显示UI
            mirror: 是否镜像显示
        """
        # 打开摄像头
        try:
            logger.info(f"正在打开摄像头 ID: {self.camera_id}")
            cap = cv2.VideoCapture(self.camera_id)
            
            if not cap.isOpened():
                logger.error(f"无法打开摄像头 ID: {self.camera_id}")
                raise RuntimeError(f"无法打开摄像头 ID: {self.camera_id}")
                
            # 可选：设置摄像头分辨率
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            logger.info("摄像头已成功打开，开始手势识别")
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    logger.error("无法读取摄像头帧")
                    break
                    
                # 镜像显示
                if mirror:
                    frame = cv2.flip(frame, 1)
                
                # 处理帧
                processed_frame, label, hand_landmarks = self.process_frame(frame)
                
                # UI绘制
                if show_ui:
                    processed_frame = self.draw_ui(processed_frame, label)
                
                # 状态机逻辑
                if label is not None:
                    # 清除非当前手势的计时器
                    for l in list(self.gesture_timers.keys()):
                        if l != label:
                            del self.gesture_timers[l]

                    # 激活状态机
                    activate_label = REVERSE_CLASS_MAPPING.get('activate', 1)
                    exit_label = REVERSE_CLASS_MAPPING.get('exit', 3)
                    
                    if not self.activated:
                        if label == activate_label and self.check_gesture_duration(label):
                            self.activated = True
                            self.gesture_timers.clear()
                            logger.info("Activate！")
                    else:
                        if label == exit_label and self.check_gesture_duration(label) and not self.current_mode:
                            self.activated = False
                            self.current_mode = None
                            self.gesture_timers.clear()
                            logger.info("Shut")
                        elif label in [
                            REVERSE_CLASS_MAPPING.get('mode1', 5),
                            REVERSE_CLASS_MAPPING.get('mode2', 6),
                            REVERSE_CLASS_MAPPING.get('mode3', 7),
                            REVERSE_CLASS_MAPPING.get('mode4', 8)
                        ] and not self.current_mode:
                            if self.check_gesture_duration(label):
                                # 计算模式编号：5->1, 6->2等
                                mode_number = label - 4
                                self.current_mode = mode_number
                                logger.info(f"进入模式{self.current_mode}")
                                self.gesture_timers.clear()
                        elif self.current_mode:
                            self.handle_mode_operations(label)
                
                # 显示帧
                cv2.imshow('gesture', processed_frame)
                
                # 按ESC键退出
                if cv2.waitKey(1) & 0xFF == 27:
                    logger.info("用户按ESC键退出程序")
                    break
                    
        except Exception as e:
            logger.error(f"实时识别过程中发生错误: {str(e)}")
            raise
        finally:
            logger.info("关闭摄像头和窗口")
            if 'cap' in locals() and cap is not None:
                cap.release()
            cv2.destroyAllWindows()
            
            # 确保MediaPipe资源释放
            if hasattr(self, 'hands'):
                self.hands.close()

def sample_collection_mode():
    """启动样本收集模式"""
    from importlib import import_module
    
    try:
        # 动态导入特征提取器模块
        feature_extractor = import_module('1_feature_extractor')
        
        print("=== 手势样本收集模式 ===")
        print("可用的手势类别:")
        for class_id, class_name in CLASS_MAPPING.items():
            print(f"  {class_id}: {class_name}")
        
        while True:
            try:
                class_id = int(input("\n请输入要收集的手势类别ID (输入0退出): "))
                if class_id == 0:
                    break
                    
                if class_id not in CLASS_MAPPING:
                    print(f"错误: 无效的类别ID {class_id}")
                    continue
                
                num_samples = int(input(f"要收集多少个 '{CLASS_MAPPING[class_id]}' 样本? (默认20): ") or "20")
                
                # 调用样本收集函数
                feature_extractor.collect_new_samples(class_id, num_samples)
                
                cont = input("\n继续收集其他类别? (y/n): ").lower()
                if cont != 'y':
                    break
                    
            except ValueError:
                print("请输入有效的数字")
            except KeyboardInterrupt:
                print("\n用户中断")
                break
                
        print("样本收集完成！")
        
    except ImportError:
        print("错误: 无法导入特征提取器模块。请确保1_feature_extractor.py存在。")
    except Exception as e:
        print(f"样本收集过程中出错: {str(e)}")

def show_usage_guide():
    """显示使用指南"""
    print("\n=== 手势识别系统使用指南 ===")
    print("\n系统状态:")
    print("1. 未激活状态: 系统启动时处于未激活状态")
    print("2. 已激活状态: 做激活手势(1-activate)3秒后进入")
    print("3. 模式选择状态: 激活后可选择进入不同模式")
    print("4. 模式操作状态: 进入特定模式后可执行相应操作")
    
    print("\n手势列表:")
    for class_id, class_name in CLASS_MAPPING.items():
        print(f"  {class_id}: {class_name}")
    
    print("\n基本操作流程:")
    print("1. 激活: 做'1-activate'手势3秒激活系统")
    print("2. 选择模式: 激活后做'5-mode1'至'8-mode4'手势1秒选择模式")
    print("3. 模式内操作: 进入模式后，可用以下手势:")
    print("   - 上: '10-up'手势")
    print("   - 下: '2-down'手势")
    print("   - 左: '4-left'手势")
    print("   - 右: '9-right'手势")
    print("4. 退出: 在任意模式中做'3-exit'手势1秒退出当前模式")
    print("   或在模式选择状态做'3-exit'手势1秒退出系统")
    
    print("\n特别说明:")
    print("- 所有操作手势需要保持1秒才会触发（激活手势需要3秒）")
    print("- 每个模式对应不同的功能控制集")
    print("- 界面上会显示实时手势识别结果和状态指示")
    print("- 按ESC键随时退出程序")

def main():
    """主程序入口"""
    parser = argparse.ArgumentParser(description='手势识别控制系统')
    parser.add_argument('--collect', action='store_true', help='进入样本收集模式')
    parser.add_argument('--guide', action='store_true', help='显示使用指南')
    parser.add_argument('--camera', type=int, default=CAMERA_ID, help='摄像头ID (默认: 0)')
    parser.add_argument('--no-ui', action='store_true', help='不显示UI界面')
    parser.add_argument('--no-mirror', action='store_true', help='不使用镜像显示')
    
    args = parser.parse_args()
    
    # 显示使用指南
    if args.guide:
        show_usage_guide()
        return
    
    # 样本收集模式
    if args.collect:
        sample_collection_mode()
        return
    
    try:
        # 创建识别器实例
        recognizer = GestureRecognizer(camera_id=args.camera)
        
        # 启动实时识别
        recognizer.real_time_recognition(
            show_ui=not args.no_ui,
            mirror=not args.no_mirror
        )
    except Exception as e:
        print(f"程序运行出错: {str(e)}")
        logger.error(f"程序运行出错: {str(e)}")
        return 1
    
    return 0

if __name__ == '__main__': 
    exit_code = main()
    sys.exit(exit_code)
