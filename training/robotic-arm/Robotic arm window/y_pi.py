import cv2
import torch
import numpy as np
import time
from ultralytics import YOLO
from picamera2 import Picamera2

class BookDetector:
    def __init__(self, camera_index=1, model_path="best.pt", camera_matrix=None, distance_module=False):
        """
        初始化书本检测器
        
        参数:
        camera_index -- 摄像头索引，默认为1
        model_path -- YOLOv8模型路径，默认使用best
        camera_matrix -- 摄像头内参矩阵，如果有标定结果的话
        distance_module -- 是否使用测距模块
        """
        # # 初始化摄像头
        # self.cap = cv2.VideoCapture(camera_index)
        # if not self.cap.isOpened():
        #     raise ValueError(f"无法打开摄像头 {camera_index}")
        
        # # 获取摄像头分辨率
        # self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        # self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # self.frame_center = (self.frame_width // 2, self.frame_height // 2)
        
        # print(f"摄像头分辨率: {self.frame_width}x{self.frame_height}")
        
        self.picam2 = Picamera2(camera_index)
        
        # 配置摄像头参数（示例配置，根据需要调整）
        config = self.picam2.create_preview_configuration(
        main={
            "size": (320, 240),    # 分辨率
            "format": "BGR888"     # 像素格式
        }
        )
        self.picam2.configure(config)
        self.picam2.start()  # 启动摄像头
        
        # 动态获取实际分辨率（关键修改！）
        self.frame_width = self.picam2.camera_config["main"]["size"][0]  # 获取实际宽度
        self.frame_height = self.picam2.camera_config["main"]["size"][1]  # 获取实际高度
        self.frame_center = (self.frame_width // 2, self.frame_height // 2)
        
        print(f"实际摄像头分辨率: {self.frame_width}x{self.frame_height}")  # 调试输出
        
        # 加载YOLOv8模型
        try:
            self.model = YOLO(model_path)
            print(f"成功加载模型: {model_path}")
        except Exception as e:
            raise ValueError(f"加载模型失败: {e}")
        
        # 设置摄像头内参，用于坐标转换
        self.camera_matrix = camera_matrix
        
        # 是否使用测距模块
        self.use_distance_module = distance_module
        
        # 机械臂当前位置
        self.current_arm_pos = None
        
        # 书本的平均尺寸（厘米），用于估计距离
        self.average_book_size = {"width": 18, "height": 26}
        
        # 图像缩放比例：像素/厘米（需要标定）
        self.pixel_to_cm_ratio = 10  # 默认值，需要通过标定获得更准确的值
        
        # 设置关注的标签（如果只想检测书籍）
        self.target_classes = ['book', 'face']
    
    def simulate_distance_sensor(self, box_w, box_h):
        """
        使用目标尺寸估计距离（如果没有实际的测距传感器）
        
        参数:
        box_w, box_h -- 检测框的宽度和高度（像素）
        
        返回:
        estimated_distance -- 估计的距离（厘米）
        """
        # 基于检测框大小估计距离
        # 假设：距离与检测框大小成反比，需要标定获得更准确的关系
        book_width_pixels = box_w
        book_height_pixels = box_h
        
        # 通过已知的书本实际尺寸和像素尺寸估计距离
        distance_from_width = self.average_book_size["width"] * self.frame_width / (book_width_pixels * self.pixel_to_cm_ratio)
        distance_from_height = self.average_book_size["height"] * self.frame_height / (book_height_pixels * self.pixel_to_cm_ratio)
        
        # 取平均值
        return (distance_from_width + distance_from_height) / 2
    
    def read_distance_module(self):
        """
        从距离传感器读取距离（如果有的话）
        
        返回:
        distance -- 测量的距离（厘米）
        """
        # 如果有实际的距离传感器，这里实现与传感器的通信
        # 由于目前没有实际的传感器，返回一个模拟值
        return 40.0  # 默认距离值
    
    def convert_to_arm_coordinates(self, x_img, y_img, distance, camera_height=1.0):
        """
        将图像坐标转换为机械臂坐标系
        
        参数:
        x_img, y_img -- 图像中的目标坐标（像素）
        distance -- 从摄像头到目标的距离（厘米）
        camera_height -- 摄像头距离末端执行器（灯罩）的高度（厘米）
        
        返回:
        x_arm, y_arm, z_arm -- 机械臂坐标系中的坐标（厘米）
        """
        # 计算图像中心点的偏移
        dx_pixels = x_img - self.frame_center[0]
        dy_pixels = y_img - self.frame_center[1]
        
        # 将像素偏移转换为实际距离偏移（厘米）
        # 这里使用简化的针孔相机模型
        # 假设FOV（视场角）为60度，可以根据实际摄像头参数调整
        fov_x = 60  # 水平视场角（度）
        fov_y = 45  # 垂直视场角（度）
        
        # 根据视场角和分辨率计算像素到实际距离的比例
        angle_per_pixel_x = np.radians(fov_x) / self.frame_width
        angle_per_pixel_y = np.radians(fov_y) / self.frame_height
        
        # 计算角度偏移
        angle_x = dx_pixels * angle_per_pixel_x
        angle_y = dy_pixels * angle_per_pixel_y
        
        # 根据角度和距离计算实际偏移（厘米）
        dx_cm = np.tan(angle_x) * distance
        dy_cm = np.tan(angle_y) * distance
        
        # 考虑摄像头安装在末端执行器上，且末端执行器朝向目标
        # 需要进行坐标转换：
        # 1. 图像坐标系：x向右，y向下
        # 2. 机械臂坐标系：x向前，y向左
        
        # 获取当前机械臂末端位置和朝向
        if self.current_arm_pos is None:
            # 如果没有当前位置信息，假设机械臂正对前方
            # x_arm是摄像头朝向的方向
            x_arm = distance
            y_arm = -dx_cm  # 图像x轴向右，机械臂y轴向左
            z_arm = 0  # 假设目标在桌面上
        else:
            # 如果有当前位置信息，基于当前位置进行相对计算
            # 这里需要考虑机械臂末端的朝向
            current_x, current_y, current_z = self.current_arm_pos
            
            # 这里需要更复杂的变换，考虑机械臂末端的姿态
            # 简化处理：假设机械臂末端总是朝向目标
            base_angle = np.arctan2(current_y, current_x)
            
            # 基于当前机械臂姿态计算目标位置
            x_arm = current_x + dx_cm * np.cos(base_angle) - dy_cm * np.sin(base_angle)
            y_arm = current_y + dx_cm * np.sin(base_angle) + dy_cm * np.cos(base_angle)
            z_arm = 0  # 假设目标在桌面上
        
        return x_arm, y_arm, z_arm
    
    def calibrate_pixel_ratio(self, known_distance, known_object_width_cm, detected_width_pixels):
        """
        校准像素到厘米的比例
        
        参数:
        known_distance -- 已知的摄像头到目标的距离（厘米）
        known_object_width_cm -- 已知目标的实际宽度（厘米）
        detected_width_pixels -- 检测到的目标宽度（像素）
        """
        # 计算在给定距离下，每厘米对应的像素数
        self.pixel_to_cm_ratio = (detected_width_pixels * known_distance) / (known_object_width_cm * self.frame_width)
        print(f"像素比例校准: {self.pixel_to_cm_ratio} 像素/厘米")
    
    def set_current_arm_position(self, position):
        """
        设置当前机械臂位置
        
        参数:
        position -- (x, y, z) 位置元组
        """
        self.current_arm_pos = position
    
    def detect_and_track(self, confidence_threshold=0.5): # 默认置信度阈值
        """
        检测并跟踪书本
        
        参数:
        confidence_threshold -- 检测置信度阈值
        
        返回:
        target_coords -- 目标的机械臂坐标系坐标 (x, y, z)，如果没有检测到目标则返回None
        detection_result -- 检测框等信息
        frame -- 当前帧图像
        """
        # 从CSI摄像头捕获帧
        frame = self.picam2.capture_array()
        
        if frame is None or frame.size == 0:
            print("无法读取摄像头帧")
            return None, None, None
        
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) 
        
        # 使用YOLOv8进行目标检测
        results = self.model(frame, conf=confidence_threshold, imgsz=320)
        
        best_target = None
        highest_conf = 0
        detection_info = None
        
        # 处理检测结果
        if len(results) > 0:
            # 获取第一个结果的所有检测框
            boxes = results[0].boxes
            
            # 遍历所有检测到的目标
            for i, box in enumerate(boxes):
                # 获取类别
                cls_id = int(box.cls.item())
                cls_name = results[0].names[cls_id]
                
                # 判断是否是我们关心的目标类别
                if cls_name.lower() in [c.lower() for c in self.target_classes]:
                    # 获取置信度
                    conf = float(box.conf.item())
                    
                    # 获取边界框坐标
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    
                    # 计算边界框中心点
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    
                    # 计算边界框尺寸
                    box_width = x2 - x1
                    box_height = y2 - y1
                    
                    # 如果这是目前最高置信度的目标，则更新
                    if conf > highest_conf:
                        highest_conf = conf
                        
                        # 获取距离信息
                        if self.use_distance_module:
                            distance = self.read_distance_module()
                        else:
                            distance = self.simulate_distance_sensor(box_width, box_height)
                        
                        # 转换为机械臂坐标
                        arm_coords = self.convert_to_arm_coordinates(center_x, center_y, distance)
                        
                        best_target = arm_coords
                        detection_info = {
                            "class": cls_name,
                            "confidence": conf,
                            "box": (x1, y1, x2, y2),
                            "center": (center_x, center_y),
                            "distance": distance
                        }
                        
                        # 在图像上标记目标
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
                        cv2.putText(frame, f"{cls_name}: {conf:.2f}, D: {distance:.1f}cm", 
                                   (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 在图像上显示相机中心点
        cv2.circle(frame, self.frame_center, 10, (255, 0, 0), 2)
        cv2.line(frame, (self.frame_center[0]-15, self.frame_center[1]), 
                (self.frame_center[0]+15, self.frame_center[1]), (255, 0, 0), 2)
        cv2.line(frame, (self.frame_center[0], self.frame_center[1]-15), 
                (self.frame_center[0], self.frame_center[1]+15), (255, 0, 0), 2)
        
        if best_target:
            x_arm, y_arm, z_arm = best_target
            cv2.putText(frame, f"Arm Coords: ({x_arm:.1f}, {y_arm:.1f}, {z_arm:.1f})", 
                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return best_target, detection_info, frame
    
    def release(self):
        """释放资源"""
        self.picam2.stop()  # 停止摄像头
        self.picam2.close() # 关闭摄像头
        cv2.destroyAllWindows()

# 示例使用方法
if __name__ == "__main__":
    detector = BookDetector(camera_index=0)
    
    try:
        while True:
            target_coords, detection_info, frame = detector.detect_and_track()
            
            if frame is not None:
                cv2.imshow("Book Tracking", frame)
            
            if target_coords:
                print(f"目标坐标: {target_coords}, 距离: {detection_info['distance']:.1f}cm")
            
            # 按ESC退出
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
    
    finally:
        detector.release()
