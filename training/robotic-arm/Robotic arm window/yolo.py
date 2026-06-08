import cv2
import torch
import numpy as np
import time
from ultralytics import YOLO

class ObjectDetector:
    def __init__(self, camera_index=0, model_path="yolov8n.pt", camera_matrix=None, distance_module=False):
        """
        初始化目标检测器
        
        参数:
        camera_index -- 摄像头索引，默认为0
        model_path -- YOLOv8模型路径，默认使用yolov8n
        camera_matrix -- 摄像头内参矩阵，如果有标定结果的话
        distance_module -- 是否使用测距模块
        """
        # 初始化摄像头
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise ValueError(f"无法打开摄像头 {camera_index}")
        
        # 获取摄像头分辨率
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_center = (self.frame_width // 2, self.frame_height // 2)
        
        print(f"摄像头分辨率: {self.frame_width}x{self.frame_height}")
        
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
        
        # 目标尺寸参考值（厘米），用于估计距离
        self.reference_sizes = {
            "book": {"width": 20, "height": 25},
            "face": {"width": 16, "height": 24},
            "person": {"width": 40, "height": 170}
        }
        
        # 图像缩放比例：像素/厘米（需要标定）
        self.pixel_to_cm_ratio = 10  # 默认值，需要通过标定获得更准确的值
        
        # 设置关注的标签
        self.all_classes = ['book', 'face', 'person', 'cell phone', 'laptop']
        
        # 当前目标类型
        self.target_classes = self.all_classes
        
        # 超声波测距模块读取的原始距离
        self.ultrasonic_distance = None
    
    def set_target_classes(self, classes):
        """
        设置需要检测的目标类型
        
        参数:
        classes -- 目标类型列表或单个类型字符串
        """
        if isinstance(classes, str):
            classes = [classes]
        
        # 检查类别是否在支持列表中
        for cls in classes:
            if cls not in self.all_classes and cls != 'all':
                print(f"警告：不支持的类别 '{cls}'")
        
        if 'all' in classes:
            self.target_classes = self.all_classes
        else:
            self.target_classes = classes
        
        print(f"设置目标检测类别为: {self.target_classes}")
    
    def estimate_distance(self, box_w, box_h, object_type="book"):
        """
        使用目标尺寸估计距离（如果没有实际的测距传感器）
        
        参数:
        box_w, box_h -- 检测框的宽度和高度（像素）
        object_type -- 目标类型，用于获取参考尺寸
        
        返回:
        estimated_distance -- 估计的距离（厘米）
        """
        # 获取参考尺寸
        if object_type not in self.reference_sizes:
            object_type = "book"  # 默认使用书本尺寸
        
        ref_width = self.reference_sizes[object_type]["width"]
        ref_height = self.reference_sizes[object_type]["height"]
        
        # 计算距离
        distance_from_width = ref_width * self.frame_width / (box_w * self.pixel_to_cm_ratio)
        distance_from_height = ref_height * self.frame_height / (box_h * self.pixel_to_cm_ratio)
        
        # 取平均值（更稳定）
        return (distance_from_width + distance_from_height) / 2
    
    def update_ultrasonic_distance(self, distance):
        """
        更新超声波测距模块读取的距离
        
        参数:
        distance -- 超声波测距值（厘米）
        """
        self.ultrasonic_distance = distance
    
    def get_distance(self, box_w, box_h, object_type="book"):
        """
        获取距离信息，优先使用超声波测量值
        
        参数:
        box_w, box_h -- 检测框的宽度和高度（像素）
        object_type -- 目标类型
        
        返回:
        distance -- 距离（厘米）
        """
        if self.use_distance_module and self.ultrasonic_distance is not None:
            # 使用超声波测量值
            return self.ultrasonic_distance
        else:
            # 通过图像估计距离
            return self.estimate_distance(box_w, box_h, object_type)
    
    def convert_to_arm_coordinates(self, x_img, y_img, distance, camera_offset=5.5):
        """
        将图像坐标转换为机械臂坐标系
        
        参数:
        x_img, y_img -- 图像中的目标坐标（像素）
        distance -- 从末端执行器到目标的距离（厘米）
        camera_offset -- 摄像头相对于末端执行器中心的高度偏移（厘米）
        
        返回:
        x_arm, y_arm, z_arm -- 机械臂坐标系中的坐标（厘米）
        """
        # 计算图像中心点的偏移
        dx_pixels = x_img - self.frame_center[0]
        dy_pixels = y_img - self.frame_center[1]
        
        # 将像素偏移转换为实际距离偏移（厘米）
        # 使用针孔相机模型和视场角计算
        fov_x = 60  # 水平视场角（度）- 需要根据实际摄像头调整
        fov_y = 45  # 垂直视场角（度）- 需要根据实际摄像头调整
        
        # 根据视场角和分辨率计算像素到角度的比例
        angle_per_pixel_x = np.radians(fov_x) / self.frame_width
        angle_per_pixel_y = np.radians(fov_y) / self.frame_height
        
        # 计算角度偏移
        angle_x = dx_pixels * angle_per_pixel_x
        angle_y = dy_pixels * angle_per_pixel_y
        
        # 考虑摄像头偏移距离
        adjusted_distance = np.sqrt(distance**2 + camera_offset**2)
        
        # 根据角度和距离计算实际偏移（厘米）
        dx_cm = np.tan(angle_x) * adjusted_distance
        dy_cm = np.tan(angle_y) * adjusted_distance
        
        # 机械臂坐标系：x向前，y向左，z向上
        # 图像坐标系：x向右，y向下
        if self.current_arm_pos is None:
            # 如果没有当前位置信息，假设机械臂正对前方
            # 转换为机械臂坐标系
            x_arm = adjusted_distance
            y_arm = -dx_cm  # 图像x轴向右，机械臂y轴向左
            z_arm = 0       # 假设目标在桌面上，z=0
        else:
            # 如果有当前位置，需要考虑机械臂当前姿态
            current_x, current_y, current_z = self.current_arm_pos
            
            # 简化处理：假设机械臂末端总是朝向目标方向
            # 计算当前机械臂的底座角度
            base_angle = np.arctan2(current_y, current_x)
            
            # 使用当前底座角度转换坐标
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
    
    def detect_and_track(self, confidence_threshold=0.5):
        """
        检测并跟踪目标
        
        参数:
        confidence_threshold -- 检测置信度阈值
        
        返回:
        target_coords -- 目标的机械臂坐标系坐标 (x, y, z)，如果没有检测到目标则返回None
        detection_info -- 检测框等信息
        frame -- 当前帧图像
        """
        # 读取一帧图像
        ret, frame = self.cap.read()
        if not ret:
            print("无法读取摄像头帧")
            return None, None, None
        
        # 使用YOLOv8进行目标检测
        results = self.model(frame, conf=confidence_threshold)
        
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
                        distance = self.get_distance(box_width, box_height, cls_name)
                        
                        # 转换为机械臂坐标
                        arm_coords = self.convert_to_arm_coordinates(center_x, center_y, distance)
                        
                        best_target = arm_coords
                        detection_info = {
                            "class": cls_name,
                            "confidence": conf,
                            "box": (x1, y1, x2, y2),
                            "center": (center_x, center_y),
                            "dimensions": (box_width, box_height),
                            "distance": distance
                        }
                        
                        # 在图像上标记目标
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
                        cv2.putText(frame, f"{cls_name}: {conf:.2f}", 
                                   (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        cv2.putText(frame, f"距离: {distance:.1f}cm", 
                                   (x1, y1 + box_height + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 在图像上显示相机中心点
        cv2.circle(frame, self.frame_center, 10, (255, 0, 0), 2)
        cv2.line(frame, (self.frame_center[0]-15, self.frame_center[1]), 
                (self.frame_center[0]+15, self.frame_center[1]), (255, 0, 0), 2)
        cv2.line(frame, (self.frame_center[0], self.frame_center[1]-15), 
                (self.frame_center[0], self.frame_center[1]+15), (255, 0, 0), 2)
        
        # 显示当前模式
        mode_text = "检测模式: " + " ".join(self.target_classes)
        cv2.putText(frame, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        
        # 显示坐标信息
        if best_target:
            x_arm, y_arm, z_arm = best_target
            cv2.putText(frame, f"机械臂坐标: ({x_arm:.1f}, {y_arm:.1f}, {z_arm:.1f})", 
                      (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        return best_target, detection_info, frame
    
    def release(self):
        """释放资源"""
        self.cap.release()
        cv2.destroyAllWindows()

# 示例使用方法
if __name__ == "__main__":
    detector = ObjectDetector(camera_index=0)
    
    # 设置只检测书本
    detector.set_target_classes(["book"])
    
    try:
        while True:
            target_coords, detection_info, frame = detector.detect_and_track()
            
            if frame is not None:
                cv2.imshow("目标跟踪", frame)
            
            if target_coords:
                print(f"目标坐标: {target_coords}, 距离: {detection_info['distance']:.1f}cm")
            
            # 按ESC退出，按'f'切换到人脸检测，按'b'切换到书本检测
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord('f'):
                detector.set_target_classes(["face"])
            elif key == ord('b'):
                detector.set_target_classes(["book"])
    
    finally:
        detector.release()
