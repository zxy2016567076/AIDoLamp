import cv2
import torch
import numpy as np
import time
from ultralytics import YOLO
from picamera2 import Picamera2
from serial_comm import SerialCommunicator  # 导入串口通信模块

class YOLODetector:
    def __init__(self, camera_instance=None, model_path="best.pt"):
        
        # 摄像头设置
        self.picam2 = camera_instance
        self.camera_configured = False
        self.camera_owned = camera_instance is None  # 是否为自己创建的摄像头
        
        # 如果没有传递摄像头实例，则不会自动初始化摄像头
        # 避免与主程序的摄像头冲突
        
        # 默认参数设置
        self.frame_width = 320
        self.frame_height = 240
        self.frame_center = (self.frame_width // 2, self.frame_height // 2)
        
        # 如果有摄像头实例，则获取其尺寸参数
        if self.picam2 is not None and hasattr(self.picam2, 'camera_config'):
            try:
                self.frame_width = self.picam2.camera_config["main"]["size"][0]
                self.frame_height = self.picam2.camera_config["main"]["size"][1]
                self.frame_center = (self.frame_width // 2, self.frame_height // 2)
                print(f"使用外部摄像头，分辨率: {self.frame_width}x{self.frame_height}")
            except Exception as e:
                print(f"获取摄像头分辨率时出错: {e}")
        
        # 加载YOLOv8模型
        try:
            self.model = YOLO(model_path)
            print(f"成功加载模型: {model_path}")
            # 获取模型支持的类别
            self.class_names = self.model.names
            print(f"模型支持的类别: {self.class_names}")
        except Exception as e:
            raise ValueError(f"加载模型失败: {e}")
        
        # 初始化串口通信（用于获取超声波距离）
        try:
            self.serial_comm = SerialCommunicator()
            self.use_distance_module = True
            print("串口通信模块已初始化")
        except Exception as e:
            print(f"串口通信初始化失败，将使用估计距离: {e}")
            self.serial_comm = None
            self.use_distance_module = False
        
        # 面部和书本的默认尺寸（厘米），用于估计距离
        self.average_sizes = {
            "face": {"width": 15, "height": 20},  # 人脸平均尺寸
            "book": {"width": 18, "height": 26}   # 书本平均尺寸
        }
        
        # 图像缩放比例：像素/厘米（需要标定）
        self.pixel_to_cm_ratio = 10  # 默认值
        
        # 机械臂当前位置
        self.current_arm_pos = None
        
        # 视场角设置
        self.fov_x = 60  # 水平视场角（度）
        self.fov_y = 45  # 垂直视场角（度）
        
        # 书本专用参数
        self.book_target_distance = 45  # 书本跟踪的目标距离（厘米）
        
        # 人脸专用参数
        self.face_target_distance = 40  # 人脸跟踪的目标距离（厘米）
    
    def get_distance(self):
        
        try:
            if self.serial_comm and self.use_distance_module:
                # 尝试从串口读取超声波距离
                distance_value = self.serial_comm.last_distance
                if distance_value is not None:
                    return distance_value
        except Exception as e:
            print(f"读取距离失败: {e}")
        
        # 如果无法获取实际距离，返回一个合理的默认值
        return 45.0  # 默认距离
    
    def simulate_distance(self, obj_type, box_w, box_h):
        
        if obj_type not in self.average_sizes:
            obj_type = 'book'  # 默认使用书本尺寸
        
        # 获取对象的平均尺寸
        avg_size = self.average_sizes[obj_type]
        
        # 通过已知的对象实际尺寸和像素尺寸估计距离
        distance_from_width = avg_size["width"] * self.frame_width / (box_w * self.pixel_to_cm_ratio)
        distance_from_height = avg_size["height"] * self.frame_height / (box_h * self.pixel_to_cm_ratio)
        
        # 取平均值
        return (distance_from_width + distance_from_height) / 2
    
    def _ensure_camera(self):
        if self.picam2 is None:
            try:
                # 延迟初始化自己的摄像头
                self.picam2 = Picamera2(0)
                self.camera_owned = True
                print("YOLO检测器初始化自己的摄像头")
            except Exception as e:
                print(f"初始化摄像头失败: {e}")
                return False
                
        if not self.camera_configured:
            try:
                # 只有自己创建的摄像头才需要配置和启动
                if self.camera_owned:
                    config = self.picam2.create_preview_configuration(
                        main={
                            "size": (320, 240),
                            "format": "BGR888"
                        }
                    )
                    self.picam2.configure(config)
                    
                    # 更新分辨率信息
                    self.frame_width = self.picam2.camera_config["main"]["size"][0]
                    self.frame_height = self.picam2.camera_config["main"]["size"][1]
                    self.frame_center = (self.frame_width // 2, self.frame_height // 2)
                    
                    # 如果是自己的摄像头，则启动它
                    if not self.picam2.is_running:
                        self.picam2.start()
                    
                    print(f"YOLO检测器摄像头配置完成，分辨率: {self.frame_width}x{self.frame_height}")
                
                self.camera_configured = True
                return True
            except Exception as e:
                print(f"配置摄像头失败: {e}")
                return False
        return True

    def detect(self, frame, conf_threshold=0.5, target_class=None):
        if frame is None or frame.size == 0:
            return {}
        
        # 确保图像格式正确
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            # 使用YOLOv8进行目标检测
            results = self.model(frame, conf=conf_threshold, imgsz=320)
            
            results_dict = {}
            
            if len(results) > 0:
                # 获取第一个结果的所有检测框
                boxes = results[0].boxes
                
                # 遍历所有检测到的目标
                for i, box in enumerate(boxes):
                    # 获取类别
                    cls_id = int(box.cls.item())
                    cls_name = results[0].names[cls_id]
                    
                    # 如果指定了目标类别，则仅处理该类别
                    if target_class is not None and cls_name.lower() != target_class.lower():
                        continue
                    
                    # 获取置信度
                    conf = float(box.conf.item())
                    
                    # 获取边界框坐标
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    
                    # 计算宽度和高度
                    width = x2 - x1
                    height = y2 - y1
                    
                    # 计算中心点
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    
                    # 将结果添加到字典
                    results_dict[cls_name] = [center_x, center_y, width, height]
            
            return results_dict
        
        return {}
    
    def convert_to_face_coordinates(self, x_img, y_img, width, height):
        # 获取距离信息（优先使用实际测量，否则基于人脸尺寸估计）
        if self.use_distance_module:
            distance = self.get_distance()
        else:
            distance = self.simulate_distance('face', width, height)
        
        # 计算图像中心点的偏移
        dx_pixels = x_img - self.frame_center[0]
        dy_pixels = y_img - self.frame_center[1]
        
        # 将像素偏移转换为实际距离偏移（厘米）
        # 使用简化的针孔相机模型和视场角
        angle_per_pixel_x = np.radians(self.fov_x) / self.frame_width
        angle_per_pixel_y = np.radians(self.fov_y) / self.frame_height
        
        # 计算角度偏移
        angle_x = dx_pixels * angle_per_pixel_x
        angle_y = dy_pixels * angle_per_pixel_y
        
        # 计算实际X和Y偏移（厘米）
        x_offset = np.tan(angle_x) * distance
        y_offset = np.tan(angle_y) * distance
        
        # 面对摄像头的坐标系：X轴向前，Y轴向左，Z轴向上
        # face.py期望的坐标系：(x, y, z)
        # 由于摄像头安装在机械臂上，我们把摄像头朝向作为X轴正方向
        x = distance
        y = -x_offset  # 图像坐标系中右为正，机械臂坐标系中左为正Y
        z = -y_offset  # 图像坐标系中下为正，机械臂坐标系中上为正Z
        
        # 避免Z坐标为负（低于桌面）
        z = max(z, 0)
        
        return (x, y, z)
    
    def convert_to_book_coordinates(self, x_img, y_img, width, height):
        # 获取距离信息（优先使用实际测量，否则基于书本尺寸估计）
        if self.use_distance_module:
            distance = self.get_distance()
        else:
            distance = self.simulate_distance('book', width, height)
        
        # 计算图像中心点的偏移
        dx_pixels = x_img - self.frame_center[0]
        dy_pixels = y_img - self.frame_center[1]
        
        # 将像素偏移转换为实际距离偏移（厘米）
        angle_per_pixel_x = np.radians(self.fov_x) / self.frame_width
        angle_per_pixel_y = np.radians(self.fov_y) / self.frame_height
        
        # 计算角度偏移
        angle_x = dx_pixels * angle_per_pixel_x
        angle_y = dy_pixels * angle_per_pixel_y
        
        # 计算实际X和Y偏移（厘米）
        x_offset = np.tan(angle_x) * distance
        y_offset = np.tan(angle_y) * distance
        
        # 书本通常在桌面上，因此Z坐标设为0
        # book.py期望的坐标系：(x, y, z)，其中z一般为0（桌面高度）
        x = distance
        y = -x_offset  # 图像坐标系中右为正，机械臂坐标系中左为正Y
        z = 0          # 假设书本在桌面上
        
        return (x, y, z)
    
    def detect_face(self, frame=None, draw_results=False):
        if frame is None:
            # 确保摄像头已初始化
            if not self._ensure_camera() or self.picam2 is None:
                return None
                
            try:
                frame = self.picam2.capture_array()
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            except Exception as e:
                print(f"无法从摄像头获取帧: {e}")
                return None
        
        # 仅检测人脸类别
        detections = self.detect(frame, target_class='face')
        
        face_coordinates = None
        
        if 'face' in detections:
            x, y, w, h = detections['face']
            
            # 转换为face.py所需的坐标
            face_coordinates = self.convert_to_face_coordinates(x, y, w, h)
            
            if draw_results:
                # 在图像上绘制检测结果
                x1, y1 = int(x - w/2), int(y - h/2)
                x2, y2 = int(x + w/2), int(y + h/2)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
                cv2.putText(frame, f"Face: ({face_coordinates[0]:.1f}, {face_coordinates[1]:.1f}, {face_coordinates[2]:.1f})",
                           (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        if draw_results:
            # 绘制摄像头中心
            cv2.circle(frame, self.frame_center, 10, (255, 0, 0), 2)
            cv2.line(frame, (self.frame_center[0]-15, self.frame_center[1]),
                    (self.frame_center[0]+15, self.frame_center[1]), (255, 0, 0), 2)
            cv2.line(frame, (self.frame_center[0], self.frame_center[1]-15),
                    (self.frame_center[0], self.frame_center[1]+15), (255, 0, 0), 2)
            
            return face_coordinates, frame
        
        return face_coordinates
    
    def detect_book(self, frame=None, draw_results=False):
        if frame is None:
            # 确保摄像头已初始化
            if not self._ensure_camera() or self.picam2 is None:
                return None
                
            try:
                frame = self.picam2.capture_array()
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            except Exception as e:
                print(f"无法从摄像头获取帧: {e}")
                return None
        
        # 仅检测书本类别
        detections = self.detect(frame, target_class='book')
        
        book_coordinates = None
        
        if 'book' in detections:
            x, y, w, h = detections['book']
            
            # 转换为book.py所需的坐标
            book_coordinates = self.convert_to_book_coordinates(x, y, w, h)
            
            if draw_results:
                # 在图像上绘制检测结果
                x1, y1 = int(x - w/2), int(y - h/2)
                x2, y2 = int(x + w/2), int(y + h/2)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
                cv2.putText(frame, f"Book: ({book_coordinates[0]:.1f}, {book_coordinates[1]:.1f}, {book_coordinates[2]:.1f})",
                           (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        if draw_results:
            # 绘制摄像头中心
            cv2.circle(frame, self.frame_center, 10, (255, 0, 0), 2)
            cv2.line(frame, (self.frame_center[0]-15, self.frame_center[1]),
                    (self.frame_center[0]+15, self.frame_center[1]), (255, 0, 0), 2)
            cv2.line(frame, (self.frame_center[0], self.frame_center[1]-15),
                    (self.frame_center[0], self.frame_center[1]+15), (255, 0, 0), 2)
            
            return book_coordinates, frame
        
        return book_coordinates
    
    def release(self):
        """释放资源"""
        try:
            # 只有当摄像头是由当前类创建的时候才关闭它
            if self.camera_owned and self.picam2 is not None:
                if hasattr(self.picam2, 'is_running') and self.picam2.is_running:
                    self.picam2.stop()
                if hasattr(self.picam2, 'close'):
                    self.picam2.close()
                print("YOLO检测器释放摄像头资源")
            
            # 清理窗口
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"释放资源时出错: {e}")


# 测试示例
if __name__ == "__main__":
    detector = YOLODetector(model_path="best.pt")
    
    try:
        # 1. 测试人脸检测
        print("\n==== 测试人脸检测 ====")
        cv2.namedWindow("Face Detection", cv2.WINDOW_NORMAL)
        for _ in range(30):  # 显示30帧
            coords, frame = detector.detect_face(draw_results=True)
            if coords:
                print(f"人脸坐标: {coords}")
            cv2.imshow("Face Detection", frame)
            key = cv2.waitKey(100) & 0xFF
            if key == 27:  # ESC键退出
                break
        cv2.destroyAllWindows()
        
        # 2. 测试书本检测
        print("\n==== 测试书本检测 ====")
        cv2.namedWindow("Book Detection", cv2.WINDOW_NORMAL)
        for _ in range(30):  # 显示30帧
            coords, frame = detector.detect_book(draw_results=True)
            if coords:
                print(f"书本坐标: {coords}")
            cv2.imshow("Book Detection", frame)
            key = cv2.waitKey(100) & 0xFF
            if key == 27:  # ESC键退出
                break
        cv2.destroyAllWindows()
        
    finally:
        detector.release()
