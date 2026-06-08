from picamera2 import Picamera2
import cv2
import time
import numpy as np
import onnxruntime as ort
import os
from datetime import datetime

class OptimizedDetector:
    def __init__(self, model_path, classes, conf_thresholds=None, iou_threshold=0.45, 
                 input_size=(320, 320)):  # Reduced input size for speed
        # Verify model file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件未找到: {model_path}")
            
        print(f"正在加载ONNX模型: {model_path}")
        
        # Configure ONNX Runtime for Raspberry Pi
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = 4  # 适用于4核Raspberry Pi
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL  # 顺序执行以降低内存占用
        
        # Initialize ONNX Runtime 
        self.session = ort.InferenceSession(
            model_path, 
            options, 
            providers=['CPUExecutionProvider']
        )
        
        # Store model parameters
        self.classes = classes
        self.input_size = input_size
        self.iou_threshold = iou_threshold
        
        # Class-specific confidence thresholds
        if conf_thresholds is None:
            # Default confidence thresholds (higher for 'book' class)
            self.conf_thresholds = {
                'person_a': 0.30,
                'person_b': 0.30,
                'person_c': 0.30,
                'book': 0.45  # 给书本类使用更高的阈值以减少误检测
            }
        else:
            self.conf_thresholds = conf_thresholds
            
        # 假阳性识别和跟踪
        self.book_detections = []  # 保存历史book检测结果用于分析
        self.max_history = 10      # 最大历史记录数
            
        # Get model info
        self.get_model_info()
        
        # Performance metrics
        self.inference_times = []
        self.preprocess_times = []
        self.postprocess_times = []
        
        # Initialize camera
        self.picam2 = Picamera2()
        self.configure_camera()
        
    def get_model_info(self):
        """提取和验证模型输入/输出信息"""
        # Input details
        model_inputs = self.session.get_inputs()
        self.input_name = model_inputs[0].name
        self.input_shape = model_inputs[0].shape
        self.input_width = self.input_size[0]
        self.input_height = self.input_size[1]
        
        # Output details
        model_outputs = self.session.get_outputs()
        self.output_name = model_outputs[0].name
        self.output_shape = model_outputs[0].shape
        
        # Log model details
        print("="*50)
        print("模型信息:")
        print(f"输入: {self.input_name}, 形状: {self.input_shape}")
        print(f"输出: {self.output_name}, 形状: {self.output_shape}")
        print(f"类别: {len(self.classes)} - {self.classes}")
        print(f"类别阈值: {self.conf_thresholds}")
        print("="*50)
        
    def configure_camera(self):
        """配置相机，优化树莓派性能"""
        # 为对象检测创建配置
        config = self.picam2.create_preview_configuration(
            main={"size": (640, 480), "format": "RGB888"},
            controls={
                "FrameRate": 15,  # 降低帧率以提高稳定性
                "FrameDurationLimits": (33333, 83333),  # 帧率限制 (12-30 FPS)
                "ExposureTime": 10000,  # 曝光时间，更短的曝光可以减少运动模糊
                "AnalogueGain": 1.0     # 增益设置
            }
        )
        self.picam2.configure(config)
        self.picam2.start()
        
        # 相机初始化等待
        print("等待相机初始化...")
        time.sleep(1.5)  # 相机预热时间
        print("相机准备就绪")

    def preprocess(self, image):
        """优化的图像预处理"""
        start_time = time.time()
        
        # 存储原始尺寸
        original_h, original_w = image.shape[:2]
        
        # 计算信箱比例以保持纵横比
        scale = min(self.input_width / original_w, self.input_height / original_h)
        new_w = int(original_w * scale)
        new_h = int(original_h * scale)
        
        # 调整图像大小，同时保持纵横比
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # 创建信箱填充
        # 使用灰色(114, 114, 114)作为YOLO预处理标准
        letterbox_img = np.full((self.input_height, self.input_width, 3), 114, dtype=np.uint8)
        
        # 计算偏移量使图像居中
        offset_x = (self.input_width - new_w) // 2
        offset_y = (self.input_height - new_h) // 2
        
        # 将调整大小的图像放在填充画布上
        letterbox_img[offset_y:offset_y+new_h, offset_x:offset_x+new_w] = resized
        
        # 归一化像素值并转换为NCHW格式
        preprocessed = letterbox_img.astype(np.float32) / 255.0
        preprocessed = np.transpose(preprocessed, (2, 0, 1))  # HWC to CHW
        preprocessed = np.expand_dims(preprocessed, axis=0)   # 添加批次维度
        
        # 存储预处理信息以供后处理使用
        preprocess_info = {
            "original_size": (original_w, original_h),
            "new_size": (new_w, new_h),
            "offset": (offset_x, offset_y),
            "scale": scale
        }
        
        self.preprocess_times.append(time.time() - start_time)
        if len(self.preprocess_times) > 30:
            self.preprocess_times.pop(0)
            
        return preprocessed, preprocess_info

    def postprocess(self, output, preprocess_info):
        """针对树莓派优化的后处理逻辑"""
        start_time = time.time()
        
        # 提取第一个输出(批次大小为1)
        predictions = np.squeeze(output[0])
        
        # 获取预处理信息
        original_w, original_h = preprocess_info["original_size"]
        offset_x, offset_y = preprocess_info["offset"]
        scale = preprocess_info["scale"]
        
        # 根据信心阈值过滤检测
        mask = predictions[:, 4] >= min(self.conf_thresholds.values())
        predictions = predictions[mask]
        
        if len(predictions) == 0:
            self.postprocess_times.append(time.time() - start_time)
            if len(self.postprocess_times) > 30:
                self.postprocess_times.pop(0)
            return [], [], [], []
            
        # 获取类别分数和ID
        class_scores = predictions[:, 5:]
        class_ids = np.argmax(class_scores, axis=1)
        class_confidences = np.max(class_scores, axis=1)
        
        boxes = []
        confidences = []
        filtered_class_ids = []
        detection_metrics = []
        
        # 应用类别特定的阈值和额外过滤逻辑
        for i, class_id in enumerate(class_ids):
            if 0 <= class_id < len(self.classes):
                class_name = self.classes[class_id]
                conf_threshold = self.conf_thresholds.get(class_name, 0.3)
                
                # 应用类别特定的阈值
                if class_confidences[i] >= conf_threshold:
                    # 计算边界框坐标
                    x, y, w, h = predictions[i, :4]
                    
                    # 获取信箱图像中的框坐标
                    x1 = x - w/2
                    y1 = y - h/2
                    x2 = x + w/2
                    y2 = y + h/2
                    
                    # 转回原始图像尺寸
                    x1_orig = (x1 - offset_x) / scale
                    y1_orig = (y1 - offset_y) / scale
                    x2_orig = (x2 - offset_x) / scale
                    y2_orig = (y2 - offset_y) / scale
                    
                    # 裁剪到图像边界
                    x1_orig = max(0, min(x1_orig, original_w))
                    y1_orig = max(0, min(y1_orig, original_h))
                    x2_orig = max(0, min(x2_orig, original_w))
                    y2_orig = max(0, min(y2_orig, original_h))
                    
                    # 计算长宽比和面积比例用于book类额外过滤
                    width = x2_orig - x1_orig
                    height = y2_orig - y1_orig
                    
                    # 计算一些检测指标用于过滤假阳性
                    aspect_ratio = width / height if height > 0 else 0
                    area_ratio = (width * height) / (original_w * original_h)
                    
                    # 特别处理book类，减少误报
                    if class_name == 'book':
                        # 书本通常有特定的长宽比，不会太窄或太宽
                        if 0.5 <= aspect_ratio <= 2.0:
                            # 确保book不是微小物体(假阳性更可能出现在小边界框)
                            if area_ratio > 0.01:
                                boxes.append([int(x1_orig), int(y1_orig), 
                                              int(x2_orig), int(y2_orig)])
                                confidences.append(float(class_confidences[i]))
                                filtered_class_ids.append(class_id)
                                detection_metrics.append({
                                    'aspect_ratio': aspect_ratio,
                                    'area_ratio': area_ratio
                                })
                    else:
                        # 非book类的正常处理
                        boxes.append([int(x1_orig), int(y1_orig), 
                                      int(x2_orig), int(y2_orig)])
                        confidences.append(float(class_confidences[i]))
                        filtered_class_ids.append(class_id)
                        detection_metrics.append({
                            'aspect_ratio': aspect_ratio,
                            'area_ratio': area_ratio
                        })
        
        # 保存book检测结果用于分析
        self.track_book_detections(boxes, confidences, filtered_class_ids)
        
        # 执行非最大抑制
        if boxes:
            indices = cv2.dnn.NMSBoxes(
                boxes, 
                confidences, 
                min(self.conf_thresholds.values()), 
                self.iou_threshold
            )
            
            result_boxes = [boxes[i] for i in indices]
            result_confidences = [confidences[i] for i in indices]
            result_class_ids = [filtered_class_ids[i] for i in indices]
            result_metrics = [detection_metrics[i] for i in indices]
        else:
            result_boxes = []
            result_confidences = []
            result_class_ids = []
            result_metrics = []
        
        self.postprocess_times.append(time.time() - start_time)
        if len(self.postprocess_times) > 30:
            self.postprocess_times.pop(0)
            
        return result_boxes, result_confidences, result_class_ids, result_metrics
        
    def track_book_detections(self, boxes, confidences, class_ids):
        """追踪book检测以分析假阳性"""
        for i, class_id in enumerate(class_ids):
            if 0 <= class_id < len(self.classes) and self.classes[class_id] == 'book':
                self.book_detections.append({
                    'time': time.time(),
                    'confidence': confidences[i],
                    'box': boxes[i]
                })
                
        # 限制历史记录大小
        if len(self.book_detections) > self.max_history:
            self.book_detections.pop(0)

    def draw_detections(self, image, boxes, confidences, class_ids, metrics):
        """在图像上绘制检测框和标签"""
        # 不同类别的颜色
        colors = [
            (0, 255, 0),    # 绿色 - person_a
            (0, 0, 255),    # 红色 - person_b
            (255, 0, 0),    # 蓝色 - person_c
            (0, 255, 255),  # 黄色 - book
        ]
        
        # 绘制每个检测
        for i, box in enumerate(boxes):
            # 获取检测信息
            x1, y1, x2, y2 = box
            confidence = confidences[i]
            class_id = class_ids[i]
            
            # 验证class_id是否有效
            if 0 <= class_id < len(self.classes):
                class_name = self.classes[class_id]
                color = colors[class_id % len(colors)]
            else:
                class_name = "未知"
                color = (200, 200, 200)  # 未知类别用灰色
            
            # 绘制边界框
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # 创建带有类名和置信度的标签
            label = f"{class_name} {confidence:.2f}"
            
            # 获取标签大小以便更好的定位
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            
            # 绘制标签背景
            cv2.rectangle(
                image, 
                (x1, y1 - label_height - 10), 
                (x1 + label_width, y1), 
                color, 
                -1
            )
            
            # 绘制标签文本
            cv2.putText(
                image, 
                label, 
                (x1, y1 - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (255, 255, 255), 
                2
            )
            
            # 对于book类显示更多的信息(调试用)
            if class_name == 'book' and metrics and i < len(metrics):
                ar_text = f"AR: {metrics[i]['aspect_ratio']:.2f}"
                area_text = f"Area: {metrics[i]['area_ratio']:.3f}"
                
                cv2.putText(
                    image,
                    ar_text,
                    (x1, y2 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    color,
                    1
                )
                
                cv2.putText(
                    image,
                    area_text,
                    (x1, y2 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    color,
                    1
                )
            
        return image

    def draw_performance_info(self, image, fps, frame_count):
        """在图像上绘制性能信息"""
        # 计算平均处理时间
        avg_preprocess = sum(self.preprocess_times) / max(len(self.preprocess_times), 1) * 1000
        avg_inference = sum(self.inference_times) / max(len(self.inference_times), 1) * 1000
        avg_postprocess = sum(self.postprocess_times) / max(len(self.postprocess_times), 1) * 1000
        total_time = avg_preprocess + avg_inference + avg_postprocess
        
        # 绘制半透明背景
        overlay = image.copy()
        cv2.rectangle(overlay, (5, 5), (250, 130), (0, 0, 0), -1)
        alpha = 0.6
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
        
        # 绘制性能指标
        cv2.putText(image, f"FPS: {fps:.1f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.putText(image, f"预处理: {avg_preprocess:.1f}ms", (10, 55), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.putText(image, f"推理: {avg_inference:.1f}ms", (10, 75), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.putText(image, f"后处理: {avg_postprocess:.1f}ms", (10, 95), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.putText(image, f"总处理: {total_time:.1f}ms", (10, 115), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 添加日期时间
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(image, current_time, (image.shape[1] - 190, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
        # 绘制阈值信息
        cv2.putText(image, f"Book阈值: {self.conf_thresholds['book']:.2f}", 
                    (10, image.shape[0] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    
        # 显示总帧数
        cv2.putText(image, f"帧数: {frame_count}", 
                    (image.shape[1] - 100, image.shape[0] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return image

    def detect(self, image):
        """在图像上运行检测并返回标注结果"""
        # 预处理图像
        preprocessed, preprocess_info = self.preprocess(image)
        
        # 运行推理
        start_time = time.time()
        outputs = self.session.run(None, {self.input_name: preprocessed})
        inference_time = time.time() - start_time
        
        self.inference_times.append(inference_time)
        if len(self.inference_times) > 30:
            self.inference_times.pop(0)
        
        # 后处理结果
        boxes, confidences, class_ids, metrics = self.postprocess(outputs, preprocess_info)
        
        # 在图像上绘制检测结果
        result = self.draw_detections(image.copy(), boxes, confidences, class_ids, metrics)
        
        return result, boxes, confidences, class_ids, metrics

# 主执行函数
def main():
    # 定义类名 - 应与模型训练匹配
    CLASSES = ["person_a", "person_b", "person_c", "book"]
    
    # 模型路径 - 确保路径指向你的模型文件
    # 如果没有onnx模型，可以先使用pt模型，再基于README_optimization.md中的建议转换
    MODEL_PATH = "runs/detect/my_custom_model/weights/best_quant.onnx"
    
    # 如果找不到onnx模型，尝试使用PT模型
    if not os.path.exists(MODEL_PATH):
        MODEL_PATH = "best.pt"
        print(f"找不到ONNX模型，将使用PyTorch模型: {MODEL_PATH}")
        
        # 在这种情况下，我们需要使用不同的方法
        if os.path.exists(MODEL_PATH):
            print("请先将PyTorch模型转换为ONNX以获得更好的性能")
            print("你可以按照README_optimization.md中的说明进行操作")
        else:
            print("无法找到任何可用模型!")
            return
    
    # 类别特定的置信度阈值
    conf_thresholds = {
        'person_a': 0.25,
        'person_b': 0.25,
        'person_c': 0.25,
        'book': 0.45  # 书本使用更高的阈值
    }
    
    # 创建检测器
    try:
        # 使用较小的输入尺寸以提高速度 (320x320 代替 640x640)
        detector = OptimizedDetector(
            model_path=MODEL_PATH,
            classes=CLASSES,
            conf_thresholds=conf_thresholds,
            iou_threshold=0.45,
            input_size=(320, 320)  # 较小的尺寸以提高速度
        )
    except FileNotFoundError as e:
        print(f"错误: {e}")
        print("请检查模型文件是否存在于指定路径。")
        return
    
    # 性能跟踪
    prev_time = time.time()
    frame_count = 0
    fps_display = 0
    smoothing_factor = 0.9  # FPS平滑因子
    
    # 键盘控制
    book_threshold_delta = 0.05  # 阈值调整步长
    window_width = 640  # 初始窗口宽度
    window_height = 480  # 初始窗口高度
    
    try:
        print("按'q'退出, 按'+'增加book阈值, 按'-'降低book阈值...")
        print("按'w'增大窗口尺寸, 按's'减小窗口尺寸...")
        
        # 创建并设置窗口
        cv2.namedWindow("YOLOv8优化检测", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("YOLOv8优化检测", window_width, window_height)
        
        while True:
            # 捕获帧
            frame = detector.picam2.capture_array()
            
            # 检查帧是否为空或异常
            if frame is None or frame.size == 0:
                print("警告：捕获的帧为空，重试...")
                time.sleep(0.1)
                continue
            
            # 检查帧的形状是否正确
            if frame.shape[0] == 0 or frame.shape[1] == 0:
                print(f"警告：帧大小异常: {frame.shape}，重试...")
                time.sleep(0.1)
                continue
                
            # 打印帧的尺寸（仅在头几帧）
            if frame_count < 3:
                print(f"帧尺寸: {frame.shape}")
            
            # 执行检测
            result, boxes, confidences, class_ids, metrics = detector.detect(frame)
            
            # 计算FPS
            current_time = time.time()
            fps = 1 / (current_time - prev_time)
            fps_display = fps_display * smoothing_factor + fps * (1 - smoothing_factor)
            prev_time = current_time
            frame_count += 1
            
            # 显示性能信息
            result = detector.draw_performance_info(result, fps_display, frame_count)
            
            # 显示检测结果
            cv2.namedWindow("YOLOv8优化检测", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("YOLOv8优化检测", 640, 480)
            cv2.imshow("YOLOv8优化检测", result)
            
            # 检查退出和控制键
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('+') or key == ord('='):
                # 增加book阈值
                detector.conf_thresholds['book'] = min(0.95, 
                                                      detector.conf_thresholds['book'] + book_threshold_delta)
                print(f"Book阈值增加到: {detector.conf_thresholds['book']:.2f}")
            elif key == ord('-') or key == ord('_'):
                # 降低book阈值
                detector.conf_thresholds['book'] = max(0.05, 
                                                      detector.conf_thresholds['book'] - book_threshold_delta)
                print(f"Book阈值降低到: {detector.conf_thresholds['book']:.2f}")
                
    except KeyboardInterrupt:
        print("用户中断")
    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        # 清理
        detector.picam2.stop()
        cv2.destroyAllWindows()
        
        # 报告平均FPS
        if frame_count > 0:
            print(f"平均FPS: {fps_display:.1f}")
            print(f"处理帧数: {frame_count}")

if __name__ == "__main__":
    main()
