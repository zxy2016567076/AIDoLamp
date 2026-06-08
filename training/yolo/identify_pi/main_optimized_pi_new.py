from picamera2 import Picamera2
import cv2
import time
import numpy as np
import onnxruntime as ort
import os

class YOLOv8ONNXDetector:
    def __init__(self, model_path, classes, conf_threshold=0.25, iou_threshold=0.45):
        # 验证模型文件是否存在
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件未找到: {model_path}")
            
        print(f"加载 ONNX 模型: {model_path}")
        
        # 配置 ONNX Runtime 选项以在 Raspberry Pi 上提高性能
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = 4  # 根据 Pi 型号调整 (Pi 4 有 4 核)
        
        # 初始化 ONNX Runtime 推理会话
        self.session = ort.InferenceSession(
            model_path, 
            options, 
            providers=['CPUExecutionProvider']
        )
        
        # 存储模型参数
        self.classes = classes
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # 获取模型信息
        self.get_model_info()
        
        # 初始化摄像头
        self.picam2 = Picamera2()
        self.configure_camera()
        
    def get_model_info(self):
        """提取并验证模型输入/输出信息"""
        # 输入信息
        model_inputs = self.session.get_inputs()
        self.input_name = model_inputs[0].name
        self.input_shape = model_inputs[0].shape
        self.input_width = self.input_shape[3]
        self.input_height = self.input_shape[2]
        
        # 输出信息
        model_outputs = self.session.get_outputs()
        self.output_name = model_outputs[0].name
        self.output_shape = model_outputs[0].shape
        
        # 验证模型
        if len(self.input_shape) != 4 or self.input_shape[0] != 1:
            raise ValueError(f"输入形状异常: {self.input_shape}. 期望值: [1, 3, height, width]")
        
        # 打印模型详细信息
        print("="*50)
        print("模型信息:")
        print(f"输入: {self.input_name}, 形状: {self.input_shape}")
        print(f"输出: {self.output_name}, 形状: {self.output_shape}")
        print(f"类别数: {len(self.classes)} - {self.classes}")
        print("="*50)
        
    def configure_camera(self):
        """配置并启动摄像头以适应树莓派的最佳设置"""
        # 创建适用于目标检测的配置
        config = self.picam2.create_preview_configuration(
            main={"size": (640, 480), "format": "RGB888"},
            controls={"FrameRate": 15}  # 降低帧率以提高稳定性
        )
        self.picam2.configure(config)
        self.picam2.start()
        
        # 等待摄像头初始化和调整
        print("等待摄像头初始化...")
        time.sleep(2)  # 摄像头预热时间
        print("摄像头已就绪")

    def preprocess(self, image):
        """对输入图像进行预处理以适配 YOLO 模型"""
        # 存储原始尺寸
        original_h, original_w = image.shape[:2]
        
        # 计算信箱缩放比例以保持宽高比
        scale = min(self.input_width / original_w, self.input_height / original_h)
        new_w = int(original_w * scale)
        new_h = int(original_h * scale)
        
        # 调整图像大小以保持宽高比
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # 创建信箱填充
        # 使用灰色 (114, 114, 114) 作为 YOLO 预处理的标准
        letterbox_img = np.full((self.input_height, self.input_width, 3), 114, dtype=np.uint8)
        
        # 计算偏移量以居中
        offset_x = (self.input_width - new_w) // 2
        offset_y = (self.input_height - new_h) // 2
        
        # 将调整大小的图像放置在填充的信箱画布上
        letterbox_img[offset_y:offset_y+new_h, offset_x:offset_x+new_w] = resized
        
        # 将像素值归一化到 [0, 1] 并转换为 NCHW 格式
        preprocessed = letterbox_img.astype(np.float32) / 255.0
        preprocessed = np.transpose(preprocessed, (2, 0, 1))  # HWC 转为 CHW
        preprocessed = np.expand_dims(preprocessed, axis=0)  # 添加批次维度
        
        # 保存预处理信息以供后处理使用
        preprocess_info = {
            "original_size": (original_w, original_h),
            "new_size": (new_w, new_h),
            "offset": (offset_x, offset_y),
            "scale": scale
        }
        
        return preprocessed, preprocess_info

    def postprocess(self, output, preprocess_info):
        """处理 YOLO 模型输出以提取检测结果"""
        # 提取第一个输出 (批次大小为 1)
        predictions = np.squeeze(output[0])
        
        # 获取预处理信息
        original_w, original_h = preprocess_info["original_size"]
        offset_x, offset_y = preprocess_info["offset"]
        scale = preprocess_info["scale"]
        
        # 从输出张量中提取信息
        # YOLOv8 输出格式: [x, y, w, h, confidence, class_scores...]
        
        # 根据置信度阈值过滤检测结果
        mask = predictions[:, 4] >= self.conf_threshold
        predictions = predictions[mask]
        
        if len(predictions) == 0:
            return [], [], []
            
        # 获取类别分数和 ID
        class_scores = predictions[:, 5:]
        class_ids = np.argmax(class_scores, axis=1)
        
        # 根据类别置信度过滤
        class_confidences = np.max(class_scores, axis=1)
        mask = class_confidences >= self.conf_threshold
        
        boxes = predictions[mask, :4]
        class_ids = class_ids[mask]
        confidences = class_confidences[mask]
        
        # 将框从 [x, y, w, h] 转换为 [x1, y1, x2, y2]
        # 同时移除偏移并重新缩放到原始图像尺寸
        x = boxes[:, 0]
        y = boxes[:, 1]
        w = boxes[:, 2]
        h = boxes[:, 3]
        
        # 获取信箱图像中的框坐标
        x1 = x - w/2
        y1 = y - h/2
        x2 = x + w/2
        y2 = y + h/2
        
        # 重新缩放到原始图像尺寸
        boxes_scaled = []
        for i in range(len(boxes)):
            # 从信箱图像中移除偏移
            x1_orig = (x1[i] - offset_x) / scale
            y1_orig = (y1[i] - offset_y) / scale
            x2_orig = (x2[i] - offset_x) / scale
            y2_orig = (y2[i] - offset_y) / scale
            
            # 裁剪到图像边界
            x1_orig = max(0, min(x1_orig, original_w))
            y1_orig = max(0, min(y1_orig, original_h))
            x2_orig = max(0, min(x2_orig, original_w))
            y2_orig = max(0, min(y2_orig, original_h))
            
            boxes_scaled.append([int(x1_orig), int(y1_orig), int(x2_orig), int(y2_orig)])
        
        # 执行非极大值抑制
        indices = cv2.dnn.NMSBoxes(
            boxes_scaled, 
            confidences, 
            self.conf_threshold, 
            self.iou_threshold
        )
        
        result_boxes = [boxes_scaled[i] for i in indices]
        result_confidences = [confidences[i] for i in indices]
        result_class_ids = [class_ids[i] for i in indices]
        
        return result_boxes, result_confidences, result_class_ids

    def draw_detections(self, image, boxes, confidences, class_ids):
        """在图像上绘制检测框和标签"""
        # 不同类别的颜色
        colors = [
            (0, 255, 0),    # 绿色
            (0, 0, 255),    # 红色
            (255, 0, 0),    # 蓝色
            (0, 255, 255),  # 黄色
        ]
        
        # 绘制每个检测结果
        for i, box in enumerate(boxes):
            # 获取检测信息
            x1, y1, x2, y2 = box
            confidence = confidences[i]
            class_id = class_ids[i]
            
            # 验证 class_id 是否有效
            if 0 <= class_id < len(self.classes):
                class_name = self.classes[class_id]
                color = colors[class_id % len(colors)]
            else:
                class_name = "未知"
                color = (200, 200, 200)  # 灰色表示未知类别
            
            # 绘制边框
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # 创建包含类别名称和置信度的标签
            label = f"{class_name} {confidence:.2f}"
            
            # 获取标签大小以便更好地定位
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
            
        return image

    def detect(self, image):
        """对图像运行检测并返回标注结果"""
        # 预处理图像
        preprocessed, preprocess_info = self.preprocess(image)
        
        # 运行推理
        outputs = self.session.run(None, {self.input_name: preprocessed})
        
        # 后处理结果
        boxes, confidences, class_ids = self.postprocess(outputs, preprocess_info)
        
        # 在图像上绘制检测结果
        result = self.draw_detections(image.copy(), boxes, confidences, class_ids)
        
        return result, boxes, confidences, class_ids

# 主程序执行
def main():
    # 定义类别名称 - 应与模型训练时一致
    CLASSES = ["person_a", "person_b", "person_c", "book"]
    
    # 模型路径 - 确保此路径指向您的量化模型
    MODEL_PATH = "runs/detect/my_custom_model/weights/best_quant.onnx"
    
    # 创建检测器
    try:
        detector = YOLOv8ONNXDetector(
            model_path=MODEL_PATH,
            classes=CLASSES,
            conf_threshold=0.30,  # 降低阈值以查看更多检测结果
            iou_threshold=0.45
        )
    except FileNotFoundError as e:
        print(f"错误: {e}")
        print("请检查模型文件是否存在于指定路径。")
        return
    
    # 性能跟踪
    prev_time = time.time()
    frame_count = 0
    fps_display = 0
    smoothing_factor = 0.9  # 用于 FPS 平滑
    
    try:
        print("按 'q' 键退出...")
        while True:
            # 捕获帧
            frame = detector.picam2.capture_array()
            
            # 执行检测
            result, boxes, confidences, class_ids = detector.detect(frame)
            
            # 更新 FPS 计算
            current_time = time.time()
            fps = 1 / (current_time - prev_time)
            fps_display = fps_display * smoothing_factor + fps * (1 - smoothing_factor)
            prev_time = current_time
            frame_count += 1
            
            # 显示 FPS
            cv2.putText(
                result, 
                f"FPS: {fps_display:.1f}", 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (0, 255, 0), 
                2
            )
            
            # 显示检测结果
            cv2.imshow("YOLOv8 目标检测", result)
            
            # 检查退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("用户中断")
    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        # 清理资源
        detector.picam2.stop()
        cv2.destroyAllWindows()
        
        # 报告平均 FPS
        if frame_count > 0:
            print(f"平均 FPS: {fps_display:.1f}")

if __name__ == "__main__":
    main()
