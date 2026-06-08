from picamera2 import Picamera2
import cv2
import time
import numpy as np
import onnxruntime as ort

class YOLOv8ONNXPicam:
    def __init__(self, model_path, classes):
        
        # 初始化ONNX推理会话
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.classes = classes
        
        # 获取输入名称 ✅
        self.input_name = self.session.get_inputs()[0].name  # 关键修复
        
        input_shape = self.session.get_inputs()[0].shape
        self.input_h = input_shape[2]  # 应为 640
        self.input_w = input_shape[3]  # 应为 640
        
        # 摄像头配置
        self.picam2 = Picamera2()
        self.configure_camera()
        
        # 新增模型验证 ✅
        print("="*40)
        print("模型验证信息:")
        print(f"输入名称: {self.input_name}")
        print(f"输入形状: {self.session.get_inputs()[0].shape}")
        print(f"输出形状: {self.session.get_outputs()[0].shape}")
        print(f"代码定义的类别数: {len(classes)}")
        print("="*40)
        
    def configure_camera(self):
        """ 优化摄像头配置 """
        config = self.picam2.create_preview_configuration(
            main={"size": (640, 480), "format": "RGB888"},
            controls={"FrameRate": 30}  # 降低帧率提升稳定性
        )
        self.picam2.configure(config)
        self.picam2.start()
        time.sleep(2)  # 摄像头预热

    def preprocess(self, image):
        """ 适配 640x640 输入的预处理流程 """
        h, w = image.shape[:2]
        
        # 保持宽高比的缩放（长边缩放到640）
        scale = 640 / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        resized = cv2.resize(image, (new_w, new_h))
        
        # 填充为正方形 640x640
        canvas = np.full((640, 640, 3), 114, dtype=np.uint8)
        canvas[:new_h, :new_w] = resized
        
        # 转换为模型输入格式（CHW + 归一化）
        blob = cv2.dnn.blobFromImage(canvas, 1/255.0, swapRB=True, crop=False)
        return blob, (w, h), (new_w / w, new_h / h)

    def postprocess(self, outputs, orig_size, ratios):
        """ 修正后的后处理流程 """
        outputs = np.squeeze(outputs[0])
        boxes = []
        scores = []
        class_ids = []
        
        orig_w, orig_h = orig_size
        ratio_w, ratio_h = ratios
        
        for detection in outputs:
            # 提取类别分数（假设模型输出结构为4坐标 + 4类别分数）
            cls_scores = detection[4:4 + len(self.classes)]  # ✅ 从索引4开始提取4个分数
            max_score = np.max(cls_scores)
            class_id = np.argmax(cls_scores)
            
            # 置信度直接使用类别分数最大值（若模型未包含obj_score）
            confidence = max_score
            
            if confidence < 0.5:  # 阈值过滤
                continue
            
            # 还原坐标（保持原有逻辑）
            cx = detection[0] * self.input_w / ratio_w
            cy = detection[1] * self.input_h / ratio_h
            w = detection[2] * self.input_w / ratio_w
            h = detection[3] * self.input_h / ratio_h
            
            # 边界框计算（保持原有逻辑）
            x1 = int(cx - w/2)
            y1 = int(cy - h/2)
            x2 = int(cx + w/2)
            y2 = int(cy + h/2)
            
            # 边界检查（保持原有逻辑）
            x1 = max(0, min(x1, orig_w))
            y1 = max(0, min(y1, orig_h))
            x2 = max(0, min(x2, orig_w))
            y2 = max(0, min(y2, orig_h))
            
            boxes.append([x1, y1, x2, y2])
            scores.append(float(confidence))
            class_ids.append(class_id)
        
        # NMS处理（保持原有逻辑）
        indices = cv2.dnn.NMSBoxes(boxes, scores, 0.5, 0.5)
        return (
            [boxes[i] for i in indices],
            [scores[i] for i in indices],
            [class_ids[i] for i in indices]
        )
    def draw_detections(self, image, boxes, scores, class_ids):
        """ 带错误处理的绘制方法 """
        for box, score, class_id in zip(boxes, scores, class_ids):
            # 类别ID有效性检查 ✅
            if class_id < 0 or class_id >= len(self.classes):
                label = f"unknown: {score:.2f}"
            else:
                label = f"{self.classes[class_id]}: {score:.2f}"
            
            color = (0, 255, 0)
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, 2)
            cv2.putText(image, label, (box[0], box[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return image

# -------------------------- 初始化 --------------------------
CLASSES = ["person_a", "person_b", "person_c", "book"]  # 替换为实际类别
model = YOLOv8ONNXPicam(
    model_path="best_quant.onnx",
    classes=CLASSES
)

prev_time = time.time()
frame_count = 0
total_fps = 0

try:
    print("摄像头已启动，按 Q 键退出...")
    while True:
        # 捕获帧
        frame = model.picam2.capture_array()
        
        # 预处理 + 推理
        blob, orig_size, ratios = model.preprocess(frame)
        outputs = model.session.run(None, {model.input_name: blob})
        
        # 后处理
        boxes, scores, class_ids = model.postprocess(outputs, orig_size, ratios)
        
        # 计算FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        total_fps += fps
        frame_count += 1
        
        # 绘制结果
        if boxes:  # 只有检测到对象时才绘制
            frame = model.draw_detections(frame, boxes, scores, class_ids)
        
        # 显示FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("YOLOv8 ONNX (PiCam2)", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("用户中断操作")
finally:
    model.picam2.stop()
    cv2.destroyAllWindows()
    if frame_count > 0:
        print(f"平均FPS: {total_fps / frame_count:.1f}")
    else:
        print("未处理任何帧")