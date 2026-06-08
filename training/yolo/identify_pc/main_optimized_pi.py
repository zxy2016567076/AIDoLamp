from picamera2 import Picamera2
import cv2
import time
import numpy as np
import onnxruntime as ort

class YOLOv8ONNXPicam:
    def __init__(self, model_path, classes):
        # 初始化ONNX推理会话
        self.session = ort.InferenceSession(model_path,
                                          providers=['CPUExecutionProvider'])  # 树莓派使用CPU
        self.classes = classes
        
        # 获取模型输入参数
        self.input_name = self.session.get_inputs()[0].name
        input_shape = self.session.get_inputs()[0].shape
        #self.input_h, self.input_w = input_shape[2], input_shape[3]
        self.input_w = 640  # ✅ 强制写为整数
        self.input_h = 480  # ✅ 强制写为整数
        
        # 摄像头参数
        self.picam2 = Picamera2()
        self.configure_camera()

    def configure_camera(self):
        """ 配置Picamera2参数 """
        config = self.picam2.create_preview_configuration(
            main={"size": (640, 480), "format": "RGB888"},
            controls={"FrameRate": 30}
        )
        self.picam2.configure(config)
        self.picam2.start()

    def preprocess(self, image):
        """ 优化后的预处理流程 """
        # 保持宽高比的缩放
        h, w = image.shape[:2]
        scale = min(self.input_h / h, self.input_w / w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        resized = cv2.resize(image, (new_w, new_h))
        canvas = np.full((self.input_h, self.input_w, 3), 114, dtype=np.uint8)
        canvas[:new_h, :new_w] = resized
        
        # 转换为模型输入格式
        blob = cv2.dnn.blobFromImage(
            canvas, 1/255.0, swapRB=True, crop=False)
        return blob, (w, h), (new_w / w, new_h / h)

    def postprocess(self, outputs, orig_size, ratios):
        """ 优化的后处理流程 """
        outputs = np.squeeze(outputs[0])
        boxes = []
        scores = []
        class_ids = []
        
        orig_w, orig_h = orig_size
        ratio_w, ratio_h = ratios
        
        for detection in outputs:
            if detection[4] < 0.5:  # 置信度阈值
                continue
            
            scores_all = detection[5:]
            class_id = np.argmax(scores_all)
            confidence = detection[4] * scores_all[class_id]
            
            if confidence < 0.5:
                continue
            
            # 还原坐标到原始图像尺寸
            cx = detection[0] * self.input_w / ratio_w
            cy = detection[1] * self.input_h / ratio_h
            w = detection[2] * self.input_w / ratio_w
            h = detection[3] * self.input_h / ratio_h
            
            x1 = int((cx - w/2))
            y1 = int((cy - h/2))
            x2 = int((cx + w/2))
            y2 = int((cy + h/2))
            
            # 边界检查
            x1 = max(0, min(x1, orig_w))
            y1 = max(0, min(y1, orig_h))
            x2 = max(0, min(x2, orig_w))
            y2 = max(0, min(y2, orig_h))
            
            boxes.append([x1, y1, x2, y2])
            scores.append(float(confidence))
            class_ids.append(class_id)
        
        # 加速的NMS实现
        indices = cv2.dnn.NMSBoxes(boxes, scores, 0.5, 0.5)
        return (
            [boxes[i] for i in indices],
            [scores[i] for i in indices],
            [class_ids[i] for i in indices]
        )

    def draw_detections(self, image, boxes, scores, class_ids):
        """ 优化的绘制方法 """
        for box, score, class_id in zip(boxes, scores, class_ids):
            color = (0, 255, 0)
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, 2)
            label = f"{self.classes[class_id]}: {score:.2f}"
            cv2.putText(image, label, (box[0], box[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return image

# -------------------------- 初始化 --------------------------
# 替换为你的实际类别列表
CLASSES = ["person_a", "person_b","person_c","book"]  
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
        
        # 预处理
        blob, orig_size, ratios = model.preprocess(frame)
        
        # 推理
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
        annotated_frame = model.draw_detections(frame, boxes, scores, class_ids)
        
        # 显示FPS
        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("YOLOv8 ONNX (Picam2)", annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("用户中断操作")
finally:
    model.picam2.stop()
    cv2.destroyAllWindows()
    print(f"平均FPS: {total_fps / frame_count:.1f}")