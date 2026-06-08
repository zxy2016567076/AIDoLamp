# main_csi.py
from picamera2 import Picamera2, Preview
import cv2
import time
import numpy as np
import onnxruntime as ort
import libcamera

# 加载模型并检查输出
model_path = "best.onnx"
session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
print("Input shape:", session.get_inputs()[0].shape)
print("Output shape:", session.get_outputs()[0].shape)

# CSI摄像头初始化配置
picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"size": (640, 480), "format": "RGB888"},
    transform=libcamera.Transform(hflip=1)
)
picam2.configure(config)
picam2.start()

class_names = {
    0: "person_a",
    1: "person_b",
    2: "person_c",
    3: "book"
}

def preprocess(frame, target_size=320):
    """Letterbox预处理，保持宽高比"""
    h, w = frame.shape[:2]
    scale = min(target_size / h, target_size / w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(frame, (new_w, new_h))
    padded = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
    padded[:new_h, :new_w] = resized
    return padded, scale, (new_w, new_h)

prev_time = 0
frame_count = 0
fps = 0

try:
    while True:
        frame = picam2.capture_array()
        
        # Letterbox预处理
        padded_img, scale, (new_w, new_h) = preprocess(frame)
        input_img = padded_img.transpose(2, 0, 1)  # HWC -> CHW
        input_img = np.expand_dims(input_img, 0).astype(np.float32) / 255.0
        
        # 推理
        outputs = session.run([output_name], {input_name: input_img})
        predictions = outputs[0][0]  # shape: (8400, 84)
        
        # 解析输出（假设输出为[x_center, y_center, w, h, obj_score, class_scores...]）
        boxes = predictions[:, :4]
        scores = predictions[:, 4]
        class_ids = np.argmax(predictions[:, 5:], axis=1)
        
        # 应用置信度阈值过滤
        mask = scores > 0.5
        boxes = boxes[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]
        
        # 转换为原始图像坐标
        for box, score, cls_id in zip(boxes, scores, class_ids):
            x_center, y_center, w, h = box
            # 反归一化到输入尺寸（320x320）
            x_center = x_center * 320
            y_center = y_center * 320
            w = w * 320
            h = h * 320
            # 转换到Letterbox前的坐标（考虑缩放比例）
            x1 = int((x_center - w/2) / scale)
            y1 = int((y_center - h/2) / scale)
            x2 = int((x_center + w/2) / scale)
            y2 = int((y_center + h/2) / scale)
            
            # 确保坐标不超出图像边界
            x1 = max(0, min(x1, 640))
            y1 = max(0, min(y1, 480))
            x2 = max(0, min(x2, 640))
            y2 = max(0, min(y2, 480))
            
            cls_name = class_names.get(int(cls_id), "unknown")
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{cls_name}: {score:.2f}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # FPS计算
        current_time = time.time()
        frame_count += 1
        if current_time - prev_time >= 1:
            fps = frame_count / (current_time - prev_time)
            prev_time = current_time
            frame_count = 0
        
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Detection", frame)

        if cv2.waitKey(1) == ord('q'):
            break

finally:
    picam2.stop()
    cv2.destroyAllWindows()