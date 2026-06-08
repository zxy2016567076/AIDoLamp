import cv2
import numpy as np
import time
import onnxruntime as ort

#ONNX模型特性
# 优点：已量化（体积小3-5x），支持跨平台加速
# 注意：需手动处理前后处理，已优化为最大兼容性

class YOLOv8ONNX:
    def __init__(self, model_path):
        # 配置ONNX Runtime
        self.session = ort.InferenceSession(model_path, 
                                          providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        # 获取输入输出参数 
        input_shape = self.session.get_inputs()[0].shape
        self.input_h, self.input_w = input_shape[2], input_shape[3]
        self.classes = ["class1", "class2"]  # 替换为实际类别列表
        
    def preprocess(self, image):
        """ ONNX输入预处理 """
        # 与原版一致的预处理步骤 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR→RGB转换
        image = cv2.resize(image, (self.input_w, self.input_h))  # 固定尺寸缩放
        image = image.transpose(2, 0, 1)  # HWC→CHW
        image = np.expand_dims(image, axis=0).astype(np.float32) / 255.0  # 归一化+添加批次维度
        return image
    
    def postprocess(self, outputs, orig_image):
        """ 手动实现后处理+NMS过滤 """
        # 输出格式：[/output0, /output1] 或 single output 
        outputs = np.squeeze(outputs[0])  # 假设outputs[0]是检测结果
        boxes = []
        scores = []
        class_ids = []
        
        #  解析三个检测头的输出（如果有） 
        for detection in outputs:
            # 假设每个检测行格式: [x_center, y_center, width, height, score, class_scores...]
            scores_all = detection[4:]
            class_id = np.argmax(scores_all)
            confidence = scores_all[class_id]
            
            if confidence > 0.5:
                cx, cy, w, h = detection[0], detection[1], detection[2], detection[3] 
                # ===重要===还原到原图坐标
                orig_h, orig_w = orig_image.shape[:2]
                x1 = int((cx - w/2) * orig_w / self.input_w)
                y1 = int((cy - h/2) * orig_h / self.input_h)
                x2 = int((cx + w/2) * orig_w / self.input_w)
                y2 = int((cy + h/2) * orig_h / self.input_h)
                
                boxes.append([x1, y1, x2, y2])
                scores.append(float(confidence))
                class_ids.append(class_id)
        
        # OpenCV实现NMS 
        indices = cv2.dnn.NMSBoxes(boxes, scores, 0.5, 0.5)
        final_boxes = [boxes[i] for i in indices]
        final_scores = [scores[i] for i in indices]
        final_classes = [class_ids[i] for i in indices]
        
        return final_boxes, final_scores, final_classes

    def draw_detections(self, image, boxes, scores, class_ids):
        """ 手动绘制检测结果 """
        annotated_image = image.copy()
        for box, score, class_id in zip(boxes, scores, class_ids):
            # 框线和文本绘制
            color = (0, 255, 0)  # 统一使用绿色标注
            cv2.rectangle(annotated_image, (box[0], box[1]), (box[2], box[3]), color, 2)
            label = f"{self.classes[class_id]}: {score:.2f}"
            cv2.putText(annotated_image, label, (box[0], box[1]-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return annotated_image

# 初始化模型
model = YOLOv8ONNX("runs/detect/my_custom_model/weights/best_quant.onnx")

cap = cv2.VideoCapture(0)
video_writer = None # 视频写入器
prev_time = 0 # 用于计算FPS

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    # 预处理 → 推理 → 后处理
    input_tensor = model.preprocess(frame)
    outputs = model.session.run(model.output_names, {model.input_name: input_tensor})
    boxes, scores, class_ids = model.postprocess(outputs, frame)
    
    # 计算FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if (current_time - prev_time) else 0
    prev_time = current_time
    
    # 绘制检测结果
    annotated_frame = model.draw_detections(frame, boxes, scores, class_ids)
    
    # 信息叠加
    info_text = f"FPS: {fps:.2f} | Objects: {len(boxes)}"
    cv2.putText(annotated_frame, info_text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    
    # 视频保存
    if video_writer is None:
        h, w = annotated_frame.shape[:2]
        video_writer = cv2.VideoWriter('output_onnx.mp4', 
                                      cv2.VideoWriter_fourcc(*'mp4v'), 
                                      30, (w, h))
    video_writer.write(annotated_frame)
    
    cv2.imshow('YOLOv8 Detection (ONNX Quantized)', annotated_frame)
    if cv2.waitKey(1) == ord('q'): break

cap.release()
video_writer.release()
cv2.destroyAllWindows()
