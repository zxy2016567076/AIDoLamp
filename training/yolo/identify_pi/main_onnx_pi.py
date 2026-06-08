from picamera2 import Picamera2
import cv2
import time
import numpy as np
import onnxruntime as ort
import libcamera

# 加载模型并检查输出
model_path = "best.onnx"  # 模型文件路径
session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])  # 创建 ONNX Runtime 推理会话
input_name = session.get_inputs()[0].name  # 获取模型输入的名称
output_name = session.get_outputs()[0].name  # 获取模型输出的名称
print("Input shape:", session.get_inputs()[0].shape)  # 打印模型输入的形状
print("Output shape:", session.get_outputs()[0].shape)  # 打印模型输出的形状

# CSI摄像头初始化配置
picam2 = Picamera2()  # 初始化 Picamera2 摄像头
config = picam2.create_preview_configuration(
    main={"size": (640, 480), "format": "RGB888"},  # 设置主流的分辨率和格式
    transform=libcamera.Transform(hflip=1)  # 设置水平翻转
)
picam2.configure(config)  # 应用摄像头配置
picam2.start()  # 启动摄像头

# 定义类别名称映射
class_names = {
    0: "person_a",  # 类别 0 的名称
    1: "person_b",   # 类别 1 的名称
    2: "person_c",  # 类别 2 的名称
    3: "book"  # 类别 3 的名称
}

def preprocess(frame, target_size=640):
    """Letterbox预处理，保持宽高比"""
    h, w = frame.shape[:2]  # 获取图像的高度和宽度
    scale = min(target_size / h, target_size / w)  # 计算缩放比例，保持宽高比
    new_h, new_w = int(h * scale), int(w * scale)  # 计算缩放后的高度和宽度
    resized = cv2.resize(frame, (new_w, new_h))  # 调整图像大小
    padded = np.full((target_size, target_size, 3), 114, dtype=np.uint8)  # 创建填充后的图像，填充颜色为灰色 (114, 114, 114)
    padded[:new_h, :new_w] = resized  # 将调整大小的图像放入填充图像中
    return padded, scale, (new_w, new_h)  # 返回填充后的图像、缩放比例和新尺寸

prev_time = 0  # 上一次计算 FPS 的时间
frame_count = 0  # 帧计数器
fps = 0  # 当前 FPS

try:
    while True:
        frame = picam2.capture_array()  # 从摄像头捕获一帧图像

        # Letterbox预处理
        padded_img, scale, (new_w, new_h) = preprocess(frame)  # 对图像进行预处理
        input_img = padded_img.transpose(2, 0, 1)  # 将图像从 HWC 格式转换为 CHW 格式
        input_img = np.expand_dims(input_img, 0).astype(np.float32) / 255.0  # 添加批次维度并归一化到 [0, 1]

        # 推理
        outputs = session.run([output_name], {input_name: input_img})  # 使用 ONNX 模型进行推理，获取输出
        predictions = outputs[0][0]  # 获取推理结果，形状为 (8400, 85)

        # 解析输出（假设输出为[x_center, y_center, w, h, obj_score, class_scores...]）
        boxes = predictions[:, :4]  # 提取边界框信息
        scores = predictions[:, 4]  # 提取目标置信度
        class_scores = predictions[:, 5:]  # 提取类别置信度

        # 应用置信度阈值过滤
        mask = scores > 0.5  # 过滤置信度低于 0.5 的检测结果
        boxes = boxes[mask]  # 保留置信度高的边界框
        scores = scores[mask]  # 保留置信度高的分数
        class_scores = class_scores[mask]  # 保留置信度高的类别分数

        class_ids = np.argmax(class_scores, axis=1)  # 获取每个检测结果的类别 ID

        # 转换为原始图像坐标
        for box, score, cls_id in zip(boxes, scores, class_ids):
            x_center, y_center, w, h = box  # 提取边界框的中心点坐标和宽高
            # 反归一化到输入尺寸（640x640）
            x_center = x_center * 640  # 将中心点 x 坐标反归一化
            y_center = y_center * 640  # 将中心点 y 坐标反归一化
            w = w * 640  # 将宽度反归一化
            h = h * 640  # 将高度反归一化
            # 转换到Letterbox前的坐标（考虑缩放比例）
            x1 = int((x_center - w / 2) / scale)  # 计算左上角 x 坐标
            y1 = int((y_center - h / 2) / scale)  # 计算左上角 y 坐标
            x2 = int((x_center + w / 2) / scale)  # 计算右下角 x 坐标
            y2 = int((y_center + h / 2) / scale)  # 计算右下角 y 坐标

            # 确保坐标不超出图像边界
            x1 = max(0, min(x1, 640))  # 限制 x1 在图像范围内
            y1 = max(0, min(y1, 480))  # 限制 y1 在图像范围内
            x2 = max(0, min(x2, 640))  # 限制 x2 在图像范围内
            y2 = max(0, min(y2, 480))  # 限制 y2 在图像范围内

            cls_name = class_names.get(int(cls_id), "unknown")  # 获取类别名称，默认为 "unknown"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 在图像上绘制边界框
            cv2.putText(frame, f"{cls_name}: {score:.2f}", (x1, y1-10),  # 在边界框上方绘制类别名称和置信度
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # FPS计算
        current_time = time.time()  # 获取当前时间
        frame_count += 1  # 增加帧计数
        if current_time - prev_time >= 1:  # 如果超过 1 秒
            fps = frame_count / (current_time - prev_time)  # 计算 FPS
            prev_time = current_time  # 更新上一次时间
            frame_count = 0  # 重置帧计数

        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),  # 在图像上显示 FPS
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Detection", frame)  # 显示检测结果

        if cv2.waitKey(1) == ord('q'):  # 按下 'q' 键退出循环
            break

finally:
    picam2.stop()  # 停止摄像头
    cv2.destroyAllWindows()  # 关闭所有 OpenCV 窗口
