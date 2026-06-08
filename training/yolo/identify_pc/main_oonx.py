from ultralytics import YOLO
import cv2
import time

model = YOLO("runs/detect/my_custom_model/weights/best.onnx")

# 对于 ONNX 模型，不能直接设置 args 和 overrides 属性
# model.args['imgsz'] = 320  # ★ 关键设置1：覆盖默认参数
# model.overrides['imgsz'] = 320  # ★ 关键设置2：避免自动检测

cap = cv2.VideoCapture(0)  # 摄像头ID（0为默认）

videoWriter = None
prev_time = 0  # 用于计算实时FPS

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 显式缩放输入图像
    resized_frame = cv2.resize(frame, (320, 320))  # ★ 关键操作：保证输入尺寸匹配

    # 计算实时FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if (current_time - prev_time) != 0 else 0
    prev_time = current_time

    # 调整推理参数
    results = model(
        resized_frame,
        imgsz=320,  # 显式指定推理尺寸
        device='cpu',  # 强制使用CPU
        verbose=False  # 关闭冗余输出
    )
    annotated_frame = results[0].plot()

    # 获取当前帧尺寸
    frame_height, frame_width = annotated_frame.shape[:2]

    # 在帧上显示参数（FPS、分辨率、物体数量）
    cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(annotated_frame, f"Width: {frame_width}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(annotated_frame, f"Height: {frame_height}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    # 显示检测到的物体数量
    num_objects = len(results[0].boxes)
    cv2.putText(annotated_frame, f"Objects: {num_objects}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # 初始化视频写入器（使用第一帧的尺寸和固定FPS）
    if videoWriter is None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        saved_fps = cap.get(cv2.CAP_PROP_FPS)
        if saved_fps <= 0:  # 摄像头未提供有效FPS时使用默认值
            saved_fps = 30
        videoWriter = cv2.VideoWriter("output_video.mp4", fourcc, saved_fps, (frame_width, frame_height))
    
    # 写入标注后的帧
    videoWriter.write(annotated_frame)

    # 显示实时画面
    cv2.imshow("Detection", annotated_frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
videoWriter.release()
cv2.destroyAllWindows()