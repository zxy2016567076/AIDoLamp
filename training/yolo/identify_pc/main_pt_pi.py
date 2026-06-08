from picamera2 import Picamera2
import cv2
import time
from ultralytics import YOLO

# -------------------------- 初始化摄像头 --------------------------
picam2 = Picamera2()

# 配置摄像头（移除 AfMode 和 AwbMode）
config = picam2.create_preview_configuration(
    main={
        "size": (640, 480),  # 分辨率
        "format": "RGB888"   # 颜色格式
    },
    controls={
        "FrameRate": 30      # 仅设置帧率
    }
)
picam2.configure(config)

# -------------------------- 加载模型 --------------------------
try:
    model = YOLO("best.pt")  # 替换为你的模型路径
except Exception as e:
    print(f"[错误] 模型加载失败: {e}")
    exit()

# -------------------------- 主循环 --------------------------
try:
    picam2.start()
    print("摄像头已启动，按 Q 键退出...")

    prev_time = 0
    while True:
        # 捕获帧
        frame = picam2.capture_array()

        # 计算 FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if prev_time > 0 else 0
        prev_time = current_time

        # 目标检测
        results = model(frame)
        annotated_frame = results[0].plot()

        # 显示信息
        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("CSI Camera - YOLOv8", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("用户中断操作")
finally:
    picam2.stop()
    cv2.destroyAllWindows()