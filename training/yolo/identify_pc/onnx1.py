from ultralytics import YOLO

model = YOLO("runs/detect/my_custom_model/weights/best.pt")  # 加载训练好的模型
model.export(
    format="onnx",
    dynamic=True,       # 树莓派建议固定输入尺寸（简化部署）False,True是不固定
    imgsz=640,          # 输入尺寸与训练一致
    opset=12,            # 兼容 ONNX Runtime 的算子版本
    simplify=True,        # 简化模型结构（移除冗余节点）
)