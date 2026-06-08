from onnxruntime.quantization import quantize_static, QuantType, CalibrationDataReader
import numpy as np
import cv2
import os

class CustomDataReader(CalibrationDataReader):
    def __init__(self, image_folder, input_size=640):
        self.image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder)]
        self.input_size = input_size  # 输入尺寸必须与训练/导出一致
        self.index = 0

    def get_next(self):
        if self.index >= len(self.image_paths):
            return None
        
        # 读取原始图像
        image = cv2.imread(self.image_paths[self.index])
        h, w = image.shape[:2]

        # 保持宽高比的缩放（与训练/推理一致）
        scale = self.input_size / max(h, w)  # 长边缩放到目标尺寸
        new_h, new_w = int(h * scale), int(w * scale)
        resized = cv2.resize(image, (new_w, new_h))

        # 填充为正方形（与训练预处理一致）
        canvas = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        canvas[:new_h, :new_w] = resized

        # 转换为模型输入格式（CHW + 归一化）
        blob = canvas.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))  # HWC → CHW
        blob = np.expand_dims(blob, axis=0)   # 添加batch维度

        self.index += 1
        return {"images": blob}  # 输入名称必须与ONNX模型匹配

# 量化配置（根据硬件选择QInt8或QUInt8）
quantize_static(
    model_input="runs/detect/my_custom_model/weights/best.onnx",
    model_output="runs/detect/my_custom_model/weights/best_quant.onnx",
    calibration_data_reader=CustomDataReader("calibration_images/", input_size=640),
    quant_format=QuantType.QInt8,       # 树莓派建议使用QInt8
    activation_type=QuantType.QInt8,
    weight_type=QuantType.QInt8,
    optimize_model=True,                # 优化模型结构
    use_external_data_format=False,     # 单文件模式（简化部署）
)