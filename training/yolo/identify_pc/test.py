import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession("runs/detect/my_custom_model/weights/best_quant.onnx")
output_shape = sess.get_outputs()[0].shape
print(f"模型输出形状: {output_shape}")
