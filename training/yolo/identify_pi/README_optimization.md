# YOLOv8 Raspberry Pi 优化文档

## 原始代码中的问题

分析 `main_optimized_pi.py` 后，我发现了几个可能导致树莓派上识别效果差和检测框集中在左上角的问题：

1. **预处理和后处理坐标变换问题**：
   - 坐标计算逻辑存在缺陷，没有正确处理 letterbox 填充的偏移量
   - 尺度变换（scale）没有正确应用到坐标转换中
   - 没有正确将模型输出坐标映射回原始图像坐标系

2. **ONNX 运行时配置**：
   - 没有针对树莓派进行性能优化配置
   - 缺少线程数和图优化设置

3. **置信度阈值设置**：
   - 过滤逻辑不完善，可能让低质量检测通过

4. **模型验证不完整**：
   - 缺少对输入输出形状的全面验证
   - 没有对模型文件存在性的检查

## 优化的改进

新版本 `main_optimized_pi_new.py` 进行了全面优化：

### 1. 代码结构优化
- 更清晰的类结构和方法组织
- 增强的错误处理和异常捕获
- 更全面的代码注释

### 2. ONNX 运行时优化
```python
options = ort.SessionOptions()
options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
options.intra_op_num_threads = 4  # 根据树莓派型号调整（树莓派4有4个核心）
```

### 3. 预处理优化
- 正确实现 letterbox 填充，保持宽高比
- 计算并储存中心偏移量，用于后处理坐标恢复
- 使用标准的 YOLO 预处理方法（114灰色填充）

### 4. 后处理关键修复
```python
# 从 letterbox 图像中移除偏移并缩放回原始图像尺寸
x1_orig = (x1[i] - offset_x) / scale
y1_orig = (y1[i] - offset_y) / scale
x2_orig = (x2[i] - offset_x) / scale
y2_orig = (y2[i] - offset_y) / scale
```

### 5. 可视化增强
- 为不同类别设置不同颜色
- 添加标签背景，提高可读性
- 优化标签位置计算

### 6. 性能优化
- 帧率平滑处理，显示更稳定
- 降低默认帧率（15 FPS），提高树莓派性能稳定性
- 更合理的异常和资源处理

## 使用说明

1. 确保模型文件路径正确：
   ```python
   MODEL_PATH = "runs/detect/my_custom_model/weights/best_quant.onnx"
   ```

2. 确保类别与训练一致：
   ```python
   CLASSES = ["person_a", "person_b", "person_c", "book"]
   ```

3. 如果检测效果不理想，可以调整置信度阈值：
   ```python
   detector = YOLOv8ONNXDetector(
       model_path=MODEL_PATH,
       classes=CLASSES,
       conf_threshold=0.30,  # 降低此阈值可以看到更多检测结果
       iou_threshold=0.45
   )
   ```

4. 运行程序：
   ```bash
   python main_optimized_pi_new.py
   ```

5. 按 'q' 键退出程序

## 调试建议

如果在树莓派上仍有问题：

1. 检查模型路径是否正确（程序会验证并提示）
2. 查看打印的模型信息，确认输入输出形状正确
3. 降低置信度阈值（如0.2）尝试看到更多检测框
4. 尝试修改相机配置以适应不同光照条件
