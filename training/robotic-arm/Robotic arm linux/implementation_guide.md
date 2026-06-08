# YOLO检测与机械臂跟踪实现指南

本文档详细说明如何使用更新后的`yolo.py`和新创建的`object_tracking.py`文件实现以下功能：

1. 在`_interactive_state`函数中仅识别人脸
2. 在`_work_state`函数中仅识别书本
3. 将YOLO模型返回的坐标转换为机械臂所需的坐标系

## 1. 文件功能说明

### 1.1 `yolo.py`
我已更新`yolo.py`文件，新增了`YOLODetector`类，主要功能包括：

- 加载并使用YOLO模型（同时支持face和book检测）
- 提供两个专用方法：`detect_face()`和`detect_book()`，分别用于仅检测人脸和仅检测书本
- 自动将YOLO返回的像素坐标转换为机械臂可用的3D坐标系
- 集成`serial_comm.py`中的超声波测距，提高坐标计算精度
- 支持在检测结果上绘制可视化信息（可选）

### 1.2 `object_tracking.py`
这是一个新文件，合并了原`face.py`和`book.py`的功能，主要包括：

- 统一的`ObjectTracker`类，处理人脸和书本的机械臂跟踪
- 提供`track_face()`和`track_book()`两个主要方法，分别用于人脸跟踪和书本跟踪
- 自动进行逆运动学计算，输出舵机角度
- 在调试模式下支持3D可视化

## 2. 在main.py中的使用方法

由于需求要求不直接修改`main.py`文件，以下是**如何修改**的具体指导，您可以根据此指导手动修改`main.py`。

### 2.1 导入新模块

首先，在`main.py`文件顶部的导入部分添加：

```python
from yolo import YOLODetector
from object_tracking import ObjectTracker
```

### 2.2 初始化对象

在`SmartLampSystem`的`__init__`函数中，将原有的YOLO初始化替换为：

```python
# 将原有的这一行：
self.yolo = CameraYOLO("best.pt")

# 替换为：
self.yolo_detector = YOLODetector(model_path="best.pt")
self.object_tracker = ObjectTracker()
```

### 2.3 修改`_interactive_state`函数中的YOLO部分

在`_interactive_state`函数中，找到`yolo_tracker`线程函数，修改为：

```python
def yolo_tracker():
    """人脸跟踪线程"""
    while not self.stop_threads:
        try:
            # 捕获摄像头帧
            frame = self.cam2.capture_array()
            
            # 使用更新后的YOLO检测器仅检测人脸
            face_coordinates = self.yolo_detector.detect_face(frame)
            
            # 如果检测到人脸，计算机械臂角度
            if face_coordinates:
                # 使用object_tracker计算机械臂角度
                base_angle, wrist_angle = self.object_tracker.track_face(face_coordinates)
                
                # 设置舵机角度（只需更新底座和腕部）
                # 将角度转换为度数（0-180度范围）
                base_deg = int(np.degrees(base_angle))
                wrist_deg = int(np.degrees(wrist_angle))
                
                # 发送舵机控制命令
                self.serial_comm.send_servo_action('Alldro', base_deg, 90, 90, wrist_deg)
                
                with self.lock:
                    self.yolo_results = {"face": face_coordinates}
            
        except Exception as e:
            print(f"人脸跟踪异常: {e}")
```

### 2.4 修改`_work_state`函数中的YOLO部分

在`_work_state`函数中，找到`yolo_tracker`线程函数，修改为：

```python
def yolo_tracker():
    """书本跟踪线程"""
    while not self.stop_threads:
        try:
            frame = self.cam2.capture_array()
            
            # 使用更新后的YOLO检测器仅检测书本
            book_coordinates = self.yolo_detector.detect_book(frame)
            
            # 如果检测到书本，计算机械臂角度
            if book_coordinates:
                # 使用object_tracker计算四轴机械臂角度
                base_angle, shoulder_angle, elbow_angle, wrist_angle = \
                    self.object_tracker.track_book(book_coordinates)
                
                # 将角度转换为度数（0-180度范围）
                base_deg = int(np.degrees(base_angle))
                shoulder_deg = int(np.degrees(shoulder_angle))
                elbow_deg = int(np.degrees(elbow_angle))
                wrist_deg = int(np.degrees(wrist_angle))
                
                # 发送舵机控制命令（四个舵机都需控制）
                self.serial_comm.send_servo_action('Alldro', base_deg, shoulder_deg, elbow_deg, wrist_deg)
                
                with self.lock:
                    self.yolo_results = {"book": book_coordinates}
            
            time.sleep(0.03)
        except Exception as e:
            print(f"[YOLO异常] {str(e)}")
```

## 3. 注意事项

1. **坐标系转换**：
   - YOLO检测返回的是像素坐标 (x, y, width, height)
   - `yolo.py`中的坐标转换方法将其转换为3D空间坐标 (x, y, z)，单位为厘米
   - `object_tracking.py`中的方法将这些3D坐标转换为舵机角度（弧度）

2. **距离测量**：
   - 代码优先使用`serial_comm.py`中的超声波测距模块获取距离
   - 如果无法获取实际距离，将使用目标大小（书本或人脸）估算距离

3. **类别过滤**：
   - `detect_face()`方法仅返回人脸检测结果
   - `detect_book()`方法仅返回书本检测结果
   - YOLO模型需包含'face'和'book'两个类别

4. **舵机控制**：
   - 人脸跟踪仅控制底座和腕部舵机
   - 书本跟踪控制所有四个舵机
   - 角度需从弧度转换为度数（0-180度范围）

## 4. 调试提示

1. 运行测试模式验证YOLO检测：
   ```
   python yolo.py
   ```

2. 运行视觉化测试验证坐标转换：
   ```
   python object_tracking.py
   ```

3. 添加调试输出检查坐标和角度计算：
   ```python
   # 在代码中添加
   print(f"检测坐标: {coordinates}")
   print(f"舵机角度: {base_deg}, {shoulder_deg}, {elbow_deg}, {wrist_deg}")
   ```

## 5. 可能的故障排除

1. 如果检测不稳定，可以调整置信度阈值：
   ```python
   # 在yolo.py中调整
   detections = self.detect(frame, conf_threshold=0.6, target_class='face')  # 提高置信度阈值
   ```

2. 如果舵机移动过快，可以使用平滑轨迹：
   ```python
   # 计算平滑轨迹
   trajectory = self.object_tracker.generate_smooth_trajectory(
       current_angles, (base_angle, shoulder_angle, elbow_angle, wrist_angle), steps=10
   )
   # 依次发送
   for angles in trajectory:
       # 转换为度数并发送
   ```

3. 如果距离测量不准确，可以手动校准：
   ```python
   # 在初始化后调用
   self.yolo_detector.pixel_to_cm_ratio = 12  # 调整像素到厘米的比例
