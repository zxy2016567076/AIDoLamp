import cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# 初始化YOLOv8模型（使用预训练的COCO模型，包含车辆和行人）
model = YOLO('yolov8n.pt')  # 也可以使用yolov8s.pt, yolov8m.pt等更大模型提高精度

# 初始化DeepSORT追踪器
tracker = DeepSort(max_age=30,  # 目标丢失后保持的帧数
                   n_init=3,    # 需要多少次检测才能确认新目标
                   nms_max_overlap=1.0, 
                   max_cosine_distance=0.4,
                   nn_budget=None)

# 定义要追踪的类别（COCO数据集中的车辆和行人）
CLASS_NAMES = model.names
VEHICLE_CLASSES = [2, 3, 5, 7]   # COCO类别: 2-汽车, 3-摩托车, 5-公交车, 7-卡车
PERSON_CLASSES = [0]              # 0-人

# 视频处理函数
def process_video(input_video, output_video=None):
    # 打开视频
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print(f"无法打开视频: {input_video}")
        return
    
    # 获取视频属性
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 初始化视频输出
    if output_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    # 存储追踪历史
    track_history = defaultdict(lambda: [])
    frame_count = 0
    
    print(f"开始处理视频: {input_video}")
    print(f"视频尺寸: {width}x{height}, FPS: {fps:.1f}, 总帧数: {total_frames}")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        if frame_count % 10 == 0:
            print(f"处理进度: {frame_count}/{total_frames} 帧 ({frame_count/total_frames*100:.1f}%)")
        
        # 使用YOLOv8进行目标检测
        results = model.predict(
            frame, 
            classes=VEHICLE_CLASSES + PERSON_CLASSES,  # 只检测车辆和行人
            conf=0.7,  # 置信度阈值
            #device='cpu',  # 使用GPU可改为 '0' 或 'cuda'
            device='0',
            verbose=False
        )
        
        # 准备DeepSORT的检测结果
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # 获取边界框坐标
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1
                
                # 获取置信度和类别
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                class_name = CLASS_NAMES[cls_id]
                
                # 添加到检测列表
                detections.append(([x1, y1, w, h], conf, class_name))
        
        # 使用DeepSORT更新追踪器
        tracks = tracker.update_tracks(detections, frame=frame)
        
        # 在帧上绘制结果
        for track in tracks:
            if not track.is_confirmed():
                continue
                
            track_id = track.track_id
            ltrb = track.to_ltrb()  # 获取边界框 [left, top, right, bottom]
            
            # 提取坐标
            x1, y1, x2, y2 = map(int, ltrb)
            w, h = x2 - x1, y2 - y1
            
            # 获取类别信息
            class_id = track.get_det_class()
            class_name = CLASS_NAMES.get(class_id, 'object')
            
            # 为不同类别设置不同颜色
            if class_id in VEHICLE_CLASSES:
                color = (0, 255, 255)  # 车辆: 黄色
            else:
                color = (0, 255, 0)    # 行人: 绿色
            
            # 绘制边界框和ID
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{class_name} ID:{track_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # 存储追踪历史
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            track_history[track_id].append(center)
            
            # 绘制追踪轨迹（最近30帧）
            if len(track_history[track_id]) > 1:
                points = np.array(track_history[track_id][-30:], np.int32)
                cv2.polylines(frame, [points], False, color, 2)
        
        # 显示统计信息
        stats_text = f"帧: {frame_count}/{total_frames} | 追踪目标: {len(tracks)}"
        cv2.putText(frame, stats_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # 显示/保存结果
        if output_video:
            out.write(frame)
        
        cv2.imshow('YOLOv8 + DeepSORT 多目标追踪', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 清理资源
    cap.release()
    if output_video:
        out.release()
    cv2.destroyAllWindows()
    print("处理完成!")

if __name__ == "__main__":
    # ======== 配置区 ========
    INPUT_VIDEO = "soccer_01.mp4"  # 替换为你的视频路径
    OUTPUT_VIDEO = "output_video.mp4"  # 输出视频路径，设为None则不保存
    
    # 处理视频
    process_video(INPUT_VIDEO, OUTPUT_VIDEO)