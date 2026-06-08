import cv2
import numpy as np
import time
import threading
from yolo import BookDetector
import robetpose as rp

class ArmTrackingSystem:
    def __init__(self, camera_index=0, model_path="yolov8n.pt", use_distance_sensor=False, 
                 tracking_interval=0.2, visualization=True):
        """
        初始化台灯跟踪系统
        
        参数:
        camera_index -- 摄像头索引
        model_path -- YOLOv8模型路径
        use_distance_sensor -- 是否使用测距传感器
        tracking_interval -- 跟踪更新间隔（秒）
        visualization -- 是否显示可视化界面
        """
        # 初始化书本检测器
        self.detector = BookDetector(
            camera_index=camera_index,
            model_path=model_path,
            distance_module=use_distance_sensor
        )
        
        # 系统参数
        self.tracking_interval = tracking_interval
        self.visualization = visualization
        
        # 机械臂当前位置与状态
        self.current_arm_position = (40, 0, 0)  # 默认位置，可以根据实际情况调整
        self.detector.set_current_arm_position(self.current_arm_position)
        
        # 跟踪状态
        self.is_tracking = False
        self.tracking_thread = None
        
        # 跟踪历史记录
        self.tracking_history = []
        self.max_history_length = 100
        
        # 控制参数
        self.min_movement_threshold = 5.0  # 最小移动阈值（厘米），避免微小抖动
        self.smoothing_factor = 0.5  # 平滑因子，值越大响应越平滑
        self.target_distance = 40.0  # 目标与灯之间的固定距离（厘米）
    
    def start_tracking(self):
        """开始目标跟踪"""
        if not self.is_tracking:
            self.is_tracking = True
            self.tracking_thread = threading.Thread(target=self._tracking_loop)
            self.tracking_thread.daemon = True
            self.tracking_thread.start()
            print("目标跟踪已启动")
    
    def stop_tracking(self):
        """停止目标跟踪"""
        self.is_tracking = False
        if self.tracking_thread:
            self.tracking_thread.join(timeout=1.0)
            self.tracking_thread = None
        print("目标跟踪已停止")
    
    def _tracking_loop(self):
        """跟踪主循环"""
        last_move_time = 0
        
        while self.is_tracking:
            current_time = time.time()
            
            # 控制更新频率
            if current_time - last_move_time < self.tracking_interval:
                time.sleep(0.01)  # 短暂休眠以减少CPU使用
                continue
            
            # 检测目标
            target_coords, detection_info, frame = self.detector.detect_and_track()
            
            # 如果检测到目标，更新机械臂位置
            if target_coords:
                x_target, y_target, z_target = target_coords
                
                # 添加到跟踪历史
                self.tracking_history.append((x_target, y_target, z_target))
                if len(self.tracking_history) > self.max_history_length:
                    self.tracking_history.pop(0)
                
                # 计算平滑后的目标位置（防止抖动）
                smoothed_target = self._calculate_smoothed_target()
                
                # 计算当前位置与目标位置的距离
                current_x, current_y, current_z = self.current_arm_position
                distance_to_target = np.sqrt(
                    (smoothed_target[0] - current_x)**2 + 
                    (smoothed_target[1] - current_y)**2
                )
                
                # 只有当移动距离超过阈值时才移动机械臂
                if distance_to_target > self.min_movement_threshold:
                    # 调用机械臂逆运动学计算关节角度
                    try:
                        base_angle, shoulder_angle, elbow_angle, wrist_angle, ee_pos = rp.calculate_ik(
                            smoothed_target, distance=self.target_distance
                        )
                        
                        # 在这里添加实际控制机械臂的代码
                        # 如：send_angles_to_servos(base_angle, shoulder_angle, elbow_angle, wrist_angle)
                        
                        # 更新当前位置
                        self.current_arm_position = smoothed_target
                        self.detector.set_current_arm_position(smoothed_target)
                        
                        print(f"移动机械臂到：{smoothed_target}, 距离：{distance_to_target:.1f}cm")
                        print(f"关节角度：底座={np.degrees(base_angle):.1f}°, "
                              f"肩部={np.degrees(shoulder_angle):.1f}°, "
                              f"肘部={np.degrees(elbow_angle):.1f}°, "
                              f"腕部={np.degrees(wrist_angle):.1f}°")
                        
                        # 如果需要调试，可以可视化机械臂姿态
                        if self.visualization:
                            joints = rp.forward_kinematics(base_angle, shoulder_angle, elbow_angle, wrist_angle)
                            rp.plot_robotic_arm(joints, target_pos=smoothed_target, 
                                               end_effector_pos=ee_pos, 
                                               title="台灯跟踪状态")
                    
                    except Exception as e:
                        print(f"机械臂控制错误: {e}")
                
                last_move_time = current_time
            
            # 显示视频帧
            if self.visualization and frame is not None:
                cv2.imshow("实时跟踪", frame)
                
                # 按ESC键退出
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    self.stop_tracking()
    
    def _calculate_smoothed_target(self):
        """计算平滑后的目标位置，减少抖动"""
        if len(self.tracking_history) == 0:
            return self.current_arm_position
        
        # 使用最近几个点的加权平均
        window_size = min(10, len(self.tracking_history))
        recent_targets = self.tracking_history[-window_size:]
        
        # 应用指数加权，最近的点权重最大
        weights = np.exp(np.linspace(0, 2, window_size)) - 1
        weights = weights / np.sum(weights)
        
        # 计算加权平均
        x_avg = sum(weights[i] * recent_targets[i][0] for i in range(window_size))
        y_avg = sum(weights[i] * recent_targets[i][1] for i in range(window_size))
        z_avg = 0  # 假设目标始终在桌面上，z=0
        
        # 使用平滑因子在当前位置和新位置之间插值
        current_x, current_y, current_z = self.current_arm_position
        smoothed_x = current_x * self.smoothing_factor + x_avg * (1 - self.smoothing_factor)
        smoothed_y = current_y * self.smoothing_factor + y_avg * (1 - self.smoothing_factor)
        
        return (smoothed_x, smoothed_y, z_avg)
    
    def calibrate(self):
        """校准系统 - 可以用已知尺寸的物体来校准距离估计"""
        print("请将标准物体（如A4纸）放在摄像头前方约40厘米处...")
        
        while True:
            target_coords, detection_info, frame = self.detector.detect_and_track()
            
            if frame is not None:
                cv2.putText(frame, "按C键校准，ESC退出", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow("校准", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord('c') and detection_info:
                # 使用A4纸的尺寸（宽21厘米）进行校准
                known_width = 21.0
                x1, y1, x2, y2 = detection_info['box']
                detected_width = x2 - x1
                
                self.detector.calibrate_pixel_ratio(40.0, known_width, detected_width)
                print("校准完成！")
                time.sleep(1)
                break
        
        cv2.destroyWindow("校准")
    
    def release(self):
        """释放资源"""
        self.stop_tracking()
        self.detector.release()
        cv2.destroyAllWindows()

# 主程序
if __name__ == "__main__":
    # 初始化系统
    tracker = ArmTrackingSystem(
        camera_index=0,
        model_path="yolov8n.pt",
        use_distance_sensor=False,  # 如果有测距传感器则设为True
        visualization=True
    )
    
    try:
        # 菜单选项
        print("\n===== 台灯跟踪系统 =====")
        print("1. 开始跟踪")
        print("2. 校准系统")
        print("3. 测试模式")
        print("4. 退出")
        
        choice = input("请选择操作: ")
        
        if choice == '1':
            # 开始跟踪
            tracker.start_tracking()
            input("按Enter键停止跟踪...")
            
        elif choice == '2':
            # 校准系统
            tracker.calibrate()
            print("校准完成，重新运行程序开始跟踪")
            
        elif choice == '3':
            # 测试模式 - 仅检测不控制机械臂
            print("进入测试模式，按ESC退出")
            while True:
                target_coords, detection_info, frame = tracker.detector.detect_and_track()
                
                if frame is not None:
                    cv2.imshow("测试模式", frame)
                
                if target_coords:
                    print(f"目标坐标: {target_coords}")
                
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    break
        
        else:
            print("退出程序")
    
    finally:
        # 释放资源
        tracker.release()
