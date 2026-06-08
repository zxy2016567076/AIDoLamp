import cv2
import numpy as np
import time
import threading
from serial_comm import SerialCommunicator
from yolo import ObjectDetector
import robetpose as rp

class SmartLampController:
    def __init__(self, camera_index=0, model_path="yolov8n.pt", 
                 serial_port="/dev/ttyAMA0", baudrate=9600):
        """
        智能台灯控制器 - 集成视觉检测、机械臂控制和串口通信
        
        参数:
        camera_index -- 摄像头索引
        model_path -- YOLO模型路径
        serial_port -- 串口设备
        baudrate -- 波特率
        """
        # 系统模式 - 'standby', 'face', 'book'
        self.mode = 'standby'
        
        # 初始化目标检测器
        self.detector = ObjectDetector(
            camera_index=camera_index,
            model_path=model_path,
            distance_module=True  # 启用超声波测距模块
        )
        
        # 初始化串口通信
        self.serial = SerialCommunicator(
            port=serial_port,
            baudrate=baudrate
        )
        
        # 当前机械臂角度(弧度) - 初始姿态为竖直朝上
        self.current_angles = {
            'base': 0.0,        # 底座角度（相对x轴）
            'shoulder': 0.0,    # 肩部角度（相对z轴）
            'elbow': 0.0,       # 肘部角度（相对z轴）
            'wrist': 0.0        # 腕部角度（相对z轴）
        }
        
        # 目标姿态和角度
        self.target_angles = self.current_angles.copy()
        
        # 机械臂舵机控制参数
        self.servo_smoothing = True        # 是否启用平滑移动
        self.servo_steps = 15              # 平滑移动的步数
        self.servo_update_interval = 0.05  # 舵机更新间隔（秒）
        
        # 跟踪参数
        self.min_movement_threshold = 5.0  # 最小移动阈值（厘米）
        self.tracking_interval = 0.2       # 跟踪更新间隔（秒）
        self.last_tracking_time = 0        # 上次跟踪时间
        
        # 工作状态
        self.running = False
        self.servo_thread = None
        
        # 目标位置平滑
        self.position_history = []
        self.history_length = 10
        self.position_smoothing = 0.7  # 位置平滑因子
        
        # 超声波距离数据
        self.ultrasonic_distance = None
        
        # 启动线程
        self.main_thread = None
        
        # 初始化灯和机械臂的位置
        self._reset_position()
    
    def _reset_position(self):
        """初始化机械臂到默认姿态"""
        # 设置初始角度（度数）- 默认姿势是台灯向前倾斜
        initial_angles_deg = {
            'base': 0.0,      # 朝向x轴正方向
            'shoulder': 30.0,  # 肩部向前倾斜30度
            'elbow': 45.0,     # 肘部弯曲45度
            'wrist': 0.0       # 腕部竖直
        }
        
        # 转换为弧度
        for joint, angle in initial_angles_deg.items():
            self.current_angles[joint] = np.radians(angle)
            self.target_angles[joint] = np.radians(angle)
        
        # 通过串口发送初始角度命令
        angles_deg = self._angles_to_degrees(self.current_angles)
        self._send_servo_command(angles_deg)
        print(f"机械臂重置到初始位置: {angles_deg}")
    
    def _angles_to_degrees(self, angles_dict):
        """将弧度角度转换为度数"""
        return {
            joint: np.degrees(angle) for joint, angle in angles_dict.items()
        }
    
    def _update_ultrasonic_distance(self, distance_str):
        """处理从串口读取的超声波距离数据"""
        try:
            # 解析距离数据字符串，如 "DIST_42.5"
            if distance_str.startswith("DIST_"):
                distance = float(distance_str[5:])
                self.ultrasonic_distance = distance
                self.detector.update_ultrasonic_distance(distance)
                # print(f"超声波距离更新: {distance} cm")
        except Exception as e:
            print(f"距离数据解析错误: {e}")
    
    def _send_servo_command(self, angles_deg):
        """
        发送角度命令到舵机
        
        参数:
        angles_deg -- 角度字典，包含底座、肩部、肘部、腕部的角度（度）
        """
        try:
            # 格式化为Alldro命令（在serial_comm.py中定义）
            base = angles_deg['base']
            shoulder = angles_deg['shoulder']
            elbow = angles_deg['elbow']
            wrist = angles_deg['wrist']
            
            # 发送命令
            self.serial.send_servo_action('Alldro', base, shoulder, elbow, wrist)
        except Exception as e:
            print(f"舵机命令发送错误: {e}")
    
    def _servo_control_thread(self):
        """舵机控制线程 - 处理平滑移动"""
        while self.running:
            try:
                if self.servo_smoothing and self._angles_changed():
                    # 生成平滑轨迹
                    current_tuple = (
                        self.current_angles['base'],
                        self.current_angles['shoulder'],
                        self.current_angles['elbow'],
                        self.current_angles['wrist']
                    )
                    
                    target_tuple = (
                        self.target_angles['base'],
                        self.target_angles['shoulder'],
                        self.target_angles['elbow'],
                        self.target_angles['wrist']
                    )
                    
                    # 生成平滑轨迹
                    trajectory = rp.generate_smooth_trajectory(
                        current_tuple, target_tuple, steps=self.servo_steps
                    )
                    
                    # 逐步执行轨迹
                    for i, angles in enumerate(trajectory[1:], 1):  # 跳过第一个点（当前位置）
                        if not self.running:
                            break
                            
                        # 更新当前角度
                        self.current_angles['base'] = angles[0]
                        self.current_angles['shoulder'] = angles[1]
                        self.current_angles['elbow'] = angles[2]
                        self.current_angles['wrist'] = angles[3]
                        
                        # 转换为度数并发送
                        angles_deg = self._angles_to_degrees(self.current_angles)
                        self._send_servo_command(angles_deg)
                        
                        # 控制更新频率
                        time.sleep(self.servo_update_interval)
                else:
                    # 没有角度变化或不使用平滑，休眠一段时间
                    time.sleep(0.1)
            except Exception as e:
                print(f"舵机控制线程错误: {e}")
                time.sleep(0.1)
    
    def _angles_changed(self):
        """检查目标角度是否与当前角度不同"""
        for joint in self.current_angles:
            if abs(self.current_angles[joint] - self.target_angles[joint]) > 0.01:
                return True
        return False
    
    def _calculate_smoothed_position(self, new_position):
        """
        平滑目标位置，减少抖动
        
        参数:
        new_position -- 新检测到的目标位置 (x, y, z)
        
        返回:
        smoothed_position -- 平滑后的位置
        """
        # 添加到历史记录
        self.position_history.append(new_position)
        if len(self.position_history) > self.history_length:
            self.position_history.pop(0)
        
        # 如果历史记录太少，直接返回新位置
        if len(self.position_history) < 3:
            return new_position
        
        # 计算加权平均
        weights = np.exp(np.linspace(0, 2, len(self.position_history))) - 1
        weights = weights / np.sum(weights)
        
        x_avg = sum(weights[i] * self.position_history[i][0] for i in range(len(self.position_history)))
        y_avg = sum(weights[i] * self.position_history[i][1] for i in range(len(self.position_history)))
        z_avg = sum(weights[i] * self.position_history[i][2] for i in range(len(self.position_history)))
        
        # 如果是新检测到的位置，使用平滑因子插值
        if len(self.position_history) > 1:
            prev_x, prev_y, prev_z = self.position_history[-2]
            x_avg = prev_x * self.position_smoothing + x_avg * (1 - self.position_smoothing)
            y_avg = prev_y * self.position_smoothing + y_avg * (1 - self.position_smoothing)
            z_avg = prev_z * self.position_smoothing + z_avg * (1 - self.position_smoothing)
        
        return (x_avg, y_avg, z_avg)
    
    def _face_tracking_loop(self):
        """人脸跟踪模式主循环"""
        print("\n=== 进入人脸跟踪模式 ===")
        
        # 设置检测器只检测人脸
        self.detector.set_target_classes(["face"])
        
        # 主循环
        while self.running and self.mode == 'face':
            current_time = time.time()
            
            # 控制检测和跟踪频率
            if current_time - self.last_tracking_time < self.tracking_interval:
                time.sleep(0.01)
                continue
            
            # 获取人脸检测结果
            target_coords, detection_info, frame = self.detector.detect_and_track()
            
            # 显示视频帧
            if frame is not None:
                cv2.imshow("人脸跟踪模式", frame)
                
                # 检查按键
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    self.mode = 'standby'
                    break
                elif key == ord('b'):  # 'b'键切换到书本模式
                    self.mode = 'book'
                    break
            
            # 处理检测结果
            if target_coords:
                # 平滑坐标
                smoothed_position = self._calculate_smoothed_position(target_coords)
                x_target, y_target, z_target = smoothed_position
                
                # 确保人脸z坐标在合理范围内 - 保留实际检测到的z值，如果是0则设为30cm
                if z_target <= 5:  # 如果z值过小，可能是错误检测或在桌面
                    z_target = 30  # 假定人脸在桌面以上30cm
                
                # 计算逆运动学 - 使用face模式
                try:
                    base_angle, shoulder_angle, elbow_angle, wrist_angle, ee_pos = rp.calculate_ik(
                        (x_target, y_target, z_target), 
                        distance=40,  # 在互动模式中保持一个舒适的距离
                        lamp_mode="face"
                    )
                    
                    # 更新目标角度
                    self.target_angles['base'] = base_angle
                    self.target_angles['shoulder'] = shoulder_angle
                    self.target_angles['elbow'] = elbow_angle
                    self.target_angles['wrist'] = wrist_angle
                    
                    # 打印调试信息
                    print(f"人脸坐标: ({x_target:.1f}, {y_target:.1f}, {z_target:.1f})")
                    print(f"目标角度(度): 底座={np.degrees(base_angle):.1f}°, "
                          f"肩部={np.degrees(shoulder_angle):.1f}°, "
                          f"肘部={np.degrees(elbow_angle):.1f}°, "
                          f"腕部={np.degrees(wrist_angle):.1f}°")
                    
                except Exception as e:
                    print(f"人脸跟踪逆运动学错误: {e}")
            
            self.last_tracking_time = current_time
        
        # 清理资源
        cv2.destroyAllWindows()
        print("退出人脸跟踪模式")
    
    def _book_tracking_loop(self):
        """书本跟踪模式主循环"""
        print("\n=== 进入书本跟踪模式 ===")
        
        # 设置检测器只检测书本
        self.detector.set_target_classes(["book"])
        
        # 书本模式的最佳照明距离（厘米）- 权衡护眼和照明效果
        optimal_distance = 45
        
        # 主循环
        while self.running and self.mode == 'book':
            current_time = time.time()
            
            # 控制检测和跟踪频率
            if current_time - self.last_tracking_time < self.tracking_interval:
                time.sleep(0.01)
                continue
            
            # 获取书本检测结果
            target_coords, detection_info, frame = self.detector.detect_and_track()
            
            # 显示视频帧
            if frame is not None:
                cv2.imshow("书本跟踪模式", frame)
                
                # 检查按键
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    self.mode = 'standby'
                    break
                elif key == ord('f'):  # 'f'键切换到人脸模式
                    self.mode = 'face'
                    break
            
            # 处理检测结果
            if target_coords:
                # 平滑坐标
                smoothed_position = self._calculate_smoothed_position(target_coords)
                x_target, y_target, z_target = smoothed_position
                
                # 书本在桌面上，z=0
                z_target = 0
                
                # 计算逆运动学 - 使用book模式
                try:
                    base_angle, shoulder_angle, elbow_angle, wrist_angle, ee_pos = rp.calculate_ik(
                        (x_target, y_target, z_target), 
                        distance=optimal_distance,  # 使用最适合阅读的距离
                        lamp_mode="book"
                    )
                    
                    # 更新目标角度
                    self.target_angles['base'] = base_angle
                    self.target_angles['shoulder'] = shoulder_angle
                    self.target_angles['elbow'] = elbow_angle
                    self.target_angles['wrist'] = wrist_angle
                    
                    # 打印调试信息
                    print(f"书本坐标: ({x_target:.1f}, {y_target:.1f}, {z_target:.1f})")
                    print(f"目标角度(度): 底座={np.degrees(base_angle):.1f}°, "
                          f"肩部={np.degrees(shoulder_angle):.1f}°, "
                          f"肘部={np.degrees(elbow_angle):.1f}°, "
                          f"腕部={np.degrees(wrist_angle):.1f}°")
                    
                except Exception as e:
                    print(f"书本跟踪逆运动学错误: {e}")
            
            self.last_tracking_time = current_time
        
        # 清理资源
        cv2.destroyAllWindows()
        print("退出书本跟踪模式")
    
    def _standby_mode(self):
        """待机模式处理"""
        print("\n=== 进入待机模式 ===")
        
        # 重置机械臂姿态
        self._reset_position()
        
        # 显示菜单
        print("台灯控制系统 - 请选择模式:")
        print("1. 人脸跟踪模式 - 台灯会跟踪人脸")
        print("2. 书本跟踪模式 - 台灯会跟踪并照亮书本")
        print("3. 退出系统")
        
        # 根据用户输入切换模式
        while self.running and self.mode == 'standby':
            try:
                choice = input("请选择 (1/2/3): ")
                if choice == '1':
                    self.mode = 'face'
                    break
                elif choice == '2':
                    self.mode = 'book'
                    break
                elif choice == '3':
                    self.running = False
                    break
                else:
                    print("无效选择，请重试")
            except Exception as e:
                print(f"输入错误: {e}")
    
    def _main_loop(self):
        """主控制循环"""
        # 启动串口监听
        self.serial.set_mode_change_callback(lambda: print("串口模式变更回调触发"))
        
        # 主状态机循环
        while self.running:
            if self.mode == 'standby':
                self._standby_mode()
            elif self.mode == 'face':
                self._face_tracking_loop()
            elif self.mode == 'book':
                self._book_tracking_loop()
            else:
                print(f"未知模式: {self.mode}")
                self.mode = 'standby'
    
    def start(self):
        """启动系统"""
        if self.running:
            print("系统已经在运行")
            return
            
        self.running = True
        
        # 启动舵机控制线程
        self.servo_thread = threading.Thread(
            target=self._servo_control_thread,
            daemon=True
        )
        self.servo_thread.start()
        
        # 启动主控制循环
        self.main_thread = threading.Thread(
            target=self._main_loop,
            daemon=False  # 非守护线程，程序会等待此线程结束
        )
        self.main_thread.start()
        
        print("系统已启动")
    
    def stop(self):
        """停止系统"""
        self.running = False
        
        # 等待线程结束
        if self.main_thread and self.main_thread.is_alive():
            self.main_thread.join(timeout=2.0)
        
        if self.servo_thread and self.servo_thread.is_alive():
            self.servo_thread.join(timeout=1.0)
        
        # 释放资源
        self.detector.release()
        self.serial.stop()
        cv2.destroyAllWindows()
        
        print("系统已停止")
    
    def __del__(self):
        """析构函数，确保资源释放"""
        self.stop()

# 主程序入口
if __name__ == "__main__":
    try:
        controller = SmartLampController(
            camera_index=0,          # 摄像头索引
            model_path="yolov8n.pt", # YOLO模型路径
            serial_port="/dev/ttyAMA0",  # 树莓派串口设备
            baudrate=9600            # 波特率
        )
        
        controller.start()
        
        # 主线程等待程序结束
        print("按Ctrl+C退出程序")
        while controller.running:
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"程序异常: {e}")
    finally:
        if 'controller' in locals():
            controller.stop()
        print("程序已退出")
