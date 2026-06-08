import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 机械臂参数（与book.py和face.py相同）
L1 = 8    # 底座高度
L2 = 24   # 肩部到肘部长度
L3 = 24   # 肘部到腕部长度
L4 = 8    # 腕部到末端执行器长度（摄像头/灯罩）

# 关节角度限制（以弧度计）
BASE_LIMITS = (-np.pi * 135/180, np.pi * 135/180)     # 底座转动范围
SHOULDER_LIMITS = (-np.pi * 90/180, np.pi * 90/180)   # 肩部转动范围
ELBOW_LIMITS = (0, np.pi * 150/180)                   # 肘部转动范围
WRIST_LIMITS = (0, np.pi * 130/180)                   # 腕部转动范围

# 人脸跟踪专用参数
FIXED_SHOULDER_ANGLE = np.pi * 60/180  # 肩部固定60度向上（用于人脸跟踪）
FIXED_ELBOW_ANGLE = np.pi * 30/180     # 肘部固定30度向上（用于人脸跟踪）

class ObjectTracker:
    def __init__(self):
        """初始化对象跟踪器，统一处理人脸和书本的跟踪逻辑"""
        pass
    
    # ===================== 人脸跟踪功能 =====================
    def calculate_face_ik(self, target_pos):
        x_target, y_target, z_target = target_pos
        
        # 计算底座角度（绕z轴旋转）
        base_angle = np.arctan2(y_target, x_target)
        base_angle = np.clip(base_angle, BASE_LIMITS[0], BASE_LIMITS[1])
        
        # 固定肩部和肘部角度
        shoulder_angle = FIXED_SHOULDER_ANGLE
        elbow_angle = FIXED_ELBOW_ANGLE
        
        # 计算腕部位置（固定肩肘角度）
        forearm_angle = shoulder_angle - elbow_angle
        x_elbow = L2 * np.cos(shoulder_angle) * np.cos(base_angle)
        y_elbow = L2 * np.cos(shoulder_angle) * np.sin(base_angle)
        z_elbow = L1 + L2 * np.sin(shoulder_angle)
        
        x_wrist = x_elbow + L3 * np.cos(forearm_angle) * np.cos(base_angle)
        y_wrist = y_elbow + L3 * np.cos(forearm_angle) * np.sin(base_angle)
        z_wrist = z_elbow + L3 * np.sin(forearm_angle)
        wrist_pos = np.array([x_wrist, y_wrist, z_wrist])
        
        # 计算末端执行器到目标的方向向量
        direction_to_target = np.array([x_target, y_target, z_target]) - wrist_pos
        if np.linalg.norm(direction_to_target) > 0:
            direction_to_target /= np.linalg.norm(direction_to_target)
        
        # 计算腕部角度
        forearm_dir = np.array([
            np.cos(forearm_angle) * np.cos(base_angle),
            np.cos(forearm_angle) * np.sin(base_angle),
            np.sin(forearm_angle)
        ])
        cross_prod = np.cross(forearm_dir, direction_to_target)
        dot_prod = np.dot(forearm_dir, direction_to_target)
        wrist_angle = np.arctan2(np.linalg.norm(cross_prod), dot_prod)
        wrist_angle = np.clip(wrist_angle, WRIST_LIMITS[0], WRIST_LIMITS[1])
        
        # 计算末端执行器位置
        end_angle = forearm_angle + wrist_angle
        x_end = x_wrist + L4 * np.cos(end_angle) * np.cos(base_angle)
        y_end = y_wrist + L4 * np.cos(end_angle) * np.sin(base_angle)
        z_end = z_wrist + L4 * np.sin(end_angle)
        end_effector_pos = np.array([x_end, y_end, z_end])
        
        return base_angle, wrist_angle, end_effector_pos
    
    def face_forward_kinematics(self, base_angle, wrist_angle):
        # 固定肩部和肘部角度
        shoulder_angle = FIXED_SHOULDER_ANGLE
        elbow_angle = FIXED_ELBOW_ANGLE
        
        # 底座位置
        base_pos = np.array([0, 0, 0])
        
        # 肩部位置
        shoulder_pos = np.array([0, 0, L1])
        
        # 肘部位置
        x_elbow = L2 * np.cos(shoulder_angle) * np.cos(base_angle)
        y_elbow = L2 * np.cos(shoulder_angle) * np.sin(base_angle)
        z_elbow = L1 + L2 * np.sin(shoulder_angle)
        elbow_pos = np.array([x_elbow, y_elbow, z_elbow])
        
        # 腕部位置
        forearm_angle = shoulder_angle - elbow_angle
        x_wrist = x_elbow + L3 * np.cos(forearm_angle) * np.cos(base_angle)
        y_wrist = y_elbow + L3 * np.cos(forearm_angle) * np.sin(base_angle)
        z_wrist = z_elbow + L3 * np.sin(forearm_angle)
        wrist_pos = np.array([x_wrist, y_wrist, z_wrist])
        
        # 末端执行器位置
        end_angle = forearm_angle + wrist_angle
        x_end = x_wrist + L4 * np.cos(end_angle) * np.cos(base_angle)
        y_end = y_wrist + L4 * np.cos(end_angle) * np.sin(base_angle)
        z_end = z_wrist + L4 * np.sin(end_angle)
        end_pos = np.array([x_end, y_end, z_end])
        
        return [base_pos, shoulder_pos, elbow_pos, wrist_pos, end_pos]
    
    # ===================== 书本跟踪功能 =====================
    def calculate_book_ik(self, target_pos, distance=40):
        x_target, y_target, z_target = target_pos
        
        # 书本跟踪参数设置
        target_distance = min(max(distance, 40), 50)  # 限制在40-50cm之间
        height_offset = 30  # 灯相对于书本的高度
        
        # 计算朝向目标的方向向量
        target_xy_distance = np.sqrt(x_target**2 + y_target**2)
        
        # 计算基座到目标的方向向量
        if target_xy_distance > 0:
            base_to_target_dir = np.array([x_target, y_target]) / target_xy_distance
        else:
            base_to_target_dir = np.array([1, 0])  # 默认朝前
            
        # 灯的位置应该在目标点上方，并稍微向基座方向偏移
        horizontal_offset = min(0.25 * target_xy_distance, 15)
        x_ee = x_target - horizontal_offset * base_to_target_dir[0]
        y_ee = y_target - horizontal_offset * base_to_target_dir[1]
        z_ee = z_target + height_offset
        
        # 设置末端执行器位置
        end_effector_pos = np.array([x_ee, y_ee, z_ee])
        target_vec = np.array([x_target, y_target, z_target])
        
        # 确保末端执行器到目标的距离与指定的distance相近
        current_distance = np.linalg.norm(end_effector_pos - target_vec)
        if current_distance > 0:
            ratio = target_distance / current_distance
            direction = (end_effector_pos - target_vec) / current_distance
            end_effector_pos = target_vec + direction * target_distance
        
        # 更新末端执行器位置坐标
        x_ee, y_ee, z_ee = end_effector_pos
        
        # 计算底座角度（绕z轴旋转）
        base_angle = np.arctan2(y_ee, x_ee)
        base_angle = np.clip(base_angle, BASE_LIMITS[0], BASE_LIMITS[1])
        
        # 计算XY平面中从底座到末端执行器的距离
        r_xy = np.sqrt(x_ee**2 + y_ee**2)
        
        # 计算腕部位置
        direction_to_target = (target_vec - end_effector_pos)
        if np.linalg.norm(direction_to_target) > 0:
            direction_to_target /= np.linalg.norm(direction_to_target)
        wrist_pos = end_effector_pos - L4 * direction_to_target
        
        # 转换为2D平面问题
        r = np.sqrt(wrist_pos[0]**2 + wrist_pos[1]**2)
        z = wrist_pos[2] - L1
        
        # 从肩部到腕部的距离
        d = np.sqrt(r**2 + z**2)
        
        # 调整可达范围
        if d > L2 + L3:
            direction_to_wrist = np.array([r, z])
            direction_to_wrist /= np.linalg.norm(direction_to_wrist)
            max_d = L2 + L3 - 0.1
            r, z = direction_to_wrist * max_d
            d = max_d
        elif d < abs(L2 - L3):
            direction_to_wrist = np.array([r, z])
            if np.linalg.norm(direction_to_wrist) > 0:
                direction_to_wrist /= np.linalg.norm(direction_to_wrist)
                min_d = abs(L2 - L3) + 0.1
                r, z = direction_to_wrist * min_d
                d = min_d
        
        # 使用余弦定理计算肘部角度
        cos_elbow = (L2**2 + L3**2 - d**2) / (2 * L2 * L3)
        elbow_angle = np.pi - np.arccos(np.clip(cos_elbow, -1.0, 1.0))
        elbow_angle = np.clip(elbow_angle, ELBOW_LIMITS[0], ELBOW_LIMITS[1])
        
        # 计算肩部角度
        alpha = np.arctan2(z, r)
        beta = np.arccos(np.clip((L2**2 + d**2 - L3**2) / (2 * L2 * d), -1.0, 1.0))
        shoulder_angle = alpha + beta
        shoulder_angle = np.clip(shoulder_angle, SHOULDER_LIMITS[0], SHOULDER_LIMITS[1])
        
        # 计算腕部角度
        forearm_angle = shoulder_angle - elbow_angle
        forearm_dir = np.array([
            np.cos(forearm_angle) * np.cos(base_angle),
            np.cos(forearm_angle) * np.sin(base_angle),
            np.sin(forearm_angle)
        ])
        cross_prod = np.cross(forearm_dir, direction_to_target)
        dot_prod = np.dot(forearm_dir, direction_to_target)
        wrist_angle = np.arctan2(np.linalg.norm(cross_prod), dot_prod)
        wrist_angle = np.clip(wrist_angle, WRIST_LIMITS[0], WRIST_LIMITS[1])
        
        return base_angle, shoulder_angle, elbow_angle, wrist_angle, end_effector_pos
    
    def book_forward_kinematics(self, base_angle, shoulder_angle, elbow_angle, wrist_angle, end_effector_pos=None):
        # 底座位置
        base_pos = np.array([0, 0, 0])
        
        # 肩部位置
        shoulder_pos = np.array([0, 0, L1])
        
        # 计算肘部位置
        x_elbow = L2 * np.cos(shoulder_angle) * np.cos(base_angle)
        y_elbow = L2 * np.cos(shoulder_angle) * np.sin(base_angle)
        z_elbow = L1 + L2 * np.sin(shoulder_angle)
        elbow_pos = np.array([x_elbow, y_elbow, z_elbow])
        
        # 计算腕部位置
        forearm_angle = shoulder_angle - elbow_angle
        x_wrist = x_elbow + L3 * np.cos(forearm_angle) * np.cos(base_angle)
        y_wrist = y_elbow + L3 * np.cos(forearm_angle) * np.sin(base_angle)
        z_wrist = z_elbow + L3 * np.sin(forearm_angle)
        wrist_pos = np.array([x_wrist, y_wrist, z_wrist])
        
        # 计算末端执行器位置（如果未传入逆运动学结果，则按原逻辑计算）
        if end_effector_pos is None:
            end_angle = forearm_angle + wrist_angle
            x_end = x_wrist + L4 * np.cos(end_angle) * np.cos(base_angle)
            y_end = y_wrist + L4 * np.cos(end_angle) * np.sin(base_angle)
            z_end = z_wrist + L4 * np.sin(end_angle)
            end_pos = np.array([x_end, y_end, z_end])
        else:
            end_pos = end_effector_pos
        
        return [base_pos, shoulder_pos, elbow_pos, wrist_pos, end_pos]
    
    # ===================== 共用工具函数 =====================
    def generate_smooth_trajectory(self, start_angles, end_angles, steps=20):
        start_array = np.array(start_angles)
        end_array = np.array(end_angles)
        
        # 创建轨迹
        trajectory = []
        
        # 使用余弦平滑插值 - 在加速和减速阶段更平滑
        for i in range(steps + 1):
            # 余弦插值因子 (0->1)
            t = (1 - np.cos(i * np.pi / steps)) / 2
            
            # 计算当前步骤的角度
            current_angles = start_array * (1 - t) + end_array * t
            trajectory.append(tuple(current_angles))
        
        return trajectory
    
    def plot_robotic_arm(self, joint_positions, target_pos=None, end_effector_pos=None, title="机械臂姿态可视化"):
        # 转换为numpy数组方便处理
        positions = np.array(joint_positions)
        
        # 提取各关节坐标
        base = positions[0]
        shoulder = positions[1]
        elbow = positions[2]
        wrist = positions[3]
        end = positions[4]
        
        # 创建3D图形对象
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制连杆结构
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2],
                'o-', markersize=8, linewidth=3, color='#2E86C1')
        
        # 绘制各关节标记
        joint_labels = ['底座', '肩部', '肘部', '腕部', '末端']
        for i, (x, y, z) in enumerate(positions):
            ax.scatter(x, y, z, s=100, label=joint_labels[i])
        
        # 绘制摄像头位置（末端执行器中心）
        if end_effector_pos is not None:
            ax.scatter(end_effector_pos[0], end_effector_pos[1], end_effector_pos[2], 
                       color='blue', marker='^', s=100, label='摄像头')
        
        # 添加目标点显示
        if target_pos is not None:
            ax.scatter(target_pos[0], target_pos[1], target_pos[2],
                       s=150, c='r', marker='*', label='目标')
            
        # 绘制从末端到目标的连线(红色实线)
        if end_effector_pos is not None and target_pos is not None:
            ax.plot([end_effector_pos[0], target_pos[0]],
                    [end_effector_pos[1], target_pos[1]],
                    [end_effector_pos[2], target_pos[2]],
                    '-', color='red', linewidth=2)
            # 绘制腕部到末端的连线(蓝色实线)
            wrist = positions[3]  # 从joint_positions中提取腕部位置
            ax.plot([wrist[0], end_effector_pos[0]],
                    [wrist[1], end_effector_pos[1]],
                    [wrist[2], end_effector_pos[2]],
                    '-', color='blue', linewidth=2)
                
            # 显示距离
            distance = np.linalg.norm(np.array(end_effector_pos) - np.array(target_pos))
            mid_point = (np.array(end_effector_pos) + np.array(target_pos)) / 2
            ax.text(mid_point[0], mid_point[1], mid_point[2], 
                    f'{distance:.1f}cm', color='red', fontsize=10)
        
        # 绘制机械臂坐标系
        origin = np.zeros(3)
        ax.quiver(origin[0], origin[1], origin[2], 15, 0, 0, color='r', arrow_length_ratio=0.1, label='X轴')
        ax.quiver(origin[0], origin[1], origin[2], 0, 15, 0, color='g', arrow_length_ratio=0.1, label='Y轴')
        ax.quiver(origin[0], origin[1], origin[2], 0, 0, 15, color='b', arrow_length_ratio=0.1, label='Z轴')
        
        # 设置坐标轴参数
        max_range = max(L2, L3) * 2
        ax.set_xlabel('X (cm)')
        ax.set_ylabel('Y (cm)')
        ax.set_zlabel('Z (cm)')
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([0, L1 + L2 + L3 + L4 + 10])
        
        # 添加桌面平面（z=0）
        xx, yy = np.meshgrid(np.linspace(-max_range, max_range, 2), 
                             np.linspace(-max_range, max_range, 2))
        zz = np.zeros(xx.shape)
        ax.plot_surface(xx, yy, zz, alpha=0.2, color='gray')
        
        # 设置观察视角（俯仰角，方位角）
        ax.view_init(elev=30, azim=45)
        
        # 添加图例和标题
        plt.legend(loc='upper left', bbox_to_anchor=(0.9, 0.9))
        plt.title(f"{title}\n底座到末端长度：{np.linalg.norm(end - base):.1f}cm")
        plt.tight_layout()
        plt.show()
    
    def angles_to_degrees(self, *angles):
        """将弧度角度转换为度数"""
        return [np.degrees(angle) for angle in angles]
    
    # ===================== 公共接口方法 =====================
    def track_face(self, face_coordinates):
        base_angle, wrist_angle, end_effector_pos = self.calculate_face_ik(face_coordinates)
        return base_angle, wrist_angle
    
    def track_book(self, book_coordinates, distance=45):
        base_angle, shoulder_angle, elbow_angle, wrist_angle, end_effector_pos = self.calculate_book_ik(
            book_coordinates, distance
        )
        
        return base_angle, shoulder_angle, elbow_angle, wrist_angle


# 测试代码
if __name__ == "__main__":
    tracker = ObjectTracker()
    
    # 测试人脸跟踪
    print("\n===== 测试人脸跟踪功能 =====")
    face_positions = [
        (60, 0, 60),    # 正前方
        (60, 30, 60),   # 右前方
        (-30, 30, 50),  # 左前方
        (0, 40, 50),    # 正右侧
    ]
    for pos in face_positions:
        print(f"\n跟踪人脸位置: {pos}")
        base_angle, wrist_angle = tracker.track_face(pos)
        print(f"底座角度: {np.degrees(base_angle):.1f}°")
        print(f"腕部角度: {np.degrees(wrist_angle):.1f}°")
        
        # 计算正向运动学并可视化（仅用于测试）
        joint_positions = tracker.face_forward_kinematics(base_angle, wrist_angle)
        tracker.plot_robotic_arm(joint_positions, target_pos=pos, title=f"人脸跟踪 @ {pos}")
    
    # 测试书本跟踪
    print("\n===== 测试书本跟踪功能 =====")
    book_positions = [
        (40, 0, 0),      # 正前方的书本
        (0, 40, 0),      # 右侧的书本
        (-30, 0, 0),     # 左前方的书本
        (30, 30, 0),     # 右前方的书本
    ]
    for pos in book_positions:
        print(f"\n跟踪书本位置: {pos}")
        base_angle, shoulder_angle, elbow_angle, wrist_angle = tracker.track_book(pos)
        angles_deg = tracker.angles_to_degrees(base_angle, shoulder_angle, elbow_angle, wrist_angle)
        print(f"底座角度: {angles_deg[0]:.1f}°")
        print(f"肩部角度: {angles_deg[1]:.1f}°")
        print(f"肘部角度: {angles_deg[2]:.1f}°")
        print(f"腕部角度: {angles_deg[3]:.1f}°")
        
        # 计算正向运动学并可视化（仅用于测试）
        _, _, _, _, end_pos = tracker.calculate_book_ik(pos)
        joint_positions = tracker.book_forward_kinematics(
            base_angle, shoulder_angle, elbow_angle, wrist_angle, end_pos
        )
        tracker.plot_robotic_arm(joint_positions, target_pos=pos, 
                               end_effector_pos=end_pos, title=f"书本跟踪 @ {pos}")
