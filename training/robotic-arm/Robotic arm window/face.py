import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 机械臂参数（与book.py相同）
L1 = 8    # 底座高度
L2 = 24   # 肩部到肘部长度
L3 = 24   # 肘部到腕部长度
L4 = 8    # 腕部到末端执行器长度（摄像头/灯罩）

# 关节角度限制（以弧度计）
BASE_LIMITS = (-np.pi * 135/180, np.pi * 135/180)     # 底座转动范围
SHOULDER_LIMITS = (-np.pi * 90/180, np.pi * 90/180)   # 肩部固定角度
ELBOW_LIMITS = (0, np.pi * 150/180)                   # 肘部固定角度
WRIST_LIMITS = (0, np.pi * 130/180)                   # 腕部转动范围

# 固定角度设置（0度与Z轴重合，向X轴正方向为正）
FIXED_SHOULDER_ANGLE = np.pi * 60/180  # 肩部固定60度向上
FIXED_ELBOW_ANGLE = np.pi * 30/180     # 肘部固定30度向上

def calculate_face_ik(target_pos):
    """
    计算人脸跟踪的逆运动学，只控制底座和腕部舵机
    
    参数:
    target_pos -- 人脸位置的元组 (x, y, z)
    
    返回:
    base_angle, wrist_angle -- 底座和腕部角度（弧度）
    end_effector_pos -- 末端执行器位置
    """
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

def forward_kinematics(base_angle, wrist_angle):
    """
    正向运动学计算（简化版，固定肩肘角度）
    """
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

def plot_robotic_arm(joint_positions, target_pos=None, title="人脸跟踪机械臂姿态"):
    """可视化机械臂姿态（从book.py复制并简化）"""
    positions = np.array(joint_positions)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制连杆结构
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2],
            'o-', markersize=8, linewidth=3, color='#2E86C1')
    
    # 绘制各关节标记
    joint_labels = ['底座', '肩部', '肘部', '腕部', '摄像头']
    for i, (x, y, z) in enumerate(positions):
        ax.scatter(x, y, z, s=100, label=joint_labels[i])
    
    # 添加目标点显示（人脸位置）
    if target_pos is not None:
        ax.scatter(target_pos[0], target_pos[1], target_pos[2],
                   s=150, c='r', marker='*', label='人脸')
    
    # 设置坐标轴
    max_range = max(L2, L3) * 2
    ax.set_xlabel('X (cm)')
    ax.set_ylabel('Y (cm)')
    ax.set_zlabel('Z (cm)')
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([0, L1 + L2 + L3 + L4 + 10])
    
    # 添加桌面平面
    xx, yy = np.meshgrid(np.linspace(-max_range, max_range, 2), 
                         np.linspace(-max_range, max_range, 2))
    zz = np.zeros(xx.shape)
    ax.plot_surface(xx, yy, zz, alpha=0.2, color='gray')
    
    ax.view_init(elev=30, azim=45)
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.show()

def test_face_tracking(face_positions=None):
    """测试人脸跟踪功能"""
    if face_positions is None:
        # 默认测试坐标点（人脸位置）
        face_positions = [
            (60, 0, 60),    # 正前方
            (60, 30, 60),   # 右前方
            (-30, 30, 50),  # 左前方
            (0, 40, 50),    # 正右侧
            (0, -40, 50)    # 正左侧
        ]
    
    print("===== 人脸跟踪测试 =====")
    for pos in face_positions:
        print(f"\n跟踪人脸位置: {pos}")
        
        # 计算逆运动学
        base_angle, wrist_angle, end_pos = calculate_face_ik(pos)
        
        # 计算正向运动学
        joint_positions = forward_kinematics(base_angle, wrist_angle)
        
        # 显示结果
        print(f"底座角度: {np.degrees(base_angle):.1f}°")
        print(f"腕部角度: {np.degrees(wrist_angle):.1f}°")
        print(f"摄像头位置: ({end_pos[0]:.1f}, {end_pos[1]:.1f}, {end_pos[2]:.1f})")
        
        # 可视化
        title = f"人脸跟踪 @ {pos}"
        plot_robotic_arm(joint_positions, target_pos=pos, title=title)

if __name__ == "__main__":
    test_face_tracking()
