import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 机械臂参数
L1 = 8    # 底座高度
L2 = 24   # 肩部到肘部长度
L3 = 24   # 肘部到腕部长度
L4 = 7    # 腕部到末端执行器长度

# 关节角度范围 - 已移除限制，允许全范围运动以确保可达性
# 在实际应用中，可以根据硬件限制重新添加角度约束

def calculate_ik(target_pos, distance=40, height_offset=25):
    """
    计算四轴机械臂的逆运动学，使台灯以俯视角度朝向目标点。
    
    参数:
    target_pos -- 目标位置的元组 (x, y, z)
    distance -- 末端执行器与目标之间的期望距离（厘米）
    height_offset -- 末端执行器相对于目标点的高度偏移（厘米）
    
    返回:
    base_angle, shoulder_angle, elbow_angle, wrist_angle -- 关节角度（弧度）
    end_effector_pos -- 应用计算的角度后末端执行器的位置
    """
    x_target, y_target, z_target = target_pos
    
    # 计算一个更好的末端执行器位置，使其从上方照射目标
    # 首先获取目标点的平面距离和方向
    target_xy_distance = np.sqrt(x_target**2 + y_target**2)
    target_xy_dir = np.array([x_target, y_target]) / (target_xy_distance if target_xy_distance > 0 else 1.0)
    
    # 根据目标位置决定高度和水平偏移
    # 距离越远，高度偏移越小，以保持合理俯视角度
    adjusted_height = min(height_offset, target_xy_distance * 0.5 + 15)
    
    # 决定水平压缩因子 - 使末端执行器向基座方向偏移
    # 目标越远，偏移越大，以避免机械臂过度伸展
    horizontal_compress = min(0.3, target_xy_distance / 200)
    
    # 计算末端执行器的位置
    x_ee = x_target * (1 - horizontal_compress)
    y_ee = y_target * (1 - horizontal_compress)
    z_ee = z_target + adjusted_height
    
    end_effector_pos = np.array([x_ee, y_ee, z_ee])
    
    # 确保末端执行器距离目标点的距离接近所需的distance
    target_vec = np.array([x_target, y_target, z_target])
    actual_distance = np.linalg.norm(end_effector_pos - target_vec)
    
    # 通过缩放调整到期望距离
    if actual_distance > 0:
        scale_factor = distance / actual_distance
        direction = (end_effector_pos - target_vec) / actual_distance
        end_effector_pos = target_vec + direction * distance
    
    x_ee, y_ee, z_ee = end_effector_pos
    
    # 计算底座角度（绕z轴旋转）
    base_angle = np.arctan2(y_ee, x_ee)
    
    # 底座角度已无限制，可以完全旋转
    
    # 计算XY平面中从底座到末端执行器的距离
    r_xy = np.sqrt(x_ee**2 + y_ee**2)
    
    # 目标的方向向量（从末端执行器指向目标）
    direction_to_target = target_vec - end_effector_pos
    direction_to_target = direction_to_target / np.linalg.norm(direction_to_target)
    
    # 计算腕部位置，使末端执行器指向目标
    wrist_pos = end_effector_pos - L4 * direction_to_target
    x_wrist, y_wrist, z_wrist = wrist_pos
    r_wrist = np.sqrt(x_wrist**2 + y_wrist**2)
    
    # 现在解决二连杆平面机构（肩部到腕部）
    # 肩部在 (0, 0, L1)，我们需要到达 (r_wrist, z_wrist)
    
    # 转换为2D问题
    r = r_wrist
    z = z_wrist - L1
    
    # 从肩部到腕部的距离
    d = np.sqrt(r**2 + z**2)
    
    # 检查点是否可达 - 如果不可达则尝试调整末端执行器位置
    if d > L2 + L3:
        # 点太远，尝试将腕部点向肩部移动
        direction_to_shoulder = np.array([-r, -z])
        direction_to_shoulder = direction_to_shoulder / np.linalg.norm(direction_to_shoulder)
        
        # 计算需要移动的距离
        excess = d - (L2 + L3 - 0.1)  # 留一点余量
        
        # 移动腕部点
        r -= direction_to_shoulder[0] * excess
        z -= direction_to_shoulder[1] * excess
        
        # 重新计算距离
        d = np.sqrt(r**2 + z**2)
    
    if d < abs(L2 - L3):
        # 点太近，尝试将腕部点远离肩部
        direction_from_shoulder = np.array([r, z])
        if np.linalg.norm(direction_from_shoulder) > 0:
            direction_from_shoulder = direction_from_shoulder / np.linalg.norm(direction_from_shoulder)
            
            # 计算需要移动的距离
            deficit = (abs(L2 - L3) + 0.1) - d  # 留一点余量
            
            # 移动腕部点
            r += direction_from_shoulder[0] * deficit
            z += direction_from_shoulder[1] * deficit
            
            # 重新计算距离
            d = np.sqrt(r**2 + z**2)
    
    # 使用余弦定理找到肘部角度
    cos_elbow = (L2**2 + L3**2 - d**2) / (2 * L2 * L3)
    elbow_angle = np.arccos(np.clip(cos_elbow, -1.0, 1.0))
    
    # 我们计算的肘部角度是内角，需要转换为机器人的约定
    elbow_angle = np.pi - elbow_angle
    
    # 肘部角度已无限制，但保持在合理范围内
    if elbow_angle > np.pi * 170/180:
        elbow_angle = np.pi * 170/180  # 避免接近完全伸直
    
    # 计算肩部角度
    alpha = np.arctan2(z, r)
    beta = np.arccos(np.clip((L2**2 + d**2 - L3**2) / (2 * L2 * d), -1.0, 1.0))
    shoulder_angle = alpha + beta
    
    # 肩部角度已无限制
    
    # 计算腕部角度以指向目标
    # 我们需要末端执行器指向目标
    # 首先，计算前臂（肘部到腕部）的角度
    forearm_angle = shoulder_angle - elbow_angle
    
    # 计算从腕部到目标的方向
    wrist_to_target = target_vec - wrist_pos
    wrist_to_target = wrist_to_target / np.linalg.norm(wrist_to_target)
    
    # 计算二维平面内的角度（在base_angle定义的平面内）
    # 首先，将向量投影到该平面
    wrist_to_target_2d = np.array([
        np.sqrt(wrist_to_target[0]**2 + wrist_to_target[1]**2),
        wrist_to_target[2]
    ])
    wrist_to_target_2d = wrist_to_target_2d / np.linalg.norm(wrist_to_target_2d)
    
    # 计算角度
    forearm_dir = np.array([np.cos(forearm_angle), np.sin(forearm_angle)])
    wrist_angle = np.arctan2(wrist_to_target_2d[1], wrist_to_target_2d[0]) - np.arctan2(forearm_dir[1], forearm_dir[0])
    
    # 规范化角度到-pi到pi之间
    wrist_angle = (wrist_angle + np.pi) % (2 * np.pi) - np.pi
    
    # 腕部角度已无限制
    
    return base_angle, shoulder_angle, elbow_angle, wrist_angle, end_effector_pos

def forward_kinematics(base_angle, shoulder_angle, elbow_angle, wrist_angle):
    """
    计算四轴机械臂的正向运动学。
    
    参数:
    base_angle, shoulder_angle, elbow_angle, wrist_angle -- 关节角度（弧度）
    
    返回:
    joint_positions -- 每个关节和末端执行器的(x, y, z)位置列表
    """
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
    
    # 计算末端执行器位置
    end_angle = forearm_angle + wrist_angle
    x_end = x_wrist + L4 * np.cos(end_angle) * np.cos(base_angle)
    y_end = y_wrist + L4 * np.cos(end_angle) * np.sin(base_angle)
    z_end = z_wrist + L4 * np.sin(end_angle)
    end_pos = np.array([x_end, y_end, z_end])
    
    return [base_pos, shoulder_pos, elbow_pos, wrist_pos, end_pos]

def plot_robotic_arm(joint_positions, target_pos=None, end_effector_pos=None, title="机械臂姿态可视化"):
    """
    可视化机械臂的3D姿态
    
    参数:
    joint_positions -- 正运动学计算的关节位置列表（包含末端执行器）
    target_pos -- 目标点坐标 (可选)
    end_effector_pos -- 末端执行器计算位置 (可选)
    """
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
    joint_labels = ['底座', '肩', '肘', '腕', '末端']
    for i, (x, y, z) in enumerate(positions):
        ax.scatter(x, y, z, s=100, label=joint_labels[i])
    
    # 添加目标点显示
    if target_pos is not None:
        ax.scatter(target_pos[0], target_pos[1], target_pos[2],
                   s=150, c='r', marker='*', label='目标点')
    
    # 绘制末端误差连线（如果同时提供计算末端和目标点）
    if target_pos is not None and end_effector_pos is not None:
        ax.plot([end_effector_pos[0], target_pos[0]],
                [end_effector_pos[1], target_pos[1]],
                [end_effector_pos[2], target_pos[2]],
                '--', color='gray', linewidth=1)
        
        # 计算并显示距离
        distance = np.linalg.norm(np.array(end_effector_pos) - np.array(target_pos))
        mid_point = (np.array(end_effector_pos) + np.array(target_pos)) / 2
        ax.text(mid_point[0], mid_point[1], mid_point[2], 
                f'{distance:.1f}cm', color='red', fontsize=10)
    
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

def demo_ik_with_visualization(target_pos, height_offset=25):
    """
    演示逆运动学并可视化机械臂。
    
    参数:
    target_pos -- 目标位置的元组 (x, y, z)
    height_offset -- 末端执行器相对于目标点的高度偏移（厘米）
    """
    try:
        # 计算逆运动学
        base_angle, shoulder_angle, elbow_angle, wrist_angle, end_effector_pos = calculate_ik(
            target_pos, height_offset=height_offset
        )
        
        # 计算正向运动学以获取关节位置
        joint_positions = forward_kinematics(base_angle, shoulder_angle, elbow_angle, wrist_angle)
        
        # 显示关节角度信息
        print(f"\n=== 目标位置: ({target_pos[0]}, {target_pos[1]}, {target_pos[2]}) ===")
        print(f"末端执行器位置: ({end_effector_pos[0]:.1f}, {end_effector_pos[1]:.1f}, {end_effector_pos[2]:.1f})")
        print(f"目标到末端执行器距离: {np.linalg.norm(np.array(target_pos) - end_effector_pos):.1f} 厘米")
        print(f"关节角度（度）:")
        print(f"  底座: {np.degrees(base_angle):.1f}°")
        print(f"  肩部: {np.degrees(shoulder_angle):.1f}°")
        print(f"  肘部: {np.degrees(elbow_angle):.1f}°")
        print(f"  腕部: {np.degrees(wrist_angle):.1f}°")
        
        # 可视化机械臂
        title = f"台灯姿态 @ {target_pos}"
        plot_robotic_arm(joint_positions, target_pos=target_pos, 
                        end_effector_pos=end_effector_pos, title=title)
        
    except ValueError as e:
        print(f"错误: {e}")

# 示例用法
if __name__ == "__main__":
    # 测试不同目标位置，所有目标点z=0（桌面高度）
    test_positions = [
        (40, 0, 0),      # 正前方的书本
        (0, 40, 0),      # 右侧的书本
        (-30, 0, 0),     # 左前方的书本
        (30, 30, 0),     # 右前方的书本
        (60, 0, 0),      # 较远的书本
        (70, 70, 0),     # 远角落的书本
        (-50, -50, 0),   # 远左后方的书本
        (80, -20, 0),    # 远右后方的书本
        (20, -60, 0),    # 远左前方的书本
        (100, 10, 0)     # 极远处的书本
    ]
    
    for pos in test_positions:
        try:
            demo_ik_with_visualization(pos, height_offset=25)
        except Exception as e:
            print(f"处理目标 {pos} 时出错: {e}")
