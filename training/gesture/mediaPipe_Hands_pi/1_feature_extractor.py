import os  # 导入操作系统模块
import csv  # 导入CSV处理模块
import cv2  # 导入OpenCV模块
import numpy as np  # 导入NumPy模块
import mediapipe as mp  # 导入MediaPipe模块

def extract_hand_keypoints(image_path):
    """使用MediaPipe提取手部关键点坐标"""
    mp_hands = mp.solutions.hands  # 导入手部模型
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)  # 初始化手部检测模型
    
    image = cv2.imread(image_path)  # 读取图像
    if image is None:  # 如果图像为空
        return None  # 返回None
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换图像为RGB格式
    results = hands.process(image_rgb)  # 处理图像，提取手部关键点
    hands.close()  # 关闭模型
    
    if not results.multi_hand_landmarks:  # 如果没有检测到手部关键点
        return None  # 返回None
    
    # 提取所有21个关键点的三维坐标（63维）
    keypoints = []  # 初始化关键点列表
    hand_landmarks = results.multi_hand_landmarks[0]  # 获取第一个手部关键点
    for lm in hand_landmarks.landmark:  # 遍历所有关键点
        keypoints.extend([lm.x, lm.y, lm.z])  # 添加关键点的x, y, z坐标
    
    return np.array(keypoints)  # 返回关键点数组

def generate_feature_csv():
    CLASS_MAPPING = {
        1: 'activate', 2: 'down', 3: 'exit', 4: 'left',
        5: 'mode1', 6: 'mode2', 7: 'mode3', 8: 'mode4',
        9: 'right', 10: 'up'
    }
    
    csv_path = os.path.join('data', 'processed', 'features.csv')  # 保存特征的csv文件路径
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)  # 创建保存目录
    
    with open(csv_path, 'w', newline='') as f:  # 打开CSV文件以写入模式
        writer = csv.writer(f)  # 创建CSV写入器
        # 写入标题行（特征列名+label）
        header = [f'kp{i//3}_{["x","y","z"][i%3]}' for i in range(63)] + ['label']  # 生成标题行
        writer.writerow(header)  # 写入标题行
        
        for class_id in CLASS_MAPPING:  # 遍历所有类别
            dir_name = f"{class_id}.{CLASS_MAPPING[class_id]}"  # 生成类别目录名
            class_dir = os.path.join('data', 'raw', dir_name)  # 生成类别目录路径
            
            if not os.path.exists(class_dir):  # 如果类别目录不存在
                print(f"警告: 跳过缺失目录 {class_dir}")  # 打印警告信息
                continue  # 跳过该类别
                
            for img_file in os.listdir(class_dir):  # 遍历类别目录中的所有文件
                if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):  # 如果文件不是图像文件
                    continue  # 跳过该文件
                    
                img_path = os.path.join(class_dir, img_file)  # 生成图像文件路径
                keypoints = extract_hand_keypoints(img_path)  # 提取手部关键点
                
                if keypoints is not None and len(keypoints) == 63:  # 如果关键点有效且长度为63
                    row = list(keypoints) + [class_id]  # 生成CSV行
                    writer.writerow(row)  # 写入CSV行
                else:
                    print(f"跳过无效文件: {img_path}")  # 打印跳过信息

    print(f"特征已保存到 {csv_path}")  # 打印保存信息

if __name__ == '__main__':
    generate_feature_csv()  # 生成特征CSV文件
