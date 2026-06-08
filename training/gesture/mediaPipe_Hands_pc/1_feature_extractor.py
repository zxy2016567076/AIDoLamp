"""
特征提取模块：从手势图像中提取MediaPipe手部关键点特征
"""
import os
import csv
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm
import random
import logging
import time
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 导入配置和日志
from config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, FEATURE_CSV_PATH,
    HAND_CONFIDENCE_THRESHOLD, CLASS_MAPPING
)
from logger import setup_logger

# 设置日志记录器
logger = setup_logger("feature_extractor", "feature_extractor")

def extract_hand_keypoints(image_path, augment=False, visualize=False):
    """
    使用MediaPipe提取手部关键点坐标
    
    参数:
        image_path: 图像文件路径
        augment: 是否应用数据增强
        visualize: 是否可视化关键点
        
    返回:
        keypoints: 63维的特征向量(21个关键点的x,y,z坐标)或None(检测失败)
        augmented_img: 可视化的图像(如果visualize=True)
    """
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=HAND_CONFIDENCE_THRESHOLD
    )
    
    try:
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"无法读取图像: {image_path}")
            return None, None
        
        # 数据增强
        if augment:
            image = apply_augmentation(image)
        
        # 转换颜色并处理
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        
        # 创建可视化图像
        augmented_img = None
        if visualize:
            augmented_img = image.copy()
        
        if not results.multi_hand_landmarks:
            logger.warning(f"未检测到手部: {image_path}")
            return None, augmented_img
        
        # 提取所有21个关键点的三维坐标（63维）
        keypoints = []
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # 可视化
        if visualize:
            mp_drawing.draw_landmarks(
                augmented_img,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
        
        for lm in hand_landmarks.landmark:
            keypoints.extend([lm.x, lm.y, lm.z])
        
        hands.close()
        return np.array(keypoints), augmented_img
        
    except Exception as e:
        logger.error(f"处理图像时出错 {image_path}: {str(e)}")
        return None, None
    finally:
        hands.close()

def apply_augmentation(image):
    """应用随机数据增强变换"""
    h, w = image.shape[:2]
    
    # 随机选择一种增强方法
    aug_type = random.choice(['rotation', 'scaling', 'translation', 'none'])
    
    if aug_type == 'rotation':
        # 随机旋转 -15 到 15 度
        angle = random.uniform(-15, 15)
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
        image = cv2.warpAffine(image, M, (w, h))
    
    elif aug_type == 'scaling':
        # 随机缩放 0.9 到 1.1 倍
        scale = random.uniform(0.9, 1.1)
        image = cv2.resize(image, None, fx=scale, fy=scale)
        
        # 调整大小以匹配原始尺寸
        if scale > 1:
            # 裁剪中心区域
            center_x, center_y = image.shape[1] // 2, image.shape[0] // 2
            image = image[
                center_y - h//2:center_y + h//2,
                center_x - w//2:center_x + w//2
            ]
        else:
            # 填充以匹配原始尺寸
            new_img = np.zeros((h, w, 3), dtype=np.uint8)
            start_x = (w - image.shape[1]) // 2
            start_y = (h - image.shape[0]) // 2
            new_img[
                start_y:start_y + image.shape[0],
                start_x:start_x + image.shape[1]
            ] = image
            image = new_img
    
    elif aug_type == 'translation':
        # 随机平移 -5% 到 5%
        tx = int(w * random.uniform(-0.05, 0.05))
        ty = int(h * random.uniform(-0.05, 0.05))
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        image = cv2.warpAffine(image, M, (w, h))
    
    return image

def normalize_features(features):
    """归一化特征"""
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(features)
    
    # 保存归一化参数，以便在预测时使用
    np.save(os.path.join(PROCESSED_DATA_DIR, 'scaler_mean.npy'), scaler.mean_)
    np.save(os.path.join(PROCESSED_DATA_DIR, 'scaler_scale.npy'), scaler.scale_)
    
    return normalized_features

def visualize_class_distribution(labels):
    """可视化类别分布"""
    plt.figure(figsize=(12, 6))
    counts = np.bincount(labels)[1:]  # 从索引1开始，跳过0
    plt.bar(range(1, len(counts) + 1), counts)
    plt.xticks(range(1, len(counts) + 1))
    plt.xlabel('Gesture Class')
    plt.ylabel('Number of Samples')
    plt.title('Class Distribution')
    
    # 使用类名作为标签
    class_labels = [f"{i}.{CLASS_MAPPING.get(i, '')}" for i in range(1, len(counts) + 1)]
    plt.xticks(range(1, len(counts) + 1), class_labels, rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PROCESSED_DATA_DIR, 'class_distribution.png'))
    logger.info("类别分布图已保存")

def generate_feature_csv(augment=True, visualize=False):
    """
    生成特征CSV文件
    
    参数:
        augment: 是否使用数据增强
        visualize: 是否生成可视化样本
    """
    logger.info("开始特征提取过程...")
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    all_features = []
    all_labels = []
    all_empty_dirs = []
    visualization_samples = {}
    
    # 初始化MediaPipe
    mp_hands = mp.solutions.hands
    
    # 检查是否至少有一个可用类别目录
    has_valid_dir = False
    for class_id in CLASS_MAPPING:
        dir_name = f"{class_id}.{CLASS_MAPPING[class_id]}"
        class_dir = os.path.join(RAW_DATA_DIR, dir_name)
        if os.path.exists(class_dir):
            has_valid_dir = True
            break
    
    if not has_valid_dir:
        # 尝试搜索任何可能的手势目录
        logger.warning("未找到任何符合命名规则的类别目录，尝试扫描所有子目录...")
        for item in os.listdir(RAW_DATA_DIR):
            item_path = os.path.join(RAW_DATA_DIR, item)
            if os.path.isdir(item_path):
                logger.info(f"发现目录: {item}")
    
    with open(FEATURE_CSV_PATH, 'w', newline='') as f:
        writer = csv.writer(f)
        # 写入标题行（特征列名+label）
        header = [f'kp{i//3}_{["x","y","z"][i%3]}' for i in range(63)] + ['label'] 
        writer.writerow(header)
        
        # 检测模式：尝试两种路径格式
        # 1. 标准格式: {class_id}.{class_name}/
        # 2. 简单格式: {class_id}/
        
        total_processed = 0
        
        # 遍历所有类别目录
        for class_id in tqdm(CLASS_MAPPING, desc="处理手势类别"):
            # 尝试标准格式目录
            dir_name = f"{class_id}.{CLASS_MAPPING[class_id]}"
            class_dir = os.path.join(RAW_DATA_DIR, dir_name)
            
            # 如果标准格式不存在，尝试简单格式
            if not os.path.exists(class_dir):
                class_dir = os.path.join(RAW_DATA_DIR, str(class_id))
                
            if not os.path.exists(class_dir):
                logger.warning(f"目录不存在: {class_id}.{CLASS_MAPPING[class_id]} 或 {class_id}")
                all_empty_dirs.append(dir_name)
                continue
                
            # 获取所有图像文件
            class_images = []
            for ext in ['.jpg', '.jpeg', '.png']: # 支持的图像格式
                class_images.extend(
                    [f for f in os.listdir(class_dir) if f.lower().endswith(ext)]
                )
            
            if not class_images:
                logger.warning(f"目录中没有图像: {class_dir}")
                all_empty_dirs.append(dir_name)
                continue
                
            logger.info(f"发现 {class_id}.{CLASS_MAPPING[class_id]} - {len(class_images)} 图像")
                
            logger.info(f"处理 {class_id}.{CLASS_MAPPING[class_id]} - {len(class_images)} 图像")
            processed_count = 0
            
            # 选择一个随机样本用于可视化
            if visualize and class_images:
                visualization_samples[class_id] = os.path.join(class_dir, random.choice(class_images))
            
            for img_file in tqdm(class_images, desc=f"类别 {class_id}", leave=False):
                if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                    
                img_path = os.path.join(class_dir, img_file)
                
                # 处理原始图像 - 使用调整后的MediaPipe手部检测
                image = cv2.imread(img_path)
                if image is None:
                    logger.warning(f"无法读取图像: {img_path}")
                    continue
                
                # 使用隔离的上下文执行MediaPipe处理，避免句柄泄漏
                with mp_hands.Hands(
                    static_image_mode=True,
                    max_num_hands=1,
                    min_detection_confidence=0.3
                ) as hands:
                    # 转换颜色
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    results = hands.process(image_rgb)
                
                # 如果未检测到手部，尝试使用更低阈值
                if not results.multi_hand_landmarks:
                    logger.debug(f"未检测到手部，尝试降低阈值: {img_path}")
                    
                    # 在新的上下文中使用更低阈值
                    with mp_hands.Hands(
                        static_image_mode=True,
                        max_num_hands=1,
                        min_detection_confidence=0.1  # 非常低的阈值
                    ) as hands_retry:
                        results = hands_retry.process(image_rgb)
                        
                    if not results.multi_hand_landmarks:
                        continue  # 如果仍然无法检测，则跳过
                
                # 提取特征
                try:
                    keypoints = []
                    hand_landmarks = results.multi_hand_landmarks[0]
                    for lm in hand_landmarks.landmark:
                        keypoints.extend([lm.x, lm.y, lm.z])
                    
                    if len(keypoints) == 63:
                        all_features.append(keypoints)
                        all_labels.append(class_id)
                        row = list(keypoints) + [class_id]
                        writer.writerow(row)
                        processed_count += 1
                        
                        # 如果启用数据增强，额外创建增强版本
                        if augment:
                            # 为每个样本创建2个增强版本
                            for _ in range(2):
                                aug_image = apply_augmentation(image)
                                aug_image_rgb = cv2.cvtColor(aug_image, cv2.COLOR_BGR2RGB)
                                
                                # 对增强图像使用独立的MediaPipe实例，避免资源冲突
                                with mp_hands.Hands(
                                    static_image_mode=True,
                                    max_num_hands=1,
                                    min_detection_confidence=0.3
                                ) as hands_aug:
                                    aug_results = hands_aug.process(aug_image_rgb)
                                
                                if aug_results.multi_hand_landmarks:
                                    aug_keypoints = []
                                    aug_hand_landmarks = aug_results.multi_hand_landmarks[0]
                                    for lm in aug_hand_landmarks.landmark:
                                        aug_keypoints.extend([lm.x, lm.y, lm.z])
                                    
                                    if len(aug_keypoints) == 63:
                                        all_features.append(aug_keypoints)
                                        all_labels.append(class_id)
                                        row = list(aug_keypoints) + [class_id]
                                        writer.writerow(row)
                                        processed_count += 1
                except Exception as e:
                    logger.error(f"处理图像时出错 {img_path}: {str(e)}")
                    continue
                
            total_processed += processed_count
            logger.info(f"类别 {class_id} 成功处理 {processed_count} 图像")
    
    if all_empty_dirs:
        logger.warning(f"警告: 以下类别目录为空或不存在: {', '.join(all_empty_dirs)}")
    
    # MediaPipe资源已在with语句中自动关闭
    
    # 特征归一化和可视化
    if all_features:
        logger.info(f"共处理 {len(all_features)} 个样本")
        logger.info("归一化特征...")
        all_features = np.array(all_features)
        normalized_features = normalize_features(all_features)
        
        # 保存归一化后的特征到CSV (可选)
        norm_csv_path = os.path.join(PROCESSED_DATA_DIR, 'normalized_features.csv')
        with open(norm_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for i, features in enumerate(normalized_features):
                writer.writerow(list(features) + [all_labels[i]])
        
        # 可视化类别分布
        logger.info("生成类别分布可视化...")
        visualize_class_distribution(all_labels)
        
        # 生成样本可视化
        if visualize and visualization_samples:
            os.makedirs(os.path.join(PROCESSED_DATA_DIR, 'visualizations'), exist_ok=True)
            logger.info("生成手势可视化样本...")
            
            for class_id, img_path in visualization_samples.items():
                try:
                    _, vis_img = extract_hand_keypoints(img_path, visualize=True)
                    
                    if vis_img is not None:
                        # 添加类别标签
                        cv2.putText(
                            vis_img, 
                            f"{class_id}: {CLASS_MAPPING.get(class_id, '未知')}", 
                            (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            1, 
                            (0, 255, 0), 
                            2
                        )
                        
                        # 保存可视化图像
                        vis_path = os.path.join(PROCESSED_DATA_DIR, 'visualizations', f'class_{class_id}.jpg')
                        cv2.imwrite(vis_path, vis_img)
                        logger.info(f"已保存类别 {class_id} 的可视化图像")
                except Exception as e:
                    logger.error(f"无法为类别 {class_id} 生成可视化图像: {str(e)}")
    
    logger.info(f"特征提取完成。共处理 {len(all_features)} 个样本，已保存到 {FEATURE_CSV_PATH}")
    return len(all_features)

def collect_new_samples(class_id, num_samples=20):
    """
    通过摄像头收集新的手势样本
    
    参数:
        class_id: 要收集的手势类别ID
        num_samples: 要收集的样本数量
    """
    if class_id not in CLASS_MAPPING:
        logger.error(f"无效的类别ID: {class_id}")
        return
        
    # 确保类别目录存在
    class_name = CLASS_MAPPING[class_id]
    class_dir = os.path.join(RAW_DATA_DIR, f"{class_id}.{class_name}")
    os.makedirs(class_dir, exist_ok=True)
    
    logger.info(f"开始收集类别 {class_id}.{class_name} 的样本")
    
    # 设置MediaPipe
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7
    )
    
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("无法打开摄像头")
        return
    
    samples_collected = 0
    delay_between_captures = 0.5  # 拍摄间隔(秒)
    last_capture_time = 0
    display_countdown = 0
    
    try:
        while samples_collected < num_samples:
            ret, frame = cap.read()
            if not ret:
                logger.error("无法读取摄像头帧")
                break
                
            # 镜像显示
            frame = cv2.flip(frame, 1)
            
            # 处理帧
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            
            # 显示帧
            info_text = f"收集 {class_id}.{class_name} 样本: {samples_collected}/{num_samples}"
            cv2.putText(frame, info_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            hand_detected = False
            
            # 如果检测到手，绘制关键点
            if results.multi_hand_landmarks:
                hand_detected = True
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, 
                        hand_landmarks, 
                        mp_hands.HAND_CONNECTIONS
                    )
            
            current_time = time.time()
            
            # 如果检测到手并且已经过了足够的延迟时间
            if hand_detected and current_time - last_capture_time > delay_between_captures:
                # 如果显示倒计时，递减
                if display_countdown > 0:
                    display_countdown -= 1
                    cv2.putText(frame, f"拍摄中... {display_countdown}", (frame.shape[1]//2 - 100, frame.shape[0]//2), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                else:
                    # 拍摄新样本
                    timestamp = int(time.time() * 1000000)
                    filename = f"{class_id}.{class_name}_{timestamp}.jpg"
                    filepath = os.path.join(class_dir, filename)
                    cv2.imwrite(filepath, frame)
                    
                    samples_collected += 1
                    last_capture_time = current_time
                    display_countdown = 3  # 显示3帧的倒计时
                    logger.info(f"已保存样本 {samples_collected}/{num_samples}: {filename}")
            
            # 如果检测到手，显示准备拍摄提示
            elif hand_detected:
                next_capture_in = max(0, delay_between_captures - (current_time - last_capture_time))
                cv2.putText(frame, f"准备拍摄... {next_capture_in:.1f}s", (frame.shape[1]//2 - 150, frame.shape[0]//2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            else:
                cv2.putText(frame, "请将手放入画面中", (frame.shape[1]//2 - 150, frame.shape[0]//2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            cv2.imshow(f'收集类别 {class_id}.{class_name} 样本', frame)
            
            if cv2.waitKey(5) & 0xFF == 27:  # ESC键退出
                break
                
        logger.info(f"已收集 {samples_collected} 个样本到 {class_dir}")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()

if __name__ == '__main__':
    import argparse
    import time
    
    parser = argparse.ArgumentParser(description='手势识别特征提取工具')
    parser.add_argument('--augment', action='store_true', help='使用数据增强')
    parser.add_argument('--visualize', action='store_true', help='生成可视化')
    parser.add_argument('--collect', type=int, help='收集指定类别ID的样本')
    parser.add_argument('--samples', type=int, default=20, help='收集的样本数量')
    
    args = parser.parse_args()
    
    if args.collect is not None:
        if args.collect in CLASS_MAPPING:
            collect_new_samples(args.collect, args.samples)
        else:
            print(f"错误: 无效的类别ID {args.collect}")
            print(f"有效类别: {CLASS_MAPPING}")
    else:
        start_time = time.time()
        sample_count = generate_feature_csv(augment=args.augment, visualize=args.visualize)
        end_time = time.time()
        
        if sample_count > 0:
            print(f"特征提取完成! 处理了 {sample_count} 个样本，耗时 {end_time - start_time:.2f} 秒")
            print(f"特征已保存到: {FEATURE_CSV_PATH}")
        else:
            print("特征提取失败: 未找到有效样本")
