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
        
        # # 数据增强
        # if augment:
        #     image = apply_augmentation(image)
        
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
                        
                except Exception as e: # 处理图像时出错
                    logger.error(f"处理图像时出错 {img_path}: {str(e)}")
                    continue
                
            total_processed += processed_count # 积累处理的图像数量
            logger.info(f"类别 {class_id} 成功处理 {processed_count} 图像")
    
    if all_empty_dirs: # 如果有空目录
        logger.warning(f"警告: 以下类别目录为空或不存在: {', '.join(all_empty_dirs)}")
    
    logger.info(f"特征提取完成。共处理 {len(all_features)} 个样本，已保存到 {FEATURE_CSV_PATH}")
    return len(all_features)

if __name__ == '__main__':
    import argparse
    import time
    
    parser = argparse.ArgumentParser(description='手势识别特征提取工具')
    parser.add_argument('--augment', action='store_true', help='使用数据增强')
    parser.add_argument('--visualize', action='store_true', help='生成可视化')
    parser.add_argument('--collect', type=int, help='收集指定类别ID的样本')
    parser.add_argument('--samples', type=int, default=20, help='收集的样本数量')
    
    args = parser.parse_args()
    start_time = time.time() # 记时开始
    sample_count = generate_feature_csv(augment=args.augment, visualize=args.visualize)# 生成特征csv文件
    end_time = time.time() # 记时结束
        
    if sample_count > 0: # 如果有有效样本
        print(f"特征提取完成! 处理了 {sample_count} 个样本，耗时 {end_time - start_time:.2f} 秒")
        print(f"特征已保存到: {FEATURE_CSV_PATH}")
    else:
        print("特征提取失败: 未找到有效样本")
