import os
import random
import shutil

def sample_calibration_data(train_image_dir, calib_image_dir, num_samples=200):
    # 获取所有训练图像路径
    all_images = [os.path.join(train_image_dir, f) for f in os.listdir(train_image_dir)]
    # 随机抽样
    selected_images = random.sample(all_images, num_samples)
    # 创建校准数据目录
    os.makedirs(calib_image_dir, exist_ok=True)
    # 复制文件
    for src_path in selected_images:
        dst_path = os.path.join(calib_image_dir, os.path.basename(src_path))
        shutil.copy(src_path, dst_path)

# 示例：从训练集随机抽取 200 张作为校准数据
sample_calibration_data("datasets/my_dataset/images/all", "calibration_images", num_samples=100)