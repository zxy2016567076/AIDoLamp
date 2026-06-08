#划分训练集和验证集
import os
import shutil
from sklearn.model_selection import train_test_split

def split_dataset():
    # 数据源路径 ★直接使用已标注的完整数据集
    src_images = os.path.abspath("datasets/my_dataset/images/all")
    src_labels = os.path.abspath("datasets/my_dataset/labels/all")
    output_dir = os.path.abspath("datasets/my_dataset")

    train_ratio = 0.8

    # 自动创建输出目录
    os.makedirs(os.path.join(output_dir, "images/train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images/val"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels/train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels/val"), exist_ok=True)

    # 获取所有图片文件列表 ★关键修改点：统一小写后缀匹配
    image_files = [
        f for f in os.listdir(src_images)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))  # 转为小写后匹配
    ]

    # 分割数据集
    train_files, val_files = train_test_split(image_files, train_size=train_ratio, shuffle=True)

    # 复制训练集文件
    for img_name in train_files:
        base_name = os.path.splitext(img_name)[0]
        # 图片（保留原始文件名大小写）
        shutil.copy(
            os.path.join(src_images, img_name),
            os.path.join(output_dir, "images/train", img_name)
        )
        # 标签
        shutil.copy(
            os.path.join(src_labels, f"{base_name}.txt"),
            os.path.join(output_dir, "labels/train", f"{base_name}.txt")
        )

    # 复制验证集文件
    for img_name in val_files:
        base_name = os.path.splitext(img_name)[0]
        shutil.copy(
            os.path.join(src_images, img_name),
            os.path.join(output_dir, "images/val", img_name)
        )
        shutil.copy(
            os.path.join(src_labels, f"{base_name}.txt"),
            os.path.join(output_dir, "labels/val", f"{base_name}.txt")
        )

    print(f"处理完成！训练集: {len(train_files)} 张，验证集: {len(val_files)} 张")

if __name__ == "__main__":
    split_dataset()
