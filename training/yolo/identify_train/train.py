#训练脚本
from ultralytics import YOLO#训练脚本

def main():
    # 加载预训练模型（YOLOv8n-small）
    model = YOLO("yolov8n.pt")  # 自动下载模型（若本地不存在）

    # 训练配置
    model.train(
        data="configs/dataset.yaml",   # 指定数据集配置文件
        epochs=100,                    # 训练轮次（建议50-200）
        batch=16,                      # 批大小（显存不足时减小）
        imgsz=640,                     # 输入图像尺寸
        device="0",                    # 使用GPU（如有多个GPU设为"0,1,2,3"）
        name="my_custom_model",        # 结果保存目录名称
        exist_ok=True                  # 允许覆盖同名目录
    )

if __name__ == "__main__":
        main()
