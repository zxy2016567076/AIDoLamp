#推理脚本
from ultralytics import YOLO
import cv2

def main():
    # 加载模型（添加路径检查）
    model_path = "runs/detect/my_custom_model/weights/best.pt"
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"模型加载失败！请检查路径：{model_path}\n错误信息：{e}")
        return

    # 加载测试图片（添加文件存在性检查）
    image_path = "test_image1.jpg"
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError
    except:
        print(f"图片加载失败！请检查路径：{image_path}")
        return

    # 执行推理（调低置信度并可视化原始图片）
    cv2.imshow("Original Image", image)
    results = model.predict(image, conf=0.1)  # 调低置信度阈值
    
    # 打印原始检测结果（调试用）
    print("原始检测结果：", results[0].boxes.data.tolist())

    # 可视化检测框
    if len(results[0].boxes) > 0: # 检测到目标时显示标注结果
        annotated_image = results[0].plot() # 绘制标注框
        cv2.imshow("Detection Result", annotated_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("未检测到任何目标！尝试：1. 调低conf阈值 2. 检查模型训练是否有效")

if __name__ == "__main__":
    main()
