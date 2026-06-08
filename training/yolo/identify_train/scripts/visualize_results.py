#可视化检测结果
from ultralytics import YOLO#可视化脚本，功能：对验证集中的所有图片执行推理并显示标注结果。用空格换
import cv2
import os

def visualize_detections():
    model = YOLO("runs/detect/my_custom_model/weights/best.pt")
    dataset_dir = "datasets/my_dataset/images/val"  # 验证集图片路径

    # 遍历验证集图片并显示结果
    for img_name in os.listdir(dataset_dir): 
        img_path = os.path.join(dataset_dir, img_name) # 图片路径
        image = cv2.imread(img_path) 
        results = model.predict(image) # 执行推理
        annotated_image = results[0].plot()
        cv2.imshow("Validation Results", annotated_image)
        if cv2.waitKey(0) == ord('q'):  # 按Q键退出
            break

if __name__ == "__main__":
    visualize_detections()
