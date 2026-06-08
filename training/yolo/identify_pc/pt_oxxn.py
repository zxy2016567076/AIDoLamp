# export_model.py
import argparse
from pathlib import Path
from ultralytics import YOLO
import warnings

def main():
    # 命令行参数解析
    parser = argparse.ArgumentParser(description='YOLOv8模型导出工具')
    parser.add_argument('--model-path', type=str, 
                        default='runs/detect/my_custom_model/weights/best.pt',
                        help='预训练模型路径（默认：runs/detect/my_custom_model/weights/best.pt）')
    parser.add_argument('--output-size', type=int, default=320,
                        help='导出模型输入尺寸（建议缩减到320或160，默认：320）')
    parser.add_argument('--dynamic', action='store_true',
                        help='启用动态输入维度（适用于可变分辨率场景）')
    args = parser.parse_args()

    try:
        # 模型路径校验
        model_path = Path(args.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"模型文件 {model_path} 不存在")
            
        # 加载预训练模型
        model = YOLO(str(model_path))
        
        # 执行模型导出
        print(f"⏳ 开始导出模型到ONNX格式，输入尺寸：{args.output_size}...")
        model.export(
            format='onnx',
            imgsz=args.output_size,
            dynamic=args.dynamic, 
            opset=12,  # 使用兼容性较好的opset版本
            simplify=True,  # 自动优化计算图
            device='cpu',  # 确保导出过程不使用GPU
            nms=True  # 确保导出时包含NMS（如果Ultralytics支持）
        )
        
        # 获取导出后的路径
        output_path = model_path.with_suffix('.onnx')
        print(f"✅ 导出成功！保存路径：{output_path}")
        
    except FileNotFoundError as e:
        print(f"❌ 错误：{str(e)}")
        print("请检查以下可能原因：")
        print("1. 确认训练完成后已生成.pt模型文件")
        print("2. 若使用自定义训练目录，请通过--model-path指定正确路径")
    except Exception as e:
        print(f"❌ 导出过程中出现未预期错误：{str(e)}")
        warnings.warn("导出失败，建议检查：1) 磁盘空间 2) ultralytics版本 3) ONNX安装状态")

if __name__ == "__main__":
    main()
