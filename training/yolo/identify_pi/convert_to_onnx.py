#!/usr/bin/env python
"""
转换PyTorch模型到ONNX格式，以提高树莓派上的推理性能
"""

import torch
import sys
import os
from ultralytics import YOLO
import numpy as np

def convert_pt_to_onnx(pt_model_path, output_path=None, input_size=(320, 320),
                       opset_version=12, simplify=True, dynamic=False):
    """
    将YOLOv8 PyTorch模型转换为ONNX格式
    
    参数:
        pt_model_path: PyTorch模型路径
        output_path: 输出ONNX模型路径
        input_size: 输入尺寸(宽, 高)，默认为320x320 - 为树莓派性能优化
        opset_version: ONNX操作集版本
        simplify: 是否简化ONNX模型
        dynamic: 是否使用动态输入尺寸
    """
    if not os.path.exists(pt_model_path):  # 检查模型文件是否存在
        print(f"错误: 找不到模型文件 {pt_model_path}")  # 如果模型文件不存在，打印错误信息
        return False  # 返回False表示转换失败
        
    print(f"正在加载PyTorch模型: {pt_model_path}")  # 打印加载模型的路径信息
    
    # 如果未指定输出路径，则在输入模型旁创建
    if output_path is None:  # 如果没有提供输出路径
        base_path = os.path.splitext(pt_model_path)[0]  # 去掉输入模型文件的扩展名
        output_path = f"{base_path}_{input_size[0]}x{input_size[1]}.onnx"  # 添加尺寸后缀生成默认输出路径
    
    try:
        # 加载模型
        model = YOLO(pt_model_path)  # 使用ultralytics库加载YOLO模型
        
        # 导出为ONNX格式
        print(f"正在转换为ONNX格式，输入尺寸: {input_size}, 操作集版本: {opset_version}")  # 打印转换信息
        
        success = model.export(  # 调用YOLO模型的导出方法，将模型转换为ONNX格式
            format="onnx",  # 指定导出格式为ONNX
            imgsz=input_size,  # 设置输入图像尺寸
            opset=opset_version,  # 设置ONNX操作集版本
            simplify=simplify,  # 是否简化ONNX模型
            dynamic=dynamic,  # 是否使用动态输入尺寸
            half=False,  # 是否使用FP16精度（这里设置为False以避免兼容性问题）
            device="cpu"  # 指定使用CPU进行导出
        )
        
        if success:  # 如果导出成功
            onnx_output = success  # 获取导出的ONNX模型路径
            # 移动文件（如果导出路径不是我们想要的）
            default_name = f"{os.path.splitext(pt_model_path)[0]}.onnx"  # 默认导出的ONNX文件名
            if os.path.exists(default_name) and default_name != output_path:  # 如果默认文件存在且路径不同
                os.rename(default_name, output_path)  # 重命名文件到指定输出路径
                onnx_output = output_path  # 更新ONNX输出路径
            
            print(f"成功: ONNX模型已保存到 {onnx_output}")  # 打印成功信息
            print(f"优化提示: 此ONNX模型针对 {input_size[0]}x{input_size[1]} 输入尺寸进行了优化")  # 提示优化信息
            return True  # 返回True表示转换成功
        else:
            print("错误: 转换过程失败")  # 如果导出失败，打印错误信息
            return False  # 返回False表示转换失败
            
    except Exception as e:  # 捕获转换过程中可能出现的异常
        print(f"转换过程中出现错误: {e}")  # 打印异常信息
        return False  # 返回False表示转换失败

def quantize_onnx(onnx_model_path, output_path=None):
    """
    对ONNX模型进行量化以减小大小并提高性能
    需要安装onnxruntime-tools和onnx
    
    参数:
        onnx_model_path: 输入ONNX模型路径
        output_path: 输出量化ONNX模型路径
    """
    try:
        import onnx  # 导入onnx库
        from onnxruntime.quantization import quantize_dynamic, QuantType  # 导入量化工具
        
        if not os.path.exists(onnx_model_path):  # 检查ONNX模型文件是否存在
            print(f"错误: 找不到ONNX模型 {onnx_model_path}")  # 如果文件不存在，打印错误信息
            return False  # 返回False表示量化失败
            
        if output_path is None:  # 如果没有提供输出路径
            base_path = os.path.splitext(onnx_model_path)[0]  # 去掉输入模型文件的扩展名
            output_path = f"{base_path}_quant.onnx"  # 添加"_quant"后缀生成默认输出路径
        
        print(f"正在量化模型: {onnx_model_path}")  # 打印量化模型的路径信息
        print(f"输出: {output_path}")  # 打印量化后的输出路径
        
        # 进行动态量化
        quantize_dynamic(
            model_input=onnx_model_path,  # 输入ONNX模型路径
            model_output=output_path,  # 输出量化后的ONNX模型路径
            weight_type=QuantType.QUInt8  # 设置权重类型为8位无符号整数
        )
        
        if os.path.exists(output_path):  # 检查量化后的文件是否存在
            original_size = os.path.getsize(onnx_model_path) / (1024 * 1024)  # 获取原始模型大小（MB）
            quantized_size = os.path.getsize(output_path) / (1024 * 1024)  # 获取量化后模型大小（MB）
            
            print(f"量化成功!")  # 打印量化成功信息
            print(f"原始大小: {original_size:.2f} MB")  # 打印原始模型大小
            print(f"量化后大小: {quantized_size:.2f} MB")  # 打印量化后模型大小
            print(f"减少: {(1 - quantized_size/original_size) * 100:.1f}%")  # 打印大小减少百分比
            return True  # 返回True表示量化成功
        else:
            print("量化过程完成，但找不到输出文件")  # 如果量化后文件不存在，打印错误信息
            return False  # 返回False表示量化失败
            
    except ImportError:  # 捕获导入库失败的异常
        print("错误: 请安装所需库: pip install onnx onnxruntime-tools")  # 提示安装所需库
        return False  # 返回False表示量化失败
    except Exception as e:  # 捕获量化过程中可能出现的异常
        print(f"量化过程中出现错误: {e}")  # 打印异常信息
        return False  # 返回False表示量化失败

def main():
    # 默认模型路径
    default_model_path = "runs/detect/my_custom_model/weights/best.pt"  # 设置默认PyTorch模型路径
    
    if len(sys.argv) > 1:  # 如果命令行参数中提供了模型路径
        model_path = sys.argv[1]  # 从命令行参数获取模型路径
    else:
        model_path = default_model_path  # 否则使用默认模型路径
        print(f"未指定模型路径，将使用默认路径: {default_model_path}")  # 打印提示信息
    
    # 输入尺寸 - 使用较小尺寸以提高树莓派性能
    sizes = [(320, 320), (256, 256)]  # 定义两种输入尺寸
    
    # 转换为不同尺寸
    for size in sizes:  # 遍历每种输入尺寸
        size_str = f"{size[0]}x{size[1]}"  # 将尺寸转换为字符串格式
        print(f"\n正在转换为尺寸 {size_str}...")  # 打印当前转换的尺寸信息
        
        # 确定输出路径
        onnx_path = f"runs/detect/my_custom_model/weights/best_{size_str}.onnx"  # 设置ONNX模型输出路径
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(onnx_path), exist_ok=True)  # 创建输出目录（如果不存在）
        
        # 转换模型
        if convert_pt_to_onnx(model_path, onnx_path, input_size=size):  # 调用转换函数
            # 如果转换成功，进行量化
            print("\n正在进行模型量化...")  # 打印量化提示信息
            quant_path = f"runs/detect/my_custom_model/weights/best_{size_str}_quant.onnx"  # 设置量化模型输出路径
            quantize_onnx(onnx_path, quant_path)  # 调用量化函数
    
    print("\n转换完成!")  # 打印转换完成信息
    print("请使用quantized模型获得最佳性能:")  # 提示使用量化模型
    print("- runs/detect/my_custom_model/weights/best_320x320_quant.onnx (推荐)")  # 推荐模型路径
    print("- runs/detect/my_custom_model/weights/best_256x256_quant.onnx (最快)")  # 最快模型路径
    print("\n在main_optimized_book_aware.py中更新MODEL_PATH以使用新模型")  # 提示更新模型路径

if __name__ == "__main__":
    main()  # 调用主函数
