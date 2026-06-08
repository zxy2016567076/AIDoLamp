"""
测试脚本：检查特征提取和模型训练流程
"""
import os
import sys
import time
import logging
import argparse

def check_directories():
    """检查必要的目录结构"""
    print("检查目录结构...")
    
    from config import RAW_DATA_DIR, PROCESSED_DATA_DIR
    
    # 检查原始数据目录
    if not os.path.exists(RAW_DATA_DIR):
        print(f"错误: 原始数据目录 {RAW_DATA_DIR} 不存在")
        return False
    
    # 检查原始数据子目录（手势类别目录）
    from config import CLASS_MAPPING
    
    empty_dirs = []
    valid_dirs = []
    
    for class_id, class_name in CLASS_MAPPING.items():
        # 检查标准格式目录
        dir_path = os.path.join(RAW_DATA_DIR, f"{class_id}.{class_name}")
        simple_dir_path = os.path.join(RAW_DATA_DIR, str(class_id))
        
        if os.path.exists(dir_path):
            images = [f for f in os.listdir(dir_path) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if images:
                print(f"  √ 类别 {class_id}.{class_name}: {len(images)} 个图像")
                valid_dirs.append(class_id)
            else:
                print(f"  × 类别 {class_id}.{class_name}: 目录存在但没有图像")
                empty_dirs.append(class_id)
        elif os.path.exists(simple_dir_path):
            images = [f for f in os.listdir(simple_dir_path) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if images:
                print(f"  √ 类别 {class_id}: {len(images)} 个图像")
                valid_dirs.append(class_id)
            else:
                print(f"  × 类别 {class_id}: 目录存在但没有图像")
                empty_dirs.append(class_id)
        else:
            print(f"  × 类别 {class_id}.{class_name}: 目录不存在")
            empty_dirs.append(class_id)
    
    if not valid_dirs:
        print("错误: 没有找到任何有效的手势类别目录")
        print("提示: 请确保每个手势类别有单独的目录，格式为 '类别ID.类别名称'")
        return False
    
    print(f"找到 {len(valid_dirs)} 个有效类别目录，{len(empty_dirs)} 个空目录或不存在的目录")
    return True

def check_feature_extraction():
    """测试特征提取过程"""
    print("\n测试特征提取过程...")
    
    from config import FEATURE_CSV_PATH
    
    # 检查特征CSV是否存在
    if os.path.exists(FEATURE_CSV_PATH):
        print(f"特征CSV文件已存在: {FEATURE_CSV_PATH}")
        override = input("是否重新生成特征? (y/n): ").lower() == 'y'
        if not override:
            return True
    
    try:
        from importlib import import_module
        feature_extractor = import_module('1_feature_extractor')
        
        # 执行特征提取
        start_time = time.time()
        print("开始提取特征...")
        sample_count = feature_extractor.generate_feature_csv(augment=True, visualize=True)
        end_time = time.time()
        
        if sample_count > 0:
            print(f"特征提取成功! 共处理 {sample_count} 个样本，耗时 {end_time - start_time:.2f} 秒")
            return True
        else:
            print("特征提取失败: 未找到有效样本")
            return False
    
    except Exception as e:
        print(f"特征提取过程出错: {str(e)}")
        return False

def check_model_training():
    """测试模型训练过程"""
    print("\n测试模型训练过程...")
    
    from config import MODEL_PATH
    
    # 检查模型文件是否存在
    if os.path.exists(MODEL_PATH):
        print(f"模型文件已存在: {MODEL_PATH}")
        override = input("是否重新训练模型? (y/n): ").lower() == 'y'
        if not override:
            return True
    
    try:
        from importlib import import_module
        model_trainer = import_module('2_train_model')
        
        # 执行模型训练
        start_time = time.time()
        print("开始训练模型...")
        model, accuracy = model_trainer.train_gesture_model(compare=False)  # 不比较多个模型，加快速度
        end_time = time.time()
        
        print(f"模型训练成功! 准确率: {accuracy:.4f}，耗时 {end_time - start_time:.2f} 秒")
        return True
    
    except Exception as e:
        print(f"模型训练过程出错: {str(e)}")
        return False

def check_main_program():
    """测试主程序功能"""
    print("\n测试主程序功能...")
    
    # 检查模型文件是否存在
    from config import MODEL_PATH
    
    if not os.path.exists(MODEL_PATH):
        print(f"错误: 模型文件不存在 {MODEL_PATH}")
        print("请先完成特征提取和模型训练")
        return False
    
    print("模型文件检查通过")
    print("准备启动摄像头测试...")
    print("摄像头测试会启动实时手势识别。按ESC键退出测试。")
    
    proceed = input("是否继续? (y/n): ").lower() == 'y'
    if not proceed:
        return True
    
    try:
        from importlib import import_module
        main_program = import_module('main')
        
        # 创建识别器对象并启动识别
        recognizer = main_program.GestureRecognizer()
        recognizer.real_time_recognition()
        
        return True
    
    except Exception as e:
        print(f"主程序功能测试出错: {str(e)}")
        return False

def main():
    """主测试流程"""
    parser = argparse.ArgumentParser(description='手势识别系统测试工具')
    parser.add_argument('--skip-dirs', action='store_true', help='跳过目录检查')
    parser.add_argument('--skip-features', action='store_true', help='跳过特征提取测试')
    parser.add_argument('--skip-model', action='store_true', help='跳过模型训练测试')
    parser.add_argument('--skip-main', action='store_true', help='跳过主程序测试')
    
    args = parser.parse_args()
    
    print("=== 手势识别系统测试 ===\n")
    
    # 设置全局抑制日志输出（避免测试期间日志干扰）
    logging.basicConfig(level=logging.ERROR)
    
    # 1. 检查目录结构
    if not args.skip_dirs:
        dirs_ok = check_directories()
        if not dirs_ok:
            print("\n目录检查失败。请确保正确设置手势图像目录。")
            return 1
    
    # 2. 测试特征提取
    if not args.skip_features:
        features_ok = check_feature_extraction()
        if not features_ok:
            print("\n特征提取失败。请检查错误信息并修复问题。")
            return 1
    
    # 3. 测试模型训练
    if not args.skip_model:
        model_ok = check_model_training()
        if not model_ok:
            print("\n模型训练失败。请检查错误信息并修复问题。")
            return 1
    
    # 4. 测试主程序功能
    if not args.skip_main:
        main_ok = check_main_program()
        if not main_ok:
            print("\n主程序功能测试失败。请检查错误信息并修复问题。")
            return 1
    
    print("\n=== 测试完成! 所有测试通过 ===")
    return 0

if __name__ == '__main__':
    sys.exit(main())
