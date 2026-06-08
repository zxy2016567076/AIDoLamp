"""
模型训练模块：训练和评估手势识别模型
"""
import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_recall_fscore_support, ConfusionMatrixDisplay
)
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from sklearn.pipeline import Pipeline

# 导入配置和日志
from config import (
    CLASS_MAPPING, FEATURE_CSV_PATH, MODEL_PATH, 
    PROCESSED_DATA_DIR, TEST_SIZE, RANDOM_STATE
)
from logger import setup_logger

# 设置日志记录器
logger = setup_logger("model_trainer", "model_trainer")

def load_and_preprocess_data():
    """
    加载特征数据并进行预处理
    
    返回:
        X_train, X_test, y_train, y_test: 训练和测试数据集
        class_weights: 类别权重（用于处理不平衡数据）
    """
    logger.info(f"加载数据: {FEATURE_CSV_PATH}")
    
    try:
        df = pd.read_csv(FEATURE_CSV_PATH)
        logger.info(f"数据集大小: {df.shape}")
    except Exception as e:
        logger.error(f"读取CSV文件失败: {str(e)}")
        raise
    
    # 检查是否有缺失值
    if df.isnull().any().any():
        logger.warning(f"数据集中存在 {df.isnull().sum().sum()} 个缺失值，将被删除")
        df = df.dropna()
        logger.info(f"清理后数据集大小: {df.shape}")
    
    # 检查数据集是否为空或只有标题行
    if df.shape[0] == 0:
        logger.error("数据集为空，请先运行特征提取器收集样本")
        raise ValueError("数据集为空，无法训练模型")
    
    # 获取特征和标签
    X = df.iloc[:, :-1].values
    
    # 确保标签是整数类型
    try:
        # 首先转换为整数，处理可能的浮点数值
        y = df['label'].astype(int).values
        logger.info(f"标签类型: {y.dtype}, 标签取值范围: {np.min(y)}-{np.max(y)}")
    except Exception as e:
        logger.error(f"标签转换为整数类型失败: {str(e)}")
        # 输出标签的数据类型和唯一值，帮助诊断
        logger.error(f"标签原始类型: {df['label'].dtype}")
        logger.error(f"标签唯一值: {df['label'].unique()}")
        raise
    
    # 分析类别分布
    class_counts = df['label'].value_counts().sort_index()
    logger.info("类别分布:")
    for class_id, count in class_counts.items():
        class_name = CLASS_MAPPING.get(int(float(class_id)), "未知")
        logger.info(f"  类别 {class_id} ({class_name}): {count} 样本")
    
    # 计算类别权重处理不平衡数据
    unique_classes = np.unique(y)
    logger.info(f"唯一类别值: {unique_classes}")
    
    # 计算类别权重，处理不平衡数据
    try:
        class_weights = class_weight.compute_class_weight(
            'balanced', classes=unique_classes, y=y
        )
        class_weight_dict = {c: w for c, w in zip(unique_classes, class_weights)}
    except Exception as e:
        logger.error(f"计算类别权重失败: {str(e)}")
        # 使用均等权重作为备选
        class_weight_dict = {c: 1.0 for c in unique_classes}
    
    # 拆分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    # 特征标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # 保存归一化参数
    joblib.dump(scaler, os.path.join(PROCESSED_DATA_DIR, 'feature_scaler.pkl'))
    
    logger.info(f"训练集大小: {X_train.shape[0]}，测试集大小: {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test, class_weight_dict

def train_and_optimize_model(X_train, y_train, model_type='rf', class_weights=None):
    """
    使用网格搜索优化超参数训练指定类型的模型
    
    参数:
        X_train: 训练特征
        y_train: 训练标签
        model_type: 模型类型 ('rf'=RandomForest, 'svm'=SVM, 'knn'=KNN, 'mlp'=神经网络)
        class_weights: 类别权重字典
        
    返回:
        最佳模型
    """
    logger.info(f"开始训练和优化 {model_type} 模型...")
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    if model_type == 'rf':
        model = RandomForestClassifier(random_state=RANDOM_STATE, class_weight=class_weights)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        model_name = "随机森林"
        
    elif model_type == 'svm':
        model = SVC(random_state=RANDOM_STATE, class_weight=class_weights, probability=True)
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.01, 0.1],
            'kernel': ['rbf', 'poly', 'sigmoid']
        }
        model_name = "支持向量机"
        
    elif model_type == 'knn':
        model = KNeighborsClassifier()
        param_grid = {
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
        }
        model_name = "K近邻"
        
    elif model_type == 'mlp':
        model = MLPClassifier(random_state=RANDOM_STATE, max_iter=1000)
        param_grid = {
            'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
            'activation': ['relu', 'tanh'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'adaptive']
        }
        model_name = "神经网络"
    
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    # 使用网格搜索优化超参数
    grid_search = GridSearchCV(
        model, param_grid, cv=cv, 
        scoring='accuracy', n_jobs=-1, verbose=1
    )
    
    logger.info(f"开始 {model_name} 的网格搜索...")
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    logger.info(f"{model_name} 的最佳参数: {grid_search.best_params_}")
    logger.info(f"{model_name} 的交叉验证分数: {grid_search.best_score_:.4f}")
    
    return best_model, model_name

def compare_models(X_train, X_test, y_train, y_test, class_weights):
    """
    比较多种模型的性能并选择最佳模型
    
    返回:
        最佳模型
    """
    logger.info("开始比较不同模型的性能...")
    
    model_types = ['rf', 'svm', 'knn', 'mlp']
    models = {}
    scores = {}
    
    for model_type in model_types:
        try:
            model, model_name = train_and_optimize_model(
                X_train, y_train, model_type, class_weights
            )
            
            # 在测试集上评估
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            logger.info(f"{model_name} 测试集准确率: {accuracy:.4f}")
            
            models[model_type] = model
            scores[model_type] = accuracy
            
        except Exception as e:
            logger.error(f"训练 {model_type} 模型时出错: {str(e)}")
    
    # 选择最佳模型
    if scores:
        best_model_type = max(scores, key=scores.get)
        best_model = models[best_model_type]
        logger.info(f"最佳模型: {best_model_type} (准确率: {scores[best_model_type]:.4f})")
        return best_model, best_model_type
    else:
        logger.error("所有模型训练失败")
        raise Exception("模型训练失败")

def evaluate_model(model, X_test, y_test):
    """
    全面评估模型性能
    """
    logger.info("开始全面评估模型性能...")
    
    # 预测
    y_pred = model.predict(X_test)
    
    # 计算主要评估指标
    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"准确率: {accuracy:.4f}")
    
    # 计算各类别的精确率、召回率和F1分数
    precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred)
    
    # 输出详细分类报告
    class_report = classification_report(y_test, y_pred, target_names=[CLASS_MAPPING.get(int(i), f"类别{i}") for i in sorted(set(y_test))])
    logger.info(f"分类报告:\n{class_report}")
    
    # 计算并可视化混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    
    # 保存混淆矩阵图
    plt.figure(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, 
        display_labels=[CLASS_MAPPING.get(int(i), f"类别{i}") for i in sorted(set(y_test))]
    )
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title('混淆矩阵')
    plt.tight_layout()
    cm_path = os.path.join(PROCESSED_DATA_DIR, 'confusion_matrix.png')
    plt.savefig(cm_path)
    logger.info(f"混淆矩阵已保存到: {cm_path}")
    
    # 可视化特征重要性（如果是随机森林）
    if hasattr(model, 'feature_importances_'):
        n_features = min(20, len(model.feature_importances_))
        indices = np.argsort(model.feature_importances_)[-n_features:]
        
        plt.figure(figsize=(10, 6))
        plt.barh(range(n_features), model.feature_importances_[indices])
        plt.yticks(range(n_features), [f'特征 {i}' for i in indices])
        plt.xlabel('特征重要性')
        plt.title('前20个最重要特征')
        feat_imp_path = os.path.join(PROCESSED_DATA_DIR, 'feature_importance.png')
        plt.tight_layout()
        plt.savefig(feat_imp_path)
        logger.info(f"特征重要性图已保存到: {feat_imp_path}")
    
    return accuracy, precision, recall, f1, support

def train_gesture_model(compare=True):
    """
    主要训练流程
    
    参数:
        compare: 是否比较多个模型
    """
    # 创建必要的目录
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    try:
        # 加载和预处理数据
        X_train, X_test, y_train, y_test, class_weights = load_and_preprocess_data()
        
        if compare:
            # 比较不同模型
            best_model, best_model_type = compare_models(
                X_train, X_test, y_train, y_test, class_weights
            )
        else:
            # 只训练随机森林
            best_model, best_model_type = train_and_optimize_model(
                X_train, y_train, 'rf', class_weights
            )
        
        # 评估最佳模型
        accuracy, precision, recall, f1, support = evaluate_model(best_model, X_test, y_test)
        
        # 保存最佳模型
        joblib.dump(best_model, MODEL_PATH)
        logger.info(f"最佳模型 ({best_model_type}) 已保存到 {MODEL_PATH}")
        
        # 保存模型元数据
        model_meta = {
            'model_type': best_model_type,
            'accuracy': accuracy,
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'f1': f1.tolist(),
            'support': support.tolist(),
            'classes': sorted(set(y_test))
        }
        
        joblib.dump(model_meta, os.path.join(PROCESSED_DATA_DIR, 'model_metadata.pkl'))
        
        return best_model, accuracy
        
    except Exception as e:
        logger.error(f"训练过程中出错: {str(e)}")
        raise

if __name__ == '__main__':
    import argparse
    import time
    
    parser = argparse.ArgumentParser(description='手势识别模型训练工具')
    parser.add_argument('--compare', action='store_true', help='比较多个模型类型')
    parser.add_argument('--quick', action='store_true', help='快速训练模式 (跳过网格搜索)')
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    try:
        _, accuracy = train_gesture_model(compare=args.compare)
        end_time = time.time()
        
        logger.info(f"训练完成! 测试集准确率: {accuracy:.4f}")
        logger.info(f"总共耗时: {end_time - start_time:.2f} 秒")
        print(f"训练完成! 测试集准确率: {accuracy:.4f}")
        print(f"模型已保存到: {MODEL_PATH}")
        print(f"总共耗时: {end_time - start_time:.2f} 秒")
        
    except Exception as e:
        print(f"训练失败: {str(e)}")
