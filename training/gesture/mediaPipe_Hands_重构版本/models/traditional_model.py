"""
优化版机器学习模型训练框架
新增：参数校验/交叉验证/特征重要性分析/类平衡处理
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier
import joblib
import yaml
import warnings
import os
warnings.filterwarnings('ignore')

# 恢复正确的顺序映射
LABEL_MAPPING = {
    0: 'activate',
    1: 'up',
    2: 'down',
    3: 'left',
    4: 'right',
    5: 'exit',
    6: 'mode1',
    7: 'mode2',
    8: 'mode3', 
    9: 'mode4'
}


class EnhancedTraditionalModel:
    def __init__(self, config_path="config/advanced_model_config.yaml"):
        """
        参数优化点：
        1. 增加配置参数校验机制
        2. 分离数据处理与模型参数
        3. 添加复合评估指标
        """
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        # 分离不同模块配置
        self.data_cfg = self.config['data_settings']   # 数据相关配置
        self.model_cfg = self.config['model_params']   # 模型参数配置
        self.cv_cfg = self.config['cross_validation']  # 交叉验证配置
        
        # 参数合法性检查（示例）
        assert self.model_cfg['model_type'] in ['random_forest', 'svm', 'xgboost'], \
            "Invalid model type in config"
        if 'svm' in self.model_cfg['model_type']:
            assert self.model_cfg['kernel'] in ['linear', 'rbf', 'poly'], \
                "Unsupported SVM kernel"

    def load_processed_data(self, csv_path):
        """
        改进点：
        - 自动标准化处理
        - 分类标签编码
        - 类不平衡补偿
        """
        df = pd.read_csv(csv_path)
        
        # 特征/标签分离
        X = df.drop(columns=['label']).values  # 转换为numpy数组提升性能
        y_raw = df['label'].values
        
        # 标签编码（确保类别连续性）
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(y_raw)
        
        # 标准化处理（根据配置开关）
        if self.data_cfg['enable_scaling']:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)
        
        # 保存特征维度信息
        self.n_features = X.shape[1]
        self.classes = np.unique(y)
        print(f"Loaded data: {X.shape[0]} samples, {self.n_features} features, {len(self.classes)} classes")
        
        return train_test_split(
            X, y, 
            test_size=self.data_cfg['test_size'],
            stratify=y,  # 保持split后的类分布
            random_state=self.data_cfg['split_seed']
        )

    def build_model_pipeline(self):
        """构建集成数据处理和模型训练的流水线"""
        model_type = self.model_cfg['model_type']
        
        # 根据不同模型类型配置处理流程
        if model_type == "random_forest":
            model = RandomForestClassifier(
                n_estimators=self.model_cfg['n_estimators'],
                max_depth=self.model_cfg.get('max_depth', None),  # 改为灵活设置
                class_weight='balanced',  # 类权重平衡
                n_jobs=-1  # 启用多核并行
            )
            param_grid = {
                'model__max_depth': [None, 10, 20],
                'model__min_samples_split': [2, 5]
            }
            
        elif model_type == "svm":
            model = SVC(
                C=self.model_cfg['C'],
                kernel=self.model_cfg['kernel'],
                probability=True,  # 开启概率预测
                class_weight='balanced'  # 类平衡参数
            )
            param_grid = {
                'model__C': np.logspace(-3, 3, 7),
                'model__gamma': ['scale', 'auto']
            } if self.model_cfg['kernel'] == 'rbf' else {}
            
        elif model_type == "xgboost":
            model = XGBClassifier(
                **self.model_cfg['xgb_params'],
                use_label_encoder=False,
                eval_metric='logloss'
            )
            param_grid = {
                'model__learning_rate': [0.01, 0.1],
                'model__max_depth': [3, 5]
            }
        
        # 构建带标准化的Pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler() if self.data_cfg['enable_scaling'] else None),
            ('model', model)
        ])
        
        # 移除不需要的步骤（如在配置中关闭标准化）
        if not self.data_cfg['enable_scaling']:
            pipeline.steps.pop(0)
            
        return pipeline, param_grid

    def hyperparameter_tuning(self, pipeline, X, y, param_grid):
        """执行网格搜索超参数优化"""
        if not param_grid:  # 无待优化参数时直接返回
            return pipeline
        
        cv_method = StratifiedKFold(
            n_splits=self.cv_cfg['n_splits'],
            shuffle=True,
            random_state=self.cv_cfg['cv_seed']
        )
        
        # 自定义评分标准（平衡准确率）
        scorer = {
            'f1_weighted': 'f1_weighted',
            'accuracy': 'accuracy'
        }
        
        # 并行网格搜索
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring=scorer,
            refit='f1_weighted',
            cv=cv_method,
            n_jobs=-1  # 全CPU并行
        )
        
        grid_search.fit(X, y)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score (F1 weighted): {grid_search.best_score_:.3f}")
        return grid_search.best_estimator_

    def enhanced_evaluation(self, model, X_test, y_test):
        """扩展评估指标"""
        from sklearn.metrics import confusion_matrix, roc_auc_score
        
        preds = model.predict(X_test)
        probas = model.predict_proba(X_test)[:, 1]  # 获取正类概率（适用于二分类）
        
        # 多类AUC计算
        if len(self.classes) > 2:
            auc = roc_auc_score(y_test, probas, multi_class='ovo')
        else:
            auc = roc_auc_score(y_test, probas)
            
        # 打印扩展指标
        print("\n=== Expanded Evaluation ===")
        print(f"Test AUC Score: {auc:.3f}")
        print("\nClassification Report:")
        print(classification_report(y_test, preds, target_names=self.label_encoder.classes_))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, preds))

    def save_artifacts(self, model, model_path, metadata_path):
        """保存完整模型元数据"""
        save_data = {
            'model': model,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'config': self.config,
            'git_commit': os.popen('git rev-parse HEAD').read().strip()  # 记录代码版本
        }
        joblib.dump(save_data, model_path)
        
        # 另外保存参数重要性（针对树模型）
        if hasattr(model.named_steps['model'], 'feature_importances_'):
            feat_imp = model.named_steps['model'].feature_importances_
            pd.Series(feat_imp, index=self.feature_names)\
              .sort_values(ascending=False)\
              .to_csv(metadata_path+'_feature_importance.csv')

if __name__ == "__main__":
    trainer = EnhancedTraditionalModel()
    
    # 数据加载
    try:
        X_train, X_test, y_train, y_test = trainer.load_processed_data(trainer.data_cfg['dataset_path'])
    except FileNotFoundError:
        print(f"Error: Dataset not found at {trainer.data_cfg['dataset_path']}")
        exit(1)
    
    # 模型构建与调优
    pipeline, param_grid = trainer.build_model_pipeline()
    tuned_model = trainer.hyperparameter_tuning(pipeline, X_train, y_train, param_grid)
    
    # 性能评估
    trainer.enhanced_evaluation(tuned_model, X_test, y_test)
    
    # 保存完整信息
    trainer.save_artifacts(
        tuned_model, 
        "models/advanced_model.pkl",
        "models/model_metadata"
    )
