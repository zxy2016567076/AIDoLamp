import os  # 导入操作系统模块
import pandas as pd  # 导入Pandas模块
import joblib  # 导入Joblib模块
from sklearn.model_selection import train_test_split  # 导入训练测试集划分函数
from sklearn.ensemble import RandomForestClassifier  # 导入随机森林分类器
from sklearn.metrics import accuracy_score  # 导入准确率评估函数

def train_gesture_model():
    csv_path = os.path.join('data', 'processed', 'features.csv')  # 特征CSV文件路径
    df = pd.read_csv(csv_path)  # 读取CSV文件为DataFrame
    
    X = df.iloc[:, :-1].values  # 提取特征列
    y = df['label'].values  # 提取标签列
    
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42  # 以80/20比例划分训练集和测试集
    )
    
    # 训练随机森林分类器
    model = RandomForestClassifier(n_estimators=100, random_state=42)  # 初始化随机森林模型
    model.fit(X_train, y_train)  # 训练模型
    
    # 评估模型
    y_pred = model.predict(X_test)  # 预测测试集
    accuracy = accuracy_score(y_test, y_pred)  # 计算准确率
    print(f"测试集准确率: {accuracy:.2f}")  # 打印准确率
    
    # 保存模型
    model_path = os.path.join('data', 'processed', 'gesture_model.pkl')  # 模型保存路径
    joblib.dump(model, model_path)  # 保存模型
    print(f"模型已保存到 {model_path}")  # 打印保存信息

if __name__ == '__main__':
    train_gesture_model()  # 训练手势模型
