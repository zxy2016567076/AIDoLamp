# preprocessor.py
import os
import cv2
import csv
import joblib
import numpy as np
from tqdm import tqdm
from mediapipe import solutions as mp
from sklearn.preprocessing import StandardScaler

# 严格定义的标签映射（与训练配置保持一致）
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


class UnifiedGesturePreprocessor:
    def __init__(self):
        """初始化MediaPipe组件和特征配置"""
        self.mp_hands = mp.hands
        self.hands = self._init_mediapipe()
        self.feature_columns = self._generate_feature_columns()

    def _init_mediapipe(self):
        """安全初始化手部检测模型"""
        return self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.3
        )

    def _generate_feature_columns(self):
        """生成CSV文件的特征列名称"""
        base_columns = [f"lm{i}_x" for i in range(21)] + [f"lm{i}_y" for i in range(21)]
        geometry_columns = [
            'palm_width',
            'palm_height',
            'thumb_avg_angle',
            'index_avg_angle',
            'middle_avg_angle',
            'ring_avg_angle',
            'pinky_avg_angle'
        ]
        return base_columns + geometry_columns

    def process_dataset(self, input_dir, output_path):
        """
        主处理方法：
        1. 验证数据目录结构
        2. 按固定顺序处理标签类别
        3. 提取并保存特征数据
        """
        # 验证输入目录
        required_labels = set(LABEL_MAPPING.values())
        existing_labels = set(os.listdir(input_dir))
        missing_labels = required_labels - existing_labels
        if missing_labels:
            raise ValueError(f"Missing required directories: {missing_labels}")

        # 按LABEL_MAPPING顺序处理数据
        reverse_mapping = {v: k for k, v in LABEL_MAPPING.items()}
        all_features = []
        all_labels = []

        for label_name in LABEL_MAPPING.values():
            class_dir = os.path.join(input_dir, label_name)
            print(f"\n🔍 正在处理目录: {class_dir}")
            if not os.path.exists(class_dir):
                continue

            # 处理单个类别
            features, labels = self.process_class_directory(class_dir, reverse_mapping[label_name])
            all_features.extend(features)
            all_labels.extend(labels)

        # 保存处理后的数据
        self._save_features_csv(all_features, all_labels, output_path)
        self._fit_and_save_scaler(all_features)

    def process_class_directory(self, class_dir, label_code):
        """处理单个手势类别目录"""
        features = []
        labels = []
        image_files = self._get_valid_image_files(class_dir)

        for img_file in tqdm(image_files, desc=f"Processing {os.path.basename(class_dir)}", unit="img"):
            img_path = os.path.join(class_dir, img_file)
            feature = self.extract_features(img_path)
            if feature is not None:
                features.append(feature)
                labels.append(label_code)
        print(f"✅ 成功提取 {len(features)} 个特征")
        return np.array(features), labels

    def _get_valid_image_files(self, directory):
        """获取目录中有效的图像文件"""
        valid_exts = ('.png', '.jpg', '.jpeg')
        return [f for f in os.listdir(directory)
                if os.path.splitext(f)[1].lower() in valid_exts]

    def extract_features(self, image_path):
        """从单张图像中提取特征"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return None

            # 使用MediaPipe检测关键点
            results = self.hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if not results.multi_hand_landmarks:
                print(f"未检测到手部关键点：{image_path}")
                return None

            landmarks = results.multi_hand_landmarks[0].landmark
            return self._compute_features(landmarks)
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            return None

    def _compute_features(self, landmarks):
        """根据关键点计算特征向量"""
        # 基础坐标特征
        base_features = [lm.x for lm in landmarks] + [lm.y for lm in landmarks]
        
        # 几何特征计算
        wrist = landmarks[0]
        middle_base = landmarks[9]
        palm_width = abs(wrist.x - middle_base.x)
        palm_height = abs(wrist.y - middle_base.y)
        angles = self._calculate_finger_angles(landmarks)
        
        return base_features + [palm_width, palm_height] + list(angles.values())

    def _calculate_finger_angles(self, landmarks):
        """计算各手指的平均弯曲角度"""
        finger_joints = {
            'thumb': [1, 2, 3, 4],
            'index': [5, 6, 7, 8],
            'middle': [9, 10, 11, 12],
            'ring': [13, 14, 15, 16],
            'pinky': [17, 18, 19, 20]
        }
        
        angles = {}
        for finger, indices in finger_joints.items():
            total_angle = 0.0
            valid_segments = 0
            
            for i in range(len(indices)-2):
                a, b, c = landmarks[indices[i]], landmarks[indices[i+1]], landmarks[indices[i+2]]
                angle = self._vector_angle(a, b, c)
                if not np.isnan(angle):
                    total_angle += angle
                    valid_segments += 1
            
            avg_angle = total_angle / valid_segments if valid_segments > 0 else 0.0
            angles[f"{finger}_avg_angle"] = avg_angle
        
        return angles

    def _vector_angle(self, a, b, c):
        """计算三点形成的向量夹角"""
        vec_ba = np.array([a.x - b.x, a.y - b.y])
        vec_bc = np.array([c.x - b.x, c.y - b.y])
        
        dot = np.dot(vec_ba, vec_bc)
        norm = np.linalg.norm(vec_ba) * np.linalg.norm(vec_bc)
        return np.degrees(np.arccos(np.clip(dot / (norm + 1e-6), -1.0, 1.0)))

    def _save_features_csv(self, features, labels, output_path):
        """保存特征到CSV文件"""
        if len(features) == 0:
            print("❌ 错误：没有提取到任何特征，请检查输入图像或MediaPipe配置")
            return
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(self.feature_columns + ['label'])
            
            for feat, label in zip(features, labels):
                writer.writerow(list(feat) + [label])
                
            print(f"✅ 已保存特征到 {output_path}")

    def _fit_and_save_scaler(self, features):
        """训练并保存标准化器"""
        scaler = StandardScaler()
        scaler.fit(features)
        os.makedirs("models", exist_ok=True)
        joblib.dump(scaler, "models/scaler.pkl")
        print(f"✅ 标准化器已保存到 models/scaler.pkl")

    def __del__(self):
        """资源清理"""
        if hasattr(self, 'hands'):
            self.hands.close()
            
    def _calculate_geometry_features(self, landmarks):
        # 保持与特征生成逻辑严格一致
        angles = self._calculate_finger_angles(landmarks)
        return [
            abs(landmarks[0].x - landmarks[9].x), # palm_width
            abs(landmarks[0].y - landmarks[9].y), # palm_height
            angles['thumb_avg_angle'],
            angles['index_avg_angle'],
            angles['middle_avg_angle'],
            angles['ring_avg_angle'],
            angles['pinky_avg_angle']
        ]


if __name__ == "__main__":
    # 示例用法（处理整个数据集）
    processor = UnifiedGesturePreprocessor()
    processor.process_dataset(
        input_dir="data/raw",
        output_path="data/processed/features.csv"
    )
