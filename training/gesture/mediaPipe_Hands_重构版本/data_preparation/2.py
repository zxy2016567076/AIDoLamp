# preprocessor.py
import os
import cv2
import pandas as pd
import numpy as np
import joblib
from tqdm import tqdm
from pathlib import Path
from mediapipe import solutions as mp
from sklearn.preprocessing import StandardScaler

class GesturePreprocessor:
    # 与模型训练保持完全一致的标签映射
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

    def __init__(self):
        """初始化并验证关键组件"""
        self._validate_label_mapping()
        self.feature_columns = self._generate_feature_columns()
        self.hands = self._init_mediapipe()

    def _validate_label_mapping(self):
        """验证标签映射唯一性"""
        labels = list(self.LABEL_MAPPING.values())
        if len(labels) != len(set(labels)):
            raise ValueError("发现重复的标签名称")
        if len(self.LABEL_MAPPING) != 10:
            raise ValueError("标签映射必须包含10个类别")

    def _init_mediapipe(self):
        """安全初始化MediaPipe组件"""
        try:
            return mp.hands.Hands(
                static_image_mode=True,
                max_num_hands=1,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5
            )
        except Exception as e:
            raise RuntimeError(f"初始化MediaPipe失败: {str(e)}")

    def _generate_feature_columns(self):
        """生成与特征计算完全匹配的列名"""
        base_features = [f"lm_{i}_x" for i in range(21)] + [f"lm_{i}_y" for i in range(21)]
        geometry_features = [
            'palm_width',
            'palm_height',
            'thumb_avg_deg',
            'index_avg_deg',
            'middle_avg_deg',
            'ring_avg_deg',
            'pinky_avg_deg'
        ]
        return base_features + geometry_features

    def process_dataset(self, input_dir, output_dir):
        """
        主处理流程：
        1. 输入目录结构验证
        2. 数据预处理
        3. 保存结构化数据
        4. 生成标准化器
        """
        input_dir = Path(input_dir)
        output_path = Path(output_dir) / "features.csv"
        scaler_path = Path(output_dir) / "scaler.pkl"

        # 验证输入目录
        self._validate_input_structure(input_dir)

        # 初始化数据容器
        df = pd.DataFrame(columns=self.feature_columns + ['label'])
        features_list = []
        labels_list = []

        # 按规范顺序处理每个类别
        for label_id, label_name in self.LABEL_MAPPING.items():
            class_dir = input_dir / label_name
            if not class_dir.exists():
                print(f"⚠️ 警告: 跳过缺失的类别目录 {label_name}")
                continue

            class_features, class_labels = self.process_class(class_dir, label_id)
            features_list.extend(class_features)
            labels_list.extend(class_labels)

        # 合并数据
        df = pd.DataFrame(features_list, columns=self.feature_columns)
        df['label'] = labels_list

        # 输出处理
        self._save_dataset(df, output_path)
        self._save_scaler(scaler_path, df[self.feature_columns])
        print(f"\n✅ 预处理完成！结果保存至 {output_dir}")

    def _validate_input_structure(self, input_dir):
        """验证原始数据集目录结构"""
        if not input_dir.is_dir():
            raise FileNotFoundError(f"输入目录不存在: {input_dir}")

        required_labels = set(self.LABEL_MAPPING.values())
        present_labels = {d.name for d in input_dir.iterdir() if d.is_dir()}

        missing = required_labels - present_labels
        if missing:
            raise ValueError(f"缺失必要类别目录: {missing}")

    def process_class(self, class_dir, label_id):
        """处理单个类别目录"""
        features, labels = [], []
        valid_images = list(class_dir.glob("*.*"))

        for img_path in tqdm(valid_images, desc=f"处理 {class_dir.name}", unit="img"):
            if not self._is_valid_image(img_path):
                continue

            try:
                landmarks = self._detect_landmarks(img_path)
                if landmarks is not None:
                    feature = self._compute_features(landmarks)
                    features.append(feature)
                    labels.append(label_id)
            except Exception as e:
                print(f"处理 {img_path.name} 出错: {str(e)}")

        return features, labels

    def _is_valid_image(self, path):
        """验证图像文件有效性和扩展名"""
        valid_exts = {".png", ".jpg", ".jpeg"}
        return path.suffix.lower() in valid_exts and path.stat().st_size > 0

    def _detect_landmarks(self, img_path):
        """检测并返回手部关键点"""
        img = cv2.imread(str(img_path))
        if img is None:
            return None

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        
        if results.multi_hand_landmarks:
            return results.multi_hand_landmarks[0].landmark
        return None

    def _compute_features(self, landmarks):
        """从关键点计算特征向量"""
        # 基础坐标特征 (规范化为相对坐标)
        base_x = [lm.x for lm in landmarks]
        base_y = [lm.y for lm in landmarks]
        
        # 几何特征计算
        palm_w = abs(landmarks[0].x - landmarks[9].x)
        palm_h = abs(landmarks[0].y - landmarks[9].y)
        angles = self._compute_finger_angles(landmarks)

        return base_x + base_y + [palm_w, palm_h] + list(angles.values())

    def _compute_finger_angles(self, landmarks):
        """精确计算各手指平均角度"""
        joints = {
            'thumb': [1, 2, 3, 4],
            'index': [5, 6, 7, 8],
            'middle': [9, 10, 11, 12],
            'ring': [13, 14, 15, 16],
            'pinky': [17, 18, 19, 20]
        }

        angles = {}
        for finger, indices in joints.items():
            total = 0.0
            valid = 0
            
            for i in range(len(indices)-2):
                try:
                    a, b, c = landmarks[i], landmarks[i+1], landmarks[i+2]
                    angle = self._calculate_angle(a, b, c)
                    if not np.isnan(angle):
                        total += angle
                        valid += 1
                except IndexError:
                    continue

            avg_angle = total / valid if valid > 0 else 0.0
            angles[f"{finger}_avg_deg"] = avg_angle

        return angles

    def _calculate_angle(self, a, b, c):
        """安全的向量夹角计算（处理零向量）"""
        vec_ab = np.array([b.x - a.x, b.y - a.y])
        vec_cb = np.array([b.x - c.x, b.y - c.y])

        norm_ab = np.linalg.norm(vec_ab)
        norm_cb = np.linalg.norm(vec_cb)

        if norm_ab < 1e-6 or norm_cb < 1e-6:
            return 0.0

        cosine = np.dot(vec_ab, vec_cb) / (norm_ab * norm_cb)
        return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

    def _save_dataset(self, df, output_path):
        """保存预处理后的数据集"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            df.to_csv(output_path, index=False, float_format="%.6f")
            print(f"\n💾 数据集保存至 {output_path}（共计 {len(df)} 样本）")
        except Exception as e:
            raise RuntimeError(f"保存CSV失败: {str(e)}")

    def _save_scaler(self, scaler_path, features):
        """训练并保存标准化器"""
        try:
            scaler = StandardScaler().fit(features)
            joblib.dump(scaler, scaler_path)
            print(f"📦 标准化器保存至 {scaler_path}")
        except Exception as e:
            raise RuntimeError(f"保存标准化器失败: {str(e)}")

if __name__ == "__main__":
    processor = GesturePreprocessor()
    
    input_data = Path("data/raw")
    output_dir = Path("data/processed")
    
    try:
        processor.process_dataset(input_data, output_dir)
    except Exception as e:
        print(f"❌ 处理失败: {str(e)}")
