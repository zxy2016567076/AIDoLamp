"""
配置文件：集中管理所有项目参数
"""
import os

# 路径配置
DATA_DIR = 'data' #数据路径
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw') #原始数据路径
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed') #处理后数据路径
MODEL_DIR = os.path.join(DATA_DIR, 'processed') #模型路径
LOG_DIR = 'logs' #日志路径

# 确保目录存在
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR, LOG_DIR]:
    os.makedirs(directory, exist_ok=True)

# 特征提取配置
HAND_CONFIDENCE_THRESHOLD = 0.5 #手部置信度阈值
FEATURE_CSV_PATH = os.path.join(PROCESSED_DATA_DIR, 'features.csv') #特征文本路径

# 模型配置
MODEL_PATH = os.path.join(MODEL_DIR, 'gesture_model.pkl') #模型路径
TEST_SIZE = 0.2 #测试集比例
RANDOM_STATE = 42 #随机种子

# 类别映射
CLASS_MAPPING = {
    1: 'activate', 2: 'down', 3: 'exit', 4: 'left',
    5: 'mode1', 6: 'mode2', 7: 'mode3', 8: 'mode4',
    9: 'right', 10: 'up'
}
REVERSE_CLASS_MAPPING = {v: k for k, v in CLASS_MAPPING.items()}

# 实时识别配置
CAMERA_ID = 0 #摄像头
STATIC_IMAGE_MODE = False #是否为静态图像模式
MAX_NUM_HANDS = 1 #最大手数
MIN_DETECTION_CONFIDENCE = 0.7 #最小检测置信度
MIN_TRACKING_CONFIDENCE = 0.5 #最小跟踪置信度
ACTIVATION_GESTURE_DURATION = 3  # 激活手势需要保持的秒数
NORMAL_GESTURE_DURATION = 1      # 普通手势需要保持的秒数
UI_TEXT_COLOR = (0, 255, 0)    # 文本颜色
UI_TEXT_FONT_SIZE = 1 #字体大小
UI_TEXT_THICKNESS = 2 #字体粗细
UI_PROGRESS_BAR_COLOR = (0, 255, 0) #进度条颜色
UI_PROGRESS_BAR_BG_COLOR = (255, 255, 255) #进度条背景颜色
UI_PROGRESS_BAR_WIDTH = 200 #进度条宽度
UI_PROGRESS_BAR_HEIGHT = 20 #进度条高度
UI_PROGRESS_BAR_X = 20 #进度条x坐标
UI_PROGRESS_BAR_Y = 80 #进度条y坐标
UI_TEXT_X = 20 #文本x坐标
UI_TEXT_Y = 50 #文本y坐标
