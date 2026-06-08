"""
实时数据采集与自动分类工具
按数字键1-5将当前帧保存到对应类别文件夹
"""
import cv2
import os
import datetime

class DataCollector:
    def __init__(self, output_dir="data/raw"):
        self.cap = cv2.VideoCapture(0)
        self.output_dir = output_dir
        self.class_folders = {
            ord('1'): 'mode1', 
            ord('2'): 'mode2',
            ord('3'): 'mode3',
            ord('4'): 'mode4',
            ord('5'): 'activate',
            ord('6'): 'up',
            ord('7'): 'down',
            ord('8'): 'left',
            ord('9'): 'right', 
            ord('0'): 'exit'
        }
        
        # 创建输出目录
        for folder in set(self.class_folders.values()):
            os.makedirs(os.path.join(output_dir, folder), exist_ok=True)

    def capture_loop(self):
        print("""
        ==== 数据采集指南 ====
        按数字键保存对应类别：
        1 -> 模式1手势
        2 -> 模式2手势  
        3 -> 模式3手势
        4 -> 模式4手势
        5 -> 激活手掌
        6 -> 向上手势
        7 -> 向下手势
        8 -> 向左手势
        9 -> 向右手势
        0 -> 退出手势
        ESC -> 退出采集
        ====================
        """)
        
        while True:
            ret, frame = self.cap.read()
            if not ret: break
            
            cv2.imshow("Capture [按数字键记录]", frame)
            key = cv2.waitKey(1)
            
            # 处理按键输入
            if key == 27:  # ESC
                break
            elif key in self.class_folders:
                self._save_frame(key, frame)

    def _save_frame(self, key, frame):
        """保存帧到对应类别"""
        class_name = self.class_folders[key]
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{class_name}_{timestamp}.jpg"
        output_path = os.path.join(self.output_dir, class_name, filename)
        
        # 水平翻转以保持手势镜像
        cv2.imwrite(output_path, cv2.flip(frame, 1))
        print(f"Saved: {output_path}")

if __name__ == "__main__":
    collector = DataCollector()
    collector.capture_loop()
    collector.cap.release()
    cv2.destroyAllWindows()
