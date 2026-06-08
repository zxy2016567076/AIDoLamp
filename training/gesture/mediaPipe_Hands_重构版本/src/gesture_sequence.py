"""
连续手势序列识别模块
支持定义多个连续手势组成的操作指令
"""
import time
from collections import deque

class GestureSequenceDetector:
    def __init__(self, sequence_definitions):
        """
        :param sequence_definitions: 预定义的序列配置
            Example:
            {
                "lock_screen": ["fist", "palm", "fist"],
                "volume_up": ["thumbs_up", "thumbs_up"] 
            }
        """
        self.sequences = sequence_definitions
        self.history = deque(maxlen=10)  # 保存最近10次检测结果
        self.cooldown = 2.0  # 操作触发后冷却时间（秒）
        self.last_trigger_time = 0
    
    def _is_in_cooldown(self):
        """冷却时间检查"""
        return (time.time() - self.last_trigger_time) < self.cooldown

    def update(self, current_gesture):
        """更新当前手势到检测队列"""
        if current_gesture and not self._is_in_cooldown():
            self.history.append((time.time(), current_gesture))
    
    def _find_subsequence(self, pattern, tolerance=1.0):
        """
        在历史记录中寻找匹配的模式序列
        tolerance: 允许的最大手势间隔时间（秒）
        """
        target_len = len(pattern)
        if len(self.history) < target_len: return False
        
        # 逆向查找以提高效率
        for i in range(len(self.history)-target_len+1):
            subsequence = self.history[i:i+target_len]
            gestures = [item[1] for item in subsequence]
            timestamps = [item[0] for item in subsequence]

            # 检查时间连续性
            time_valid = all(
                (timestamps[j+1] - timestamps[j]) <= tolerance
                for j in range(target_len-1)
            )

            # 检查手势匹配
            gesture_match = all(
                gestures[j] == pattern[j]
                for j in range(target_len)
            )

            if time_valid and gesture_match:
                return True
        return False

    def check_sequences(self):
        """检查是否匹配任何预定义序列"""
        matched_actions = []
        for action_name, pattern in self.sequences.items():
            if self._find_subsequence(pattern):
                matched_actions.append(action_name)
                self.last_trigger_time = time.time()
                self.history.clear()  # 触发后清空历史
        
        return matched_actions

# 示例使用
if __name__ == "__main__":
    # 配置需要检测的复杂手势组合
    SEQUENCE_CONFIG = {
        "emergency_call": ["fist", "open_palm", "fist"],
        "take_screenshot": ["peace", "thumbs_up"]
    }
    
    detector = GestureSequenceDetector(SEQUENCE_CONFIG)
    
    # 模拟实时检测流
    test_sequence = ["fist", "open_palm", "fist", "peace", "thumbs_up"]
    for ges in test_sequence:
        detector.update(ges)
        print("Current history:", [g for (t,g) in detector.history])
        if actions := detector.check_sequences():
            print(f"Action triggered: {actions}")
