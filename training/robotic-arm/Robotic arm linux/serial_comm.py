import serial
import threading
import time


class SerialCommunicator:
    def __init__(self, port='/dev/ttyAMA0', baudrate=115200):
        self.ser = None
        self.running = False
        self.receive_thread = None
        self.last_received = None
        self.last_distance = None
        self.base_distance = None
        self.lock = threading.Lock()
        self.mode_change_callback = None  # 模式切换回调函数
        
        # 定义各舵机的角度范围 [最小值, 最大值, 硬件中值]
        self.servo_ranges = [
        [-135, 135, 90],   # 舵机1: -135~135度，中值0°对应硬件值90
        [-90, 90, 90],     # 舵机2: -90~90度，中值0°对应硬件值90
        [0, 150, 30],      # 舵机3: 0~150度，中值75°对应硬件值30（需校准）
        [0, 130, 65]       # 舵机4: 0~130度，中值65°对应硬件值65
    ]
        
        # 当前舵机位置
        self.current_angles = [0, 0, 0, 0]  # 默认初始位置（每个舵机的中间位置）

        # 串口初始化
        try:
            self.ser = serial.Serial(
                port=port,
                baudrate=baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=1
            )
            print(f"串口初始化成功: {port} @ {baudrate}bps")
        except Exception as e:
            print(f"[严重错误] 串口初始化失败: {str(e)}")
            self.stop()

    # ---------------------- 模式1：接收控制 ----------------------
    def start_mode_one(self):
        """启动模式一（持续接收直到距离变化>5cm）"""
        print("==== 进入模式1 ====")
        self.base_distance = None
        self._start_receiver()

    def _start_receiver(self):
        """启动接收线程（内部方法）"""
        if not self.running and self.ser:
            self.running = True
            self.receive_thread = threading.Thread(
                target=self._receive_loop,
                daemon=True
            )
            self.receive_thread.start()

    def _receive_loop(self):
        """接收线程主循环"""
        while self.running and self.ser:
            try:
                if self.ser.in_waiting > 0:
                    raw_data = self.ser.readline()
                    decoded_data = raw_data.decode('utf-8').strip()
                    
                    if decoded_data.startswith("DIST_"):
                        new_distance = float(decoded_data[5:])
                        with self.lock:
                            # 设置基准距离
                            if self.base_distance is None:
                                self.base_distance = new_distance
                                print(f"[模式1] 基准距离已设置: {self.base_distance} cm")
                            
                            # 计算变化量
                            delta = abs(new_distance - self.base_distance)
                            print(f"[模式1] 当前距离: {new_distance} cm | 变化量: {delta:.2f} cm")
                            
                            # 触发模式切换条件
                            if delta > 5:
                                print("[模式切换] 距离变化超过5cm，准备切换模式")
                                self.stop_mode_one()
                                if self.mode_change_callback:
                                    self.mode_change_callback()  # 触发回调
                                
                            self.last_distance = new_distance
            except Exception as e:
                print(f"[接收错误] {str(e)}")
                self.stop()
            time.sleep(0.01)

    def stop_mode_one(self):
        """停止模式一"""
        self.running = False
        if self.receive_thread:
            self.receive_thread.join(timeout=1)
        print("==== 模式一已停止 ====")

    # ---------------------- 模式2：发送控制 ----------------------
    def start_mode_two(self):
        """启动模式二（执行发送操作）"""
        print("\n==== 进入模式2 ====")
        self._send_initial_commands()
        print("==== 模式二就绪 ====\n")

    def _send_initial_commands(self):
        """发送初始化指令集（示例）"""
        # 将各舵机设置到中间位置
        init_angles = [
            0,                   # 舵机1: 中间位置 (-135~135)
            0,                   # 舵机2: 中间位置 (-90~90)
            75,                  # 舵机3: 中间位置 (0~150)
            65                   # 舵机4: 中间位置 (0~130)
        ]
        self.send_servos_smooth(init_angles)
        time.sleep(0.2)
        
    def send_servos(self, servo1, servo2, servo3, servo4):
        """直接设置舵机位置（无平滑过渡）"""
        self.send_servos_smooth([servo1, servo2, servo3, servo4], duration=0.1, steps=1)
    
    def send_brightness(self, level):
        """设置灯光亮度"""
        if not 0 <= level <= 100:
            raise ValueError("亮度值必须在0-100范围内")
        self.send_command(f"BRIGHT {level}")
        
    def send_color_temp(self, temp):
        """设置灯光色温"""
        if not 2500 <= temp <= 7500:
            raise ValueError("色温值必须在2500K-7500K范围内")
        self.send_command(f"TEMP {temp}")

    # ---------------------- 核心通信方法 ----------------------
    def send_command(self, command: str):
        """通用指令发送方法（重大修改）"""
        if not command.endswith("\n"):
            command += "\n"
        
        try:
            self.ser.write(command.encode('utf-8'))
            print(f"[TX] 发送指令: {command.strip()}")
        except Exception as e:
            print(f"[TX Error] {str(e)}")
            self.stop()

    # ====================== 模式命令 ======================
    def send_mode(self, mode_num: int):
        """发送工作模式命令（MODE 1 ~ MODE 4）"""
        if mode_num not in [1, 2, 3, 4]:
            raise ValueError("模式编号必须为1-4")
        self.send_command(f"MODE {mode_num}")

    # ====================== 舵机命令 ======================
    def send_servos_smooth(self, target_angles, duration=3.0, steps=20):
        """
        平滑移动舵机到目标角度
        :param target_angles: 目标角度列表 [angle1, angle2, angle3, angle4]
        :param duration: 移动总时间(秒)，默认3秒
        :param steps: 移动步数
        """
        if len(target_angles) != 4:
            raise ValueError("需要4个舵机角度参数")
        
        # 验证各舵机角度是否在其有效范围内
        for i, angle in enumerate(target_angles):
            min_angle, max_angle, _ = self.servo_ranges[i]
            if not min_angle <= angle <= max_angle:
                raise ValueError(f"舵机{i+1}角度需在{min_angle}到{max_angle}范围内，当前值:{angle}")
        
        # 记录开始角度（使用当前角度）
        start_angles = list(self.current_angles)
        
        step_delay = duration / steps
        
        # 平滑移动插值
        for step in range(1, steps + 1):
            # 计算当前步的角度(线性插值)
            interpolated_angles = []
            for i in range(4):
                start = start_angles[i]
                target = target_angles[i]
                current = start + (target - start) * step / steps
                interpolated_angles.append(int(current))
            
            # 直接发送指令，不做额外的角度映射转换
            cmd = f"Alldro {' '.join(map(str, interpolated_angles))}"
            self.send_command(cmd)
            
            # 更新当前位置
            self.current_angles = interpolated_angles.copy()
            
            # 等待指定时间
            time.sleep(step_delay)

    def send_servo_action(self, action_type: str, *args):
        """
        发送舵机动作命令
        :param action_type: 动作类型，包括：
            - 'Adroup_1', 'Adroup_2', 'Adroup_3'
            - 'Adrodown_1', 'Adrodown_2', 'Adrodown_3'
            - 'Adroleft', 'Adroright'
            - 'Alldro'
        :param args: 参数列表（仅Alldro需要4个参数）
        """
        valid_actions = [
            'Adroup_1', 'Adroup_2', 'Adroup_3',
            'Adrodown_1', 'Adrodown_2', 'Adrodown_3',
            'Adroleft', 'Adroright', 'Alldro'
        ]
        
        if action_type not in valid_actions:
            raise ValueError(f"无效动作类型，可用选项：{valid_actions}")
        
        # 参数验证
        if action_type == 'Alldro':
            if len(args) != 4:
                raise ValueError("Alldro需要4个参数")
                
            # 直接使用传入的原始角度值
            # send_servos_smooth会根据每个舵机的实际角度范围进行验证
            self.send_servos_smooth(list(args))
        else:
            if args:
                raise ValueError(f"{action_type} 不需要参数")
            cmd = action_type
            self.send_command(cmd)

    # ====================== 灯光命令 ======================
    def send_light_command(self, command_type: str):
        """
        发送灯光控制命令（支持四种格式）
        :param command_type: 指定命令字符串，可选：
            - 'light_zj'：小写格式指令1
            - 'light_jx'：小写格式指令2
            - 'LIGHT_ZJ'：大写格式指令1
            - 'LIGHT_JX'：大写格式指令2
        """
        valid_commands = [
            'light_zj', 'light_jx', 
            'LIGHT_ZJ', 'LIGHT_JX'
        ]
        
        if command_type not in valid_commands:
            raise ValueError(
                f"无效灯光命令，可用选项：{valid_commands}\n"
                "示例：send_light_command('LIGHT_ZJ')"
            )
        self.send_command(command_type)

    # ====================== OLED命令 ======================
    def send_oled_command(self, oled_type: str):
        """
        发送OLED屏幕命令
        :param oled_type: 命令类型，支持：
            - 'OLED_1'：显示预设界面1
            - 'OLED_2'：显示预设界面2
        """
        if oled_type not in ['OLED_1', 'OLED_2']:
            raise ValueError("OLED命令必须为OLED_1或OLED_2")
        self.send_command(oled_type)

    def set_mode_change_callback(self, callback):
        """设置模式切换回调函数"""
        self.mode_change_callback = callback

    # ---------------------- 系统控制 ----------------------
    def stop(self):
        """安全停止所有通信"""
        self.running = False
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("串口已安全关闭")

    def __del__(self):
        self.stop()

# ---------------------- 使用示例 ----------------------
if __name__ == "__main__":
    # 初始化通信（根据实际修改端口）
    comm = SerialCommunicator(
        port='/dev/ttyAMA0',  # 硬件串口设备 
        baudrate=115200        # 需与STM32完全一致
    )

    print("\n===== 舵机控制示例 =====")
    print("舵机1: 角度范围 -135° ~ 135°")
    print("舵机2: 角度范围 -90° ~ 90°")
    print("舵机3: 角度范围 0° ~ 150°")
    print("舵机4: 角度范围 0° ~ 130°")
    
    # 其他命令示例（已注释）
    # 模式命令
    comm.send_mode(1)
    comm.send_mode(2)
    
    #单一舵机动作
    comm.send_servo_action('Adroup_1')
    comm.send_servo_action('Adrodown_2')
    
    #灯光控制
    comm.send_light_command('LIGHT_ZJ')
    comm.send_light_command('light_jx')
    
    #OLED显示
    comm.send_oled_command('OLED_1')
    comm.send_oled_command('OLED_2')
    
    
    comm.send_servos_smooth([0, 30, 20, 5], duration=2.0)
    time.sleep(1)
