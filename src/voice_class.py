import pyaudio
import wave
import os
import requests
from aip import AipSpeech
from pydub import AudioSegment
from pydub.playback import play
import time


class VoiceAssistant:
    def __init__(self, config):
        """初始化语音助手"""
        # 加载所有配置
        self._load_config(config)

        # 初始化API客户端
        self.baidu_client = AipSpeech(self.app_id, self.api_key, self.secret_key)

        # 音频参数
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000

    def _load_config(self, config):
        """加载配置参数"""
        self.app_id = config.get("BAIDU_APP_ID", "")
        self.api_key = config.get("BAIDU_API_KEY", "")
        self.secret_key = config.get("BAIDU_SECRET_KEY", "")
        self.deepseek_key = config.get("DEEPSEEK_API_KEY", "")
        self.weather_key = config.get("WEATHER_API_KEY", "")

        # API地址
        self.deepseek_url = "https://api.deepseek.com/v1/chat/completions"
        # 和风天气 API 地址，请替换为你自己的 API Host（控制台可查）
        self.weather_url = "https://devapi.qweather.com/v7/weather/now"

    # ----------------- 音频处理核心方法 -----------------
    def record(self, filename="output.wav", duration=5):
        """录制音频"""
        p = pyaudio.PyAudio()
        stream = p.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK,
        )

        print("录音中...")
        frames = []
        for _ in range(0, int(self.RATE / self.CHUNK * duration)):
            frames.append(stream.read(self.CHUNK))

        stream.stop_stream()
        stream.close()
        p.terminate()

        with wave.open(filename, "wb") as wf:
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(p.get_sample_size(self.FORMAT))
            wf.setframerate(self.RATE)
            wf.writeframes(b"".join(frames))

    def speech_to_text(self, filename="output.wav"):
        """语音转文字"""
        with open(filename, "rb") as f:
            audio_data = f.read()

        result = self.baidu_client.asr(audio_data, "wav", 16000, {"dev_pid": 1537})
        return result.get("result", [""])[0] if result.get("err_no") == 0 else ""

    def text_to_speech(self, text, filename="output.mp3"):
        """文字转语音"""
        result = self.baidu_client.synthesis(
            text, "zh", 1, {"vol": 5, "spd": 4, "pit": 5}
        )

        if not isinstance(result, dict):
            with open(filename, "wb") as f:
                f.write(result)
            return filename
        return None

    def play_audio(self, filename):
        """播放音频"""
        audio = AudioSegment.from_file(filename)
        play(audio)
        os.remove(filename)  # 清理临时文件

    def voice_feedback(self, text):
        """直接语音播报不经过LCD"""
        if text:
            tts_file = self.text_to_speech(text)
            if tts_file:
                self.play_audio(tts_file)

    # ----------------- 功能模块 -----------------
    def _call_deepseek(self, prompt):
        """调用Deepseek API"""
        headers = {
            "Authorization": f"Bearer {self.deepseek_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.8,
        }
        try:
            response = requests.post(
                self.deepseek_url, json=payload, headers=headers, timeout=10
            )
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            return "服务暂时不可用"
        except Exception as e:
            print(f"API请求失败: {e}")
            return "网络连接异常"

    def _get_weather(self, location="101010100"):  # 默认城市码示例，可改为你所在城市
        """获取天气信息"""
        params = {
            "location": location,
            "key": self.weather_key,
        }
        try:
            response = requests.get(self.weather_url, params=params, timeout=5)
            data = response.json()
            if data.get("code") == "200":
                now = data["now"]
                location = "本地"  # 显示用城市名，可自定义
                return f"{location}天气：{now['text']}，温度{now['temp']}℃，湿度{now['humidity']}%"
            return "天气查询失败"
        except Exception as e:
            print(f"天气查询异常: {e}")
            return "无法获取天气信息"

    def process_command(self, text):
        """处理用户指令"""
        text = text.strip().replace("，", ",")  # 统一中文逗号处理
        text = text.strip()
        if not text:
            return ""

        if "退出" in text:
            return "[EXIT]"  # 定义退出指令标识
        if "你好,悠悠" in text or "你好悠悠" in text:  # 兼容两种说法
            return "[ACTIVATED]"  # 激活成功标识

        if "今天天气怎么样" in text:
            parts = text.split("今天天气怎么样")
            location = parts[0].strip() or "101010100"
            return self._get_weather(location)
        elif "讲个故事" in text:
            return self._call_deepseek("请用中文讲一个有趣的儿童故事，长度约100字")
        elif "你是谁" in text:
            return "我是悠悠,你的AI智能台灯"
        else:
            return self._call_deepseek(text)

    # ----------------- 主控制流程 -----------------
    def run(self):
        """启动语音助手"""
        try:
            while True:
                input("按回车键开始录音...（说'退出'结束）")
                self.record()  # 录制音频
                text = self.speech_to_text()
                print("识别结果:", text)

                if "退出" in text:
                    break

                response = self.process_command(text)
                print("系统响应:", response)
                if response:
                    tts_file = self.text_to_speech(response)  # 生成语音
                    if tts_file:
                        self.play_audio(tts_file)  # 播放语音
        except KeyboardInterrupt:
            print("\n程序已退出")
        except Exception as e:
            print(f"程序异常: {e}")


if __name__ == "__main__":
    # 配置信息从环境变量读取，参考 .env.example
    config = {
        "BAIDU_APP_ID": os.environ.get("BAIDU_APP_ID", ""),
        "BAIDU_API_KEY": os.environ.get("BAIDU_API_KEY", ""),
        "BAIDU_SECRET_KEY": os.environ.get("BAIDU_SECRET_KEY", ""),
        "DEEPSEEK_API_KEY": os.environ.get("DEEPSEEK_API_KEY", ""),
        "WEATHER_API_KEY": os.environ.get("WEATHER_API_KEY", ""),
    }

    assistant = VoiceAssistant(config)

    time.sleep(3)
    assistant.voice_feedback("请注意不要将头低太低")
    time.sleep(3)
    assistant.voice_feedback("请注意不要将肩膀倾斜")

    assistant.run()
