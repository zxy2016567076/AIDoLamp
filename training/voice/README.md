# 🗣️ Voice Assistant (智能语音助手)

本项目集成了百度语音 API (ASR/TTS)、DeepSeek 大模型对话以及和风天气查询功能，实现了自然语言交互的语音助手。

## 📂 文件说明

### 1. `main.py` (核心主程序)
*   **功能**：整个系统的单一入口。
*   **特性**：
    *   **语音输入**：自动录音并调用百度 API 转文字。
    *   **智能回复**：调用 DeepSeek (R1/Chat) 生成回答。
    *   **功能指令**：支持"天气"、"讲个故事"、"打印"等指令。
    *   **语音播报**：将回复内容转为语音播放。

### 2. `input.wav` / `output.wav` / `output.mp3`
*   运行时产生的临时文件，会自动生成和覆盖，无需手动管理。
*   (建议在 `.gitignore` 中忽略这些文件)

## 🚀 如何运行

1. 确保已安装依赖库 (`requirements.txt`):
   ```bash
   pip install pyaudio baidu-aip pydub requests playsound
   ```

2. 运行主程序:
   ```bash
   python main.py
   ```

3. 按提示操作：
   *   按回车键开始说话。
   *   说完后等待系统响应。
   *   说 "退出" 结束程序。

## ⚠️ 注意事项
*   本项目使用了硬编码的 API Key (百度, DeepSeek, 和风天气)。
*   **请勿** 将 `main.py` 直接公开上传到 GitHub，除非你已经删除了其中的 Key。
