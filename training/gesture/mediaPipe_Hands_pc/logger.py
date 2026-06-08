"""
日志模块：提供项目中所有日志记录功能
"""
import os
import logging
from datetime import datetime
from config import LOG_DIR

def setup_logger(name, log_file=None, level=logging.INFO):
    """设置并返回一个命名的日志记录器"""
    logger = logging.getLogger(name) 
    logger.setLevel(level) 
    
    # 防止重复添加处理器
    if logger.handlers:
        return logger
        
    # 控制台输出
    console_handler = logging.StreamHandler() 
    console_handler.setLevel(level) 
    console_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s') 
    console_handler.setFormatter(console_format) 
    logger.addHandler(console_handler) 
    
    # 文件输出
    if log_file:
        os.makedirs(LOG_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(LOG_DIR, f"{log_file}_{timestamp}.log")
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(level)
        file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    return logger
