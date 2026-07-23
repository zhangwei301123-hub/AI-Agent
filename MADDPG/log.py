import logging
import os
import sys
import warnings
from logging.handlers import RotatingFileHandler

class Log:
    """
    日志类（通过 __init__ 实现初始化，支持单例模式和警告捕获）
    适配图片中显示的 NumPy 警告场景
    """
    _instance = None
    _initialized = False  # 单例控制标记

    def __init__(self, 
                 name: str = "app", 
                 log_dir: str = "logs", 
                 level: int = logging.DEBUG,
                 max_bytes: int = 10*1024*1024,  # 10MB
                 backup_count: int = 5):
        """
        初始化方法（实现单例控制）
        """
        if Log._initialized:
            return  # 避免重复初始化

        # 初始化日志系统
        self._init_logger(name, log_dir, level, max_bytes, backup_count)
        
        # 捕获警告到日志（适配图片中的 NumPy 警告）
        warnings.filterwarnings("once", category=DeprecationWarning)
        logging.captureWarnings(True)
        
        Log._initialized = True

    def _init_logger(self, name, log_dir, level, max_bytes, backup_count):
        """核心日志配置"""
        # 创建日志目录（适配图片中的路径显示）
        os.makedirs(log_dir, exist_ok=True)

        # 日志器配置
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # 避免重复添加处理器（关键！）
        if self.logger.handlers:
            return

        # 日志格式（含时间戳，适配图片中的时间显示需求）
        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
            datefmt='%Y/%m/%d %H:%M:%S'  
        )

        # 控制台输出（对应屏幕显示）
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # 文件轮转（适配长期运行场景）
        log_file = os.path.join(log_dir, f"{name}.log")
        file_handler = RotatingFileHandler(
            log_file, 
            maxBytes=max_bytes, 
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def __getattr__(self, name):
        """委托日志方法（支持多参数自动拼接）"""
        # if name in ("info", "debug", "warning", "error", "critical"):
        #     def wrapped(*args, **kwargs):
        #         msg = " ".join(str(a) for a in args)
        #         return getattr(self.logger, name)(msg, **kwargs)
        #     return wrapped
        return getattr(self.logger, name)

# 使用示例（适配图片中的调用场景）
if __name__ == "__main__":
    # 初始化（首次调用创建实例）
    log = Log(name="numpy_ops", log_dir="ao_logs")

    # 模拟图片中的数组操作警告
    import numpy as np
    ragged_data = [[1, 2], [3, 4, 5]]  # 不规则数据
    arr = np.array(ragged_data)  # 触发 VisibleDeprecationWarning
    
    # 记录操作信息（含时间戳）
    log.warning("检测到不规则数组创建: %s", str(ragged_data))
    log.info("-------")
    log.error("555555")
    # log.log_ndarray_operation(arr)  # 实际会报错，仅演示接口用法