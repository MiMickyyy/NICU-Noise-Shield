# config.py
"""
全局配置文件，集中管理项目所有模块参数。
"""
import os

class Config:
    # === 音频 I/O 参数 ===
    SAMPLE_RATE = 44100               # 录音与播放采样率（Hz）
    ANC_CHANNELS = 2                  # ANC 系统使用双声道
    RECORD_CHANNELS = 1               # 声源检测录音通道数（单声道）
    CHANNELS = ANC_CHANNELS

    INPUT_DEVICE = None               # 输入设备索引（None 表示默认）
    OUTPUT_DEVICE = None              # 输出设备索引

    BLOCK_SIZE = 1024                 # 音频处理块大小

    # === LMS ANC 算法参数 ===
    FILTER_LEN = 128                  # LMS 滤波器长度（tap 数）
    STEP_SIZE = 0.001                 # LMS 步长因子 mu

    # === 声源检测参数 ===
    RECORD_SECONDS = 3                # 每次录制时长（秒）
    RECORD_INTERVAL = 1               # 录制触发间隔（秒）
    N_FFT = 2048                      # STFT/梅尔谱 FFT 点数
    HOP_LENGTH = 512                  # STFT/梅尔谱帧移长度
    SPECTROGRAM_SHAPE = (128, 128, 1) # 输入给 CNN 的谱图形状
    MODELS_DIR = r"C:\Users\Micky\Documents\NICU Noise Classifier\models"  # 预训练模型路径目录
    CLASS_LABELS = {                  # 声源分类标签映射
        0: "Talk",
        1: "Machine",
        2: "Warning",
        3: "Walk",
        4: "Normal"
    }
    CONF_THRESHOLD = 0.85             # 推理置信度阈值
    NORMAL_CLASS_INDEX = 4            # 置信度低于阈值的归为该类别索引

    LEVEL_MAX_POINTS = 100  # LevelVisualizer 最大点数
    LEVEL_VMIN = -120.0  # 最小 dB 值
    LEVEL_VMAX = 0.0  # 最大 dB 值


# 创建配置实例
config = Config()

