# src/source_detector.py
"""
周期性录音与声源识别模块

功能说明：
  - 每隔 config.RECORD_INTERVAL 秒录制 config.RECORD_SECONDS 秒音频
  - 提取梅尔谱图特征，并加载预训练模型预测声音类别和置信度
  - 置信度低于 THRESHOLD 时归为 Normal 类别
  - 在后台线程持续更新最新检测结果

依赖：确保项目目录下有 src/audio_recorder.py，提供 record(duration) 接口；
确保 config.MODELS_DIR 下存在 cnn_model.h5 模型文件。

提供接口：
  start()             启动后台检测线程
  stop()              停止后台检测线程
  get_current_source() -> (name, prob)  获取最新检测结果
"""
import os
import time
import threading

import numpy as np
import cv2
import librosa
import tensorflow as tf

from config import config
from audio_recorder import record  # 确保 audio_recorder.py 位于 src/ 目录下

# 置信度阈值，低于此阈值归为 Normal 类别
THRESHOLD = 0.85
NORMAL_INDEX = 4


def _load_model():
    """
    加载保存在 config.MODELS_DIR 下的预训练模型 cnn_model.h5
    """
    model_path = os.path.join(config.MODELS_DIR, "cnn_model.h5")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件 {model_path} 不存在！")
    return tf.keras.models.load_model(model_path)


def _process_audio(audio_signal: np.ndarray) -> np.ndarray:
    """
    将原始音频信号转换为模型输入的梅尔谱图特征。
    参考 src/record_inference.py 实现：
      - librosa.feature.melspectrogram
      - librosa.power_to_db
      - cv2.resize
      - expand_dims
    返回单通道特征图，形状(config.SPECTROGRAM_SHAPE)
    """
    # 计算梅尔谱图
    S = librosa.feature.melspectrogram(
        y=audio_signal,
        sr=config.SAMPLE_RATE,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH,
        n_mels=config.SPECTROGRAM_SHAPE[0]
    )
    log_S = librosa.power_to_db(S, ref=np.max)
    # 调整尺寸
    h, w = config.SPECTROGRAM_SHAPE[:2]
    resized = cv2.resize(log_S, (w, h))
    return np.expand_dims(resized, axis=-1)


class SourceDetector:
    """
    后台声源检测器，周期性录制并推理当前音源。
    """
    def __init__(self):
        self.model = _load_model()
        self.interval = config.RECORD_INTERVAL
        self.duration = config.RECORD_SECONDS
        self._current = ("Unknown", 0.0)
        self._running = False
        self._thread = None

    def _detect_loop(self):
        while self._running:
            start_time = time.time()
            # 录音
            audio = record(self.duration)
            # 特征提取 & 推理
            feat = _process_audio(audio)
            feat = np.expand_dims(feat, axis=0)
            preds = self.model.predict(feat)
            prob = float(preds[0].max())
            idx = int(np.argmax(preds, axis=1)[0])
            if prob < THRESHOLD:
                idx = NORMAL_INDEX
            name = config.CLASS_LABELS.get(idx, "Unknown")
            self._current = (name, prob)
            # 控制周期
            elapsed = time.time() - start_time
            to_sleep = self.interval - elapsed
            if to_sleep > 0:
                time.sleep(to_sleep)

    def start(self):
        """启动后台检测"""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._detect_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """停止后台检测"""
        if not self._running:
            return
        self._running = False
        self._thread.join()

    def get_current_source(self) -> tuple:
        """返回最新检测结果 (类别名称, 置信度)"""
        return self._current

