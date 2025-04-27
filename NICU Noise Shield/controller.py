# src/controller.py
"""
主控模块：协调音频输入输出、ANC 算法、实时谱图可视化与声源检测
"""
import numpy as np
import time
import threading

from config import config
from audio_io import AudioIO
from anc_lms import LMSANC
from spectrogram_visualizer import LevelVisualizer
from source_detector import SourceDetector


class ANCController:
    def __init__(self):
        # 加载配置
        self.sr = config.SAMPLE_RATE
        self.block_size = config.BLOCK_SIZE
        self.channels = config.CHANNELS
        self.filter_len = config.FILTER_LEN
        self.step_size = config.STEP_SIZE
        self.trigger_source = config.TRIGGER_SOURCE

        # 初始化模块
        self.anc = LMSANC(filter_len=self.filter_len,
                          step_size=self.step_size,
                          channels=self.channels)
        self.visualizer = LevelVisualizer(block_size=self.block_size,
                                         max_points=config.LEVEL_MAX_POINTS,
                                         vmin=config.LEVEL_VMIN,
                                         vmax=config.LEVEL_VMAX)
        self.detector  = SourceDetector()
        self.detector = SourceDetector()
        self.audio_io = AudioIO(samplerate=self.sr,
                                 block_size=self.block_size,
                                 in_device=config.INPUT_DEVICE,
                                 out_device=config.OUTPUT_DEVICE,
                                 channels=self.channels)
        self._running = False

    def _audio_callback(self, indata, outdata, frames, time_info, status):
        # ANC 误差信号
        e = self.anc.process_block(indata.copy())
        # 查询声源
        name, prob = self.detector.get_current_source()
        # 根据触发条件决定是否输出
        if name == self.trigger_source:
            outdata[:] = np.zeros_like(e)
        else:
            outdata[:] = e
            # 推送谱图更新
            self.visualizer.push_block(e)
        # 打印状态
        print(f"Source: {name} | Confidence: {prob:.2f}", end='\r')

    def start(self):
        """启动控制流程"""
        if self._running:
            return
        self._running = True
        # 启动声源检测
        self.detector.start()
        # 启动音频流
        self.audio_io.start_stream(self._audio_callback)
        print("ANCController started. Press Ctrl+C to stop.")
        # 主循环保持运行
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nStopping ANCController...")
            self.stop()

    def stop(self):
        """停止所有服务"""
        if not self._running:
            return
        self._running = False
        self.audio_io.stop_stream()
        self.visualizer.stop()
        self.detector.stop()


if __name__ == '__main__':
    controller = ANCController()
    controller.start()
