# run.py
"""
主入口：整合所有模块，启动双声道主动降噪、实时谱图展示与声源触发控制。
功能：
  1. 初始化 LMSANC 算法（双声道）
  2. 实例化 SpectrogramVisualizer 用于实时谱图
  3. 启动 SourceDetector 后台线程，周期性检测声源
  4. 使用 AudioIO 进行双通道输入输出
  5. 在回调中执行 ANC，更新谱图，并根据触发声源停止输出
  6. 控制台实时打印当前声源与置信度
"""
import time
import numpy as np

from config import config               # 仅用于 SAMPLE_RATE 及声源检测参数
from audio_io import AudioIO
from anc_lms import LMSANC
from spectrogram_visualizer import LevelVisualizer
from source_detector import SourceDetector


def main():
    # ===== 参数设置 =====
    samplerate     = config.SAMPLE_RATE       # 采样率（与声源检测保持一致）
    block_size     = 1024                     # 音频块大小
    channels       = 2                        # 双声道
    filter_len     = 128                      # LMS 滤波器长度
    step_size      = 0.001                    # LMS 步长
    trigger_source = "Talk"              # 当检测到该声源时暂停 ANC（可修改为任意类别）

    # ===== 模块初始化 =====
    anc       = LMSANC(filter_len=filter_len, step_size=step_size, channels=channels)
    visualizer= LevelVisualizer(block_size=block_size,
                                  max_points=config.LEVEL_MAX_POINTS,
                                  vmin=config.LEVEL_VMIN,
                                  vmax=config.LEVEL_VMAX)
    detector  = SourceDetector()
    detector.start()
    audio_io  = AudioIO(samplerate=samplerate,
                        block_size=block_size,
                        channels=channels)

    # ===== 音频回调函数 =====
    def audio_callback(indata, outdata, frames, time_info, status):
        # 1. LMS ANC 处理
        e = anc.process_block(indata.copy())
        # 2. 获取当前声源信息
        name, prob = detector.get_current_source()
        # 3. 根据触发条件决定是否输出降噪信号
        if name == trigger_source:
            outdata[:] = np.zeros_like(e)
        else:
            outdata[:] = e
            visualizer.push_block(e)
        # 4. 控制台实时打印状态
        print(f"Current Source: {name} | Confidence: {prob:.2f}", end='\r')

    # ===== 启动音频流 =====
    audio_io.start_stream(audio_callback)
    print("ANC system is running. Press Ctrl+C to exit.")

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping ANC system...")
    finally:
        audio_io.stop_stream()
        detector.stop()
        visualizer.stop()


if __name__ == '__main__':
    main()
