# src/audio_io.py

import sounddevice as sd
import numpy as np

class AudioIO:
    def __init__(self,
                 samplerate: int = 48000,
                 block_size: int = 1024,
                 in_device: int = None,
                 out_device: int = None,
                 channels: int = 2):
        """
        Audio I/O 初始化
        :param samplerate: 采样率
        :param block_size: 每个回调块的帧数
        :param in_device: 输入设备索引（None 表示默认）
        :param out_device: 输出设备索引（None 表示默认）
        :param channels: 通道数（双声道）
        """
        self.samplerate = samplerate
        self.block_size = block_size
        self.in_device = in_device
        self.out_device = out_device
        self.channels = channels
        self.stream = None

    def start_stream(self, callback):
        """
        启动流。回调函数 signature:
            def callback(indata: np.ndarray, outdata: np.ndarray, frames: int,
                         time: CData, status: sd.CallbackFlags) -> None
        indata.shape == (frames, channels), 同理 outdata
        """
        if self.stream is not None:
            raise RuntimeError("Stream already running")

        self.stream = sd.Stream(
            samplerate=self.samplerate,
            blocksize=self.block_size,
            dtype='float32',
            latency='low',
            channels=self.channels,
            callback=callback,
            device=(self.in_device, self.out_device)
        )
        self.stream.start()
        print(f"Audio stream started: {self.samplerate} Hz, block {self.block_size} frames")

    def stop_stream(self):
        """停止并关闭流"""
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
            print("Audio stream stopped")
