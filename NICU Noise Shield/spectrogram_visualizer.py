import numpy as np
import matplotlib.pyplot as plt
import threading
import queue

class LevelVisualizer:
    """
    实时声压级 (dB) 曲线可视化：
    根据推入的音频块计算 RMS 电平并转换为 dB，
    并在动态折线图中显示最近若干点。
    """
    def __init__(self,
                 block_size: int = 1024,
                 max_points: int = 100,
                 vmin: float = -120.0,
                 vmax: float = 0.0):
        """
        :param block_size: 每个音频块的样本数，用于电平计算
        :param max_points: 绘图窗口内保留的最大数据点数
        :param vmin: dB 轴最小值
        :param vmax: dB 轴最大值
        """
        self.block_size = block_size
        self.max_points = max_points
        self.vmin = vmin
        self.vmax = vmax

        # 队列与控制信号
        self._q = queue.Queue()
        self._running = True

        # 初始化绘图
        self.fig, self.ax = plt.subplots()
        # 预置数据
        self.levels = [self.vmin] * self.max_points
        self.line, = self.ax.plot(range(self.max_points), self.levels)
        self.ax.set_ylim(self.vmin, self.vmax)
        self.ax.set_xlim(0, self.max_points - 1)
        self.ax.set_xlabel('Frame')
        self.ax.set_ylabel('Level (dB)')
        self.ax.set_title('Real-time Level (dB)')
        plt.ion()
        plt.show()

        # 启动更新线程
        self._thread = threading.Thread(target=self._update_loop, daemon=True)
        self._thread.start()

    def push_block(self, block: np.ndarray):
        """
        接收新的音频块，将其放入队列
        支持多通道，仅使用第一个通道
        """
        if block.ndim > 1:
            block = block[:, 0]
        self._q.put(block.copy())

    def _update_loop(self):
        """
        后台线程：不断获取音频块，计算电平并更新曲线
        """
        while self._running:
            try:
                block = self._q.get(timeout=0.1)
            except queue.Empty:
                continue

            # 计算 RMS
            rms = np.sqrt(np.mean(block ** 2))
            # 转换为 dB
            db = 20 * np.log10(rms + 1e-6)

            # 更新数据窗口
            self.levels.pop(0)
            self.levels.append(db)

            # 更新绘图
            self.line.set_ydata(self.levels)
            self.ax.set_title(f'Real-time Level: {db:.1f} dB')
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.pause(0.001)

    def stop(self):
        """
        停止后台线程并关闭图形
        """
        self._running = False
        self._thread.join(timeout=1)
        plt.close(self.fig)
