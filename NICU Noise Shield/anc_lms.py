import numpy as np

class LMSANC:
    """
    双通道基于 LMS 算法的主动降噪滤波器
    每个通道具有独立的自适应滤波器，使用输入信号本身进行噪声估计并减除。
    """
    def __init__(self, filter_len: int = 128, step_size: float = 0.001, channels: int = 2):
        """
        :param filter_len: 滤波器长度（tap 数）
        :param step_size: LMS 步长因子（mu）
        :param channels: 声道数（例如双声道 = 2）
        """
        self.filter_len = filter_len
        self.mu = step_size
        self.channels = channels
        # 滤波器权重矩阵，形状 (channels, filter_len)
        self.weights = np.zeros((channels, filter_len), dtype=np.float32)
        # 输入缓冲区，保存最近 filter_len 个样本，形状 (channels, filter_len)
        self.x_buf = np.zeros((channels, filter_len), dtype=np.float32)

    def process_block(self, x_block: np.ndarray) -> np.ndarray:
        """
        对输入音频块进行 LMS 降噪处理

        :param x_block: np.ndarray, shape (block_size, channels), 原始输入
        :return: e: np.ndarray, shape (block_size, channels), 降噪后的误差信号
        """
        block_size, chs = x_block.shape
        assert chs == self.channels, "输入通道数与初始化不符"
        # 输出误差信号
        e = np.zeros_like(x_block, dtype=np.float32)

        for n in range(block_size):
            x_n = x_block[n]  # 当前帧各通道样本
            # 更新输入缓冲区
            self.x_buf[:, 1:] = self.x_buf[:, :-1]
            self.x_buf[:, 0] = x_n

            # 计算滤波输出 y[n] = w^T x_buf
            y_n = np.sum(self.weights * self.x_buf, axis=1)
            # 计算误差信号 e[n] = x[n] - y[n]
            e_n = x_n - y_n
            e[n] = e_n

            # 使用 LMS 规则更新权重: w += 2 * mu * e[n] * x_buf
            self.weights += 2 * self.mu * (e_n[:, None] * self.x_buf)

        return e
