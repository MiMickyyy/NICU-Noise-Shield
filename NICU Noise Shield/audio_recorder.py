"""
音频录制工具模块：src/audio_recorder.py
封装录音功能，提供一个简单的接口 record()，用于录制指定时长的音频，
录音采样率和通道数均从配置文件中读取（采样率为44100Hz，单通道）。
"""

import os
import wave
import numpy as np
import sounddevice as sd
from config import config


def record(duration=None):
    """
    Record a segment of audio.

    :param duration: Duration in seconds (defaults to config.RECORD_SECONDS)
    :return: Recorded audio data as a 1D NumPy array
    """
    if duration is None:
        duration = config.RECORD_SECONDS
    num_samples = int(duration * config.SAMPLE_RATE)
    print(f"Recording for {duration} seconds at {config.SAMPLE_RATE} Hz...")
    audio = sd.rec(num_samples, samplerate=config.SAMPLE_RATE, channels=config.CHANNELS, dtype='float32')
    sd.wait()  # Wait until recording is finished
    print("Recording finished.")
    return np.squeeze(audio)


def save_audio(audio_data, file_path, sample_rate):
    """
    Save the recorded audio data as a WAV file.

    :param audio_data: The audio data (NumPy array of type float32 in range [-1, 1])
    :param file_path: Full path where the WAV file will be saved
    :param sample_rate: The sampling rate to be set in the WAV file header
    """
    # Normalize audio_data and convert to int16
    max_val = np.max(np.abs(audio_data))
    if max_val > 0:
        audio_data = audio_data / max_val
    scaled = np.int16(audio_data * 32767)

    with wave.open(file_path, 'w') as wf:
        wf.setnchannels(config.CHANNELS)
        wf.setsampwidth(2)  # 16-bit audio = 2 bytes
        wf.setframerate(sample_rate)
        wf.writeframes(scaled.tobytes())
    print(f"Audio saved to: {file_path}")


if __name__ == '__main__':
    # Record a 3-second audio clip using the record() function
    audio_data = record()

    # Ensure the recordings directory exists
    os.makedirs(config.RECORDINGS_DIR, exist_ok=True)

    # Define the file path for the test recording (e.g., test_record.wav)
    test_filename = "test_record.wav"
    file_path = os.path.join(config.RECORDINGS_DIR, test_filename)

    # Save the recorded audio to the file
    save_audio(audio_data, file_path, config.SAMPLE_RATE)

    print("Test recording saved successfully.")
