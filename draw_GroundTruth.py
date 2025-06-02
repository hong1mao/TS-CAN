import json
import matplotlib.pyplot as plt
import numpy as np
from not_realtime import RealTimeRPPG


def plot_ppg_from_json(json_file_path):
    """
    从 JSON 文件中读取 PPG 波形数据并绘图，横轴为真实时间（秒）
    :param json_file_path: JSON 文件路径
    """
    # 读取 JSON 文件
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    full_package = data.get('/FullPackage', [])
    if not full_package:
        print("No PPG waveform data found.")
        return

    # 提取 waveform 和 Timestamp
    ppg_values = [item['Value']['waveform'] for item in full_package]
    timestamps = [item['Timestamp'] for item in full_package]

    # 转换为 numpy 数组便于处理
    timestamps = np.array(timestamps, dtype=np.int64)
    ppg_values = np.array(ppg_values, dtype=np.float32)

    # 时间戳转为秒，并以第一个时间戳为 0 点
    start_time = timestamps[0]
    time_axis = (timestamps - start_time) / 1e9  # 纳秒 -> 秒

    # 截取前10秒的数据
    mask = time_axis <= 10
    time_axis = time_axis[mask]
    ppg_values = ppg_values[mask]
    # 计算采样率
    sampling_rate = 1 / np.mean(np.diff(time_axis))
    print(f"Sampling Rate: {sampling_rate} Hz")
    # 带通滤波
    ppg_values = RealTimeRPPG.butter_bandpass(ppg_values, 0.6, 4, sampling_rate, order=2)

    # 绘制波形图
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(time_axis, ppg_values, color='b', label='PPG Waveform')
    plt.title('PPG Waveform from JSON (Aligned by Timestamp)')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Waveform Value')
    plt.grid(True)
    plt.legend()

    # 绘制频谱图
    plt.subplot(2, 1, 2)
    window = np.hanning(len(ppg_values))
    ppg_windowed = ppg_values * window
    fft_result = np.fft.fft(ppg_windowed)
    fft_magnitude = np.abs(fft_result)

    sampling_rate = 1 / np.mean(np.diff(time_axis))  # 实际采样率（Hz）
    frequencies = np.fft.fftfreq(len(ppg_values), d=1 / sampling_rate)

    positive_frequencies = frequencies[:len(frequencies) // 2]
    fft_magnitude = fft_magnitude[:len(fft_magnitude) // 2]

    # 转换为 BPM（乘以 60）
    bpm_axis = positive_frequencies * 60

    # 打印峰值频率对应的 BPM
    peak_index = np.argmax(fft_magnitude)
    peak_frequency = positive_frequencies[peak_index]
    peak_bpm = peak_frequency * 60
    print(f"Peak BPM: {peak_bpm:.2f}")

    plt.plot(bpm_axis, fft_magnitude, color='b')
    plt.title('FFT Spectrum of PPG Waveform (BPM)')
    plt.xlabel('Heart Rate (BPM)')
    plt.ylabel('Magnitude')
    plt.grid(True)
    plt.xlim([40, 200])  # 显示常见心率范围
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    json_file = r"E:\rppg_dataset\PURE\心率数据\01-01.json"
    plot_ppg_from_json(json_file)
