import numpy as np
import matplotlib.pyplot as plt

def plot_pixel_and_label_curves(input_npy_path, label_npy_path):
    """
    绘制视频像素变化曲线和标签变化曲线。

    参数:
        input_npy_path (str): 输入视频数据的 .npy 文件路径。
        label_npy_path (str): 标签数据的 .npy 文件路径。
    """
    # 加载数据
    video_data = np.load(input_npy_path)  # 形状: (T, 6, H, W)
    label_data = np.load(label_npy_path)  # 形状: (T,)

    # 计算视频像素的平均值（按时间轴）
    pixel_mean = np.mean(video_data, axis=(2, 3))  # 形状: (T, 6)

    # 创建画布
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))

    # 绘制视频帧差分像素变化曲线
    ax1.plot(pixel_mean[:, 1], label='Pixel Mean Value', color='blue')
    ax1.set_title('Video Pixel Mean Value Over Time')
    ax1.set_xlabel('Frame Index')
    ax1.set_ylabel('Pixel Mean Value')
    ax1.legend()

    # 绘制视频标准化像素变化曲线
    ax2.plot(pixel_mean[:, 4], label='Pixel Mean Value', color='blue')
    ax2.set_title('Video Pixel Mean Value Over Time')
    ax2.set_xlabel('Frame Index')
    ax2.set_ylabel('Pixel Mean Value')
    ax2.legend()

    # 绘制标签变化曲线
    ax3.plot(label_data, label='Label Signal', color='red')
    ax3.set_title('Label Signal Over Time')
    ax3.set_xlabel('Frame Index')
    ax3.set_ylabel('Label Value')
    ax3.legend()

    # 调整布局并显示
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 示例调用
    input_npy_path = r"D:\UBFC\val_cache\subject8_input0.npy"  # 替换为实际的.npy 文件路径
    label_npy_path = r"D:\UBFC\val_cache\subject8_label0.npy"  # 替换为实际的.npy 文件路径
    plot_pixel_and_label_curves(input_npy_path, label_npy_path)
