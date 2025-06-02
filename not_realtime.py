import logging
import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.signal import butter, filtfilt

from model.TS_CAN import TSCAN
from ultralytics import YOLO
from data.ubfc_rppg_dataset import UBFCrPPGDataset

class RealTimeRPPG:
    def __init__(self, model_path, yolo_model_path="D:\\文档\\python项目\\models\\best_face.pt", re_size=72,
                 larger_box_coef=1.5, video_path=None, accumulate_output=False, use_camera=False, use_face_detection=True):
        """Initialize real-time rPPG system."""
        self.re_size = re_size
        self.larger_box_coef = larger_box_coef
        self.frame_depth = 10  # 根据你的模型设置
        self.video_path = video_path
        self.use_camera = use_camera
        self.use_face_detection = use_face_detection
        # 加载模型
        self.model = TSCAN(frame_depth=self.frame_depth, img_size=self.re_size)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

        # 加载YOLO模型
        self.yolo_model = YOLO(yolo_model_path)
        # 设置日志级别为 WARNING，以抑制 INFO 级别的输出
        logging.getLogger('ultralytics').setLevel(logging.WARNING)

        # 初始化缓冲区
        self.frame_buffer = []
        self.rppg_history = []
        # 选择是否累积输出
        self.accumulate_output = accumulate_output
        # 记录视频的FPS
        self.fps = None

    def face_detection(self, frame):
        """Detect face in a single frame using YOLO."""
        results = self.yolo_model(frame)
        best_box = None
        max_area = 0

        for box in results[0].boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, box)
            area = (x2 - x1) * (y2 - y1)
            if area > max_area:
                max_area = area
                best_box = (x1, y1, x2, y2)

        if best_box is None:
            print("ERROR: No Face Detected")
            return [0, 0, frame.shape[0], frame.shape[1]]

        x_min, y_min, x_max, y_max = best_box
        width = x_max - x_min
        height = y_max - y_min

        # Apply larger box coefficient
        center_x = x_min + width // 2
        center_y = y_min + height // 2
        square_size = max(width, height)
        square_size = int(square_size * self.larger_box_coef)

        new_x = max(0, center_x - square_size // 2)
        new_y = max(0, center_y - square_size // 2)

        return [new_x, new_y, square_size, square_size]

    def crop_and_resize(self, frame, face_box):
        """Crop and resize face region."""
        x, y, w, h = face_box
        face_region = frame[y:y + h, x:x + w]
        resized_frame = np.zeros((1, self.re_size, self.re_size, 3))
        resized_frame[0] = cv2.resize(face_region, (self.re_size, self.re_size), interpolation=cv2.INTER_AREA)
        return resized_frame

    def preprocess_frame(self, frame):
        """Preprocess a single frame."""
        # BGR转换为RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if self.use_face_detection:
            face_box = self.face_detection(frame)
            resized_frame = self.crop_and_resize(frame, face_box)
        else:
            resized_frame = np.zeros((1, self.re_size, self.re_size, 3))
            resized_frame[0] = cv2.resize(frame, (self.re_size, self.re_size), interpolation=cv2.INTER_AREA)

        return resized_frame

    def diff_normalize_data(self, data):
        """Apply differential normalization to video data."""
        n, h, w, c = data.shape
        diffnormalized_len = n - 1
        diffnormalized_data = np.zeros((diffnormalized_len, h, w, c), dtype=np.float32)

        for j in range(diffnormalized_len):
            diffnormalized_data[j, :, :, :] = (data[j + 1, :, :, :] - data[j, :, :, :]) / (
                    data[j + 1, :, :, :] + data[j, :, :, :] + 1e-7)

        diffnormalized_data = diffnormalized_data / np.std(diffnormalized_data)
        return diffnormalized_data

    def run(self):
        """Run non-real-time rPPG system and plot the result."""
        if self.use_camera:
            print("Error: Non-real-time mode does not support camera input.")
            return
        else:
            assert self.video_path is not None, "Please provide a video path."
            cap = cv2.VideoCapture(self.video_path)
            print(f"Processing video from file: {self.video_path}")

        if not cap.isOpened():
            print("Error: Could not open video source.")
            return

        # 收集所有帧的预测结果
        all_frames_processed = []
        if self.fps is None:
            self.fps = cap.get(cv2.CAP_PROP_FPS)
            print(f"Video FPS: {self.fps}")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("End of video stream or failed to read frame.")
                    break

                processed_frame = self.preprocess_frame(frame)
                self.frame_buffer.append(processed_frame)

                if len(self.frame_buffer) >= self.frame_depth:
                    chunk = np.array(self.frame_buffer[-self.frame_depth:])
                    chunk = np.squeeze(chunk, 1)
                    data = list()

                    for data_type in ['DiffNormalized', 'Standardized']:
                        f_c = chunk.copy()
                        if data_type == "Raw":
                            data.append(f_c)
                        elif data_type == "DiffNormalized":
                            data.append(UBFCrPPGDataset.diff_normalize_data(f_c))
                        elif data_type == "Standardized":
                            data.append(UBFCrPPGDataset.standardized_data(f_c))
                        else:
                            raise ValueError("Unsupported data type!")
                    data = np.concatenate(data, axis=-1)

                    input_tensor = torch.from_numpy(data).float().unsqueeze(0)
                    input_tensor = input_tensor.permute(0, 1, 4, 2, 3)
                    N, D, C, H, W = input_tensor.shape
                    input_tensor = input_tensor.view(N * D, C, H, W)

                    with torch.no_grad():
                        output = self.model(input_tensor)

                    pred_diff = output[-1].cpu().numpy()[0]
                    all_frames_processed.append(pred_diff)

        finally:
            cap.release()

        # Plot the rPPG signal using matplotlib
        self.plot_rppg_signal(all_frames_processed)

    def plot_rppg_signal(self, rppg_history):
        """Plot the rPPG signal using matplotlib."""
        rppg_array = np.array(rppg_history)
        normalized_signal = (rppg_array - np.min(rppg_array)) / (np.max(rppg_array) - np.min(rppg_array) + 1e-7)
        # 应用带通滤波器
        normalized_signal = self.butter_bandpass(normalized_signal, 0.6, 4, self.fps, order=2)
        # 时间轴
        time = np.arange(len(normalized_signal)) / self.fps
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.plot(time, normalized_signal, color='green')
        plt.title('rPPG Signal')
        plt.xlabel('Time (s)')
        plt.ylabel('Normalized Amplitude')
        plt.grid(True)
        plt.tight_layout()
        # 绘制频谱图
        plt.subplot(2, 1, 2)
        # 加汉宁窗
        window = np.hanning(len(normalized_signal))
        normalized_signal = normalized_signal * window
        fft_result = np.fft.fft(normalized_signal)
        fft_magnitude = np.abs(fft_result)
        fft_freq = np.fft.fftfreq(len(normalized_signal), d=1 / self.fps)
        # 只绘制正频率部分
        positive_freq_indices = fft_freq >= 0
        fft_freq = fft_freq[positive_freq_indices]
        # 将频率转换为BPM
        fft_freq = fft_freq * 60
        fft_magnitude = fft_magnitude[positive_freq_indices]
        # 打印峰值频率对应的BPM
        peak_index = np.argmax(fft_magnitude)
        peak_frequency = fft_freq[peak_index]
        peak_bpm = peak_frequency
        print(f"Peak BPM: {peak_bpm:.2f}")
        plt.plot(fft_freq, fft_magnitude, color='blue')
        plt.title('PSD')
        plt.xlabel('Heart Rate (bpm)')
        plt.ylabel('Magnitude')
        plt.xlim([40,200])  # 限制x轴范围
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def butter_bandpass(sig, lowcut, highcut, fs, order=2):
        # butterworth bandpass filter
        sig = np.reshape(sig, -1)
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        y = filtfilt(b, a, sig)
        return y


if __name__ == "__main__":
    rppg_system = RealTimeRPPG(model_path="./run/run_6/tscan_best.pth",
                               use_camera=False,
                               video_path=r"E:\rppg_dataset\PURE\视频数据\1-1.mp4",
                               use_face_detection=False)
    rppg_system.run()

