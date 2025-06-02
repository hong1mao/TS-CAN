import json
import logging
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d
from model.TS_CAN import TSCAN
from ultralytics import YOLO
from data.ubfc_rppg_dataset import UBFCrPPGDataset


class RealTimeRPPG:
    def __init__(self, model_path, yolo_model_path="D:\\文档\\python项目\\models\\best_face.pt", re_size=72,
                 larger_box_coef=1.5, video_path=None, accumulate_output=False, use_camera=False, use_face_detection=True):
        self.re_size = re_size
        self.larger_box_coef = larger_box_coef
        self.frame_depth = 10
        self.video_path = video_path
        self.use_camera = use_camera
        self.use_face_detection = use_face_detection
        self.model = TSCAN(frame_depth=self.frame_depth, img_size=self.re_size)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.yolo_model = YOLO(yolo_model_path)
        logging.getLogger('ultralytics').setLevel(logging.WARNING)
        self.frame_buffer = []
        self.rppg_history = []
        self.accumulate_output = accumulate_output
        self.fps = None

    def face_detection(self, frame):
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
        center_x = x_min + width // 2
        center_y = y_min + height // 2
        square_size = int(max(width, height) * self.larger_box_coef)
        new_x = max(0, center_x - square_size // 2)
        new_y = max(0, center_y - square_size // 2)
        return [new_x, new_y, square_size, square_size]

    def crop_and_resize(self, frame, face_box):
        x, y, w, h = face_box
        face_region = frame[y:y + h, x:x + w]
        resized_frame = np.zeros((1, self.re_size, self.re_size, 3))
        resized_frame[0] = cv2.resize(face_region, (self.re_size, self.re_size), interpolation=cv2.INTER_AREA)
        return resized_frame

    def preprocess_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if self.use_face_detection:
            face_box = self.face_detection(frame)
            resized_frame = self.crop_and_resize(frame, face_box)
        else:
            resized_frame = np.zeros((1, self.re_size, self.re_size, 3))
            resized_frame[0] = cv2.resize(frame, (self.re_size, self.re_size), interpolation=cv2.INTER_AREA)
        return resized_frame

    def diff_normalize_data(self, data):
        n, h, w, c = data.shape
        diffnormalized_len = n - 1
        diffnormalized_data = np.zeros((diffnormalized_len, h, w, c), dtype=np.float32)
        for j in range(diffnormalized_len):
            diffnormalized_data[j, :, :, :] = (data[j + 1, :, :, :] - data[j, :, :, :]) / (
                    data[j + 1, :, :, :] + data[j, :, :, :] + 1e-7)
        diffnormalized_data = diffnormalized_data / np.std(diffnormalized_data)
        return diffnormalized_data

    def run(self):
        if self.use_camera:
            print("Error: Non-real-time mode does not support camera input.")
            return
        assert self.video_path is not None, "Please provide a video path."
        cap = cv2.VideoCapture(self.video_path)
        print(f"Processing video from file: {self.video_path}")
        if not cap.isOpened():
            print("Error: Could not open video source.")
            return
        if self.fps is None:
            self.fps = cap.get(cv2.CAP_PROP_FPS)
            print(f"Video FPS: {self.fps}")

        all_frames_processed = []

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

        rppg_array = np.array(all_frames_processed)
        normalized_signal = (rppg_array - np.min(rppg_array)) / (np.max(rppg_array) - np.min(rppg_array) + 1e-7)
        normalized_signal = self.butter_bandpass(normalized_signal, 0.6, 4, self.fps, order=2)
        time_axis = np.arange(len(normalized_signal)) / self.fps
        return time_axis, normalized_signal

    @staticmethod
    def butter_bandpass(sig, lowcut, highcut, fs, order=2):
        sig = np.reshape(sig, -1)
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        y = filtfilt(b, a, sig)
        return y


# 归一化函数
def normalize_signal(signal):
    return (signal - np.min(signal)) / (np.max(signal) - np.min(signal) + 1e-8)


def resample_signal(x_old, y_old, x_new):
    """使用线性插值将信号从旧时间轴插值到新时间轴"""
    f = interp1d(x_old, y_old, kind='linear', fill_value="extrapolate")
    return f(x_new)


def plot_ppg_comparison(video_path, json_path, model_path):
    # 获取模型预测信号
    rppg_system = RealTimeRPPG(model_path=model_path, video_path=video_path, use_face_detection=False)
    pred_time, pred_signal = rppg_system.run()

    # 获取真实信号
    with open(json_path, 'r') as f:
        data = json.load(f)
    full_package = data.get('/FullPackage', [])
    timestamps = np.array([item['Timestamp'] for item in full_package], dtype=np.int64)
    ppg_values = np.array([item['Value']['waveform'] for item in full_package], dtype=np.float32)

    start_time = timestamps[0]
    ppg_time = (timestamps - start_time) / 1e9  # 纳秒 -> 秒
    sampling_rate = 1 / np.mean(np.diff(ppg_time))

    # 滤波和归一化
    ppg_filtered = RealTimeRPPG.butter_bandpass(ppg_values, 0.6, 4, sampling_rate, order=2)
    ppg_normalized = normalize_signal(ppg_filtered)

    # 只保留前 10 秒的信号
    if len(ppg_time) > 10 * sampling_rate:
        ppg_time = ppg_time[:int(10 * sampling_rate) - 10]
        ppg_normalized = ppg_normalized[:int(10 * sampling_rate) - 10]

    # 将预测信号插值到 PPG 时间轴上
    pred_signal_resampled = resample_signal(pred_time, pred_signal, ppg_time)
    pred_normalized = normalize_signal(pred_signal_resampled)

    # 绘图
    plt.figure(figsize=(12, 6))
    plt.plot(ppg_time, ppg_normalized, label='Ground Truth PPG', color='blue', alpha=0.8)
    plt.plot(ppg_time, pred_normalized, label='Predicted rPPG', color='red', alpha=0.8)
    plt.title('Comparison of Ground Truth and Predicted rPPG Signal (Aligned)')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Normalized Amplitude')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    model_path = "../run/run_6/tscan_best.pth"
    video_path = r"E:\rppg_dataset\PURE\视频数据\1-1.mp4"
    json_path = r"E:\rppg_dataset\PURE\心率数据\01-01.json"
    plot_ppg_comparison(video_path, json_path, model_path)
