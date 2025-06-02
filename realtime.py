import logging

import cv2
import numpy as np
import torch
from model.TS_CAN import TSCAN
from ultralytics import YOLO
import time
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
        """Run real-time rPPG system."""
        if self.use_camera:
            print("Using camera...")
            cap = cv2.VideoCapture(0)
        else:
            assert self.video_path is not None, "Please provide a video path or set use_camera=True."
            cap = cv2.VideoCapture(self.video_path)
            print(f"Playing video from file: {self.video_path}")

        if not cap.isOpened():
            print("Error: Could not open video source.")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 30  # Default value

        frame_duration = 1 / fps
        window_size = int(fps * 3)  # 3 seconds of data

        print(f"FPS: {fps}")

        # 在初始化可视化之前创建一个新的窗口
        cv2.namedWindow('Real-Time rPPG', cv2.WINDOW_NORMAL)
        cv2.namedWindow('rPPG Signal', cv2.WINDOW_NORMAL)

        try:
            while True:
                start_time = time.time()

                ret, frame = cap.read()
                if not ret:
                    print("End of video stream or failed to read frame.")
                    break

                # Preprocess frame
                processed_frame = self.preprocess_frame(frame)
                self.frame_buffer.append(processed_frame)

                # When we have enough frames for one prediction
                if len(self.frame_buffer) >= self.frame_depth:
                    # Process data
                    chunk = np.array(self.frame_buffer[-self.frame_depth:])
                    chunk = np.squeeze(chunk, 1)
                    data = list()  # Video data
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
                    data = np.concatenate(data, axis=-1)  # concatenate all channels

                    # Prepare input tensor
                    input_tensor = torch.from_numpy(data).float()
                    input_tensor = input_tensor.unsqueeze(0) # Add batch dimension
                    input_tensor = input_tensor.permute(0, 1, 4, 2, 3) # Change to (B, T, C, H, W)
                    N, D, C, H, W = input_tensor.shape
                    input_tensor = input_tensor.view(N * D, C, H, W)

                    # Make prediction
                    with torch.no_grad():
                        output = self.model(input_tensor)

                    # Convert output to numpy array
                    pred_diff = output[-1].cpu().numpy()[0]

                    if self.accumulate_output:
                        if len(self.rppg_history) == 0:
                            self.rppg_history.append(pred_diff)
                        else:
                            self.rppg_history.append(self.rppg_history[-1] + pred_diff)
                    else:
                        self.rppg_history.append(pred_diff)

                    # Keep only the last 5 seconds of data
                    if len(self.rppg_history) > window_size:
                        self.rppg_history = self.rppg_history[-window_size:]

                # Create visualization
                vis = frame.copy()
                if len(self.rppg_history) > 1:
                    # Draw rPPG signal in a separate window
                    signal_height, signal_width = 200, 600  # 固定信号图大小
                    signal_img = np.zeros((signal_height, signal_width, 3), dtype=np.uint8)

                    normalized_signal = np.array(self.rppg_history)
                    normalized_signal = (normalized_signal - np.min(normalized_signal)) / \
                                        (np.max(normalized_signal) - np.min(normalized_signal) + 1e-7)

                    normalized_signal = normalized_signal[-signal_width:]

                    points = [(int(i * signal_width / len(normalized_signal)),
                               int(signal_height - sig * signal_height))
                              for i, sig in enumerate(normalized_signal)]

                    for i in range(1, len(points)):
                        cv2.line(signal_img, points[i - 1], points[i], (0, 255, 0), 2)

                    # 在信号图周围绘制边框
                    cv2.rectangle(signal_img, (0, 0), (signal_width - 1, signal_height - 1), (255, 255, 255), 2)

                    # 显示信号图像在新窗口中
                    cv2.imshow('rPPG Signal', signal_img)

                # Show video feed in its own window
                cv2.imshow('Real-Time rPPG', vis)

                # Control frame rate
                elapsed_time = time.time() - start_time
                if frame_duration - elapsed_time > 0:
                    time.sleep(frame_duration - elapsed_time)

                # Check for exit key
                if cv2.waitKey(1) & 0xFF == 27:
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    rppg_system = RealTimeRPPG(model_path="./run/run_6/tscan_best.pth",
                               use_camera=False,
                               video_path=r"E:\rppg_dataset\PURE\视频数据\1-1.mp4",
                               use_face_detection=False)
    rppg_system.run()
