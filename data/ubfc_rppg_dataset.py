import logging
import os
import glob
import re
import numpy as np
import cv2
from torch.utils.data import Dataset
import pandas as pd
from ultralytics import YOLO


class UBFCrPPGDataset(Dataset):
    """Simplified data loader for the UBFC-rPPG dataset."""

    def __init__(self, data_path, cached_path, file_list_path, split_ratio=(0.0, 1.0), chunk_length=128,
                 preprocess=True, YOLOv8_model_path="D:\\文档\\python项目\\models\\best_face.pt",
                 re_size=36, crop_face=True,
                 larger_box_coef=1.5, backend="HC", use_face_detection=True,
                 label_type="DiffNormalized"):
        """
        Args:
            data_path (str): Path to raw dataset folder.
            cached_path (str): Path to save preprocessed .npy files.
            file_list_path (str): Path to save/load file list CSV.
            split_ratio (tuple): (begin, end) ratio of subjects to use.
            chunk_length (int): Length of each video/label chunk.
        """
        self.data_path = data_path
        self.cached_path = cached_path
        self.file_list_path = file_list_path
        self.split_ratio = split_ratio
        self.chunk_length = chunk_length
        self.inputs = []
        self.labels = []
        self.re_size = re_size
        self.larger_box_coef = larger_box_coef
        self.backend = backend
        self.use_face_detection = use_face_detection
        self.label_type = label_type
        self.crop_face = crop_face
        # 加载预训练的 YOLOv8 模型
        self.YOLOv8_model_path = YOLOv8_model_path
        self.yolo_model = None  # 不立即加载

        # Ensure directories exist
        os.makedirs(self.cached_path, exist_ok=True)
        os.makedirs(os.path.dirname(self.file_list_path), exist_ok=True)

        # Preprocess or load from cache
        if preprocess:
            self._preprocess()
        else:
            self._load_file_list()

    def _init_YOLO(self):
        """延迟初始化 YOLOv8 模型"""
        if self.yolo_model is None:
            self.yolo_model = YOLO(self.YOLOv8_model_path)
            # 设置日志级别为 WARNING，以抑制 INFO 级别的输出
            logging.getLogger('ultralytics').setLevel(logging.WARNING)

    def __getstate__(self):
        # 在进程拷贝时排除 yolo_model
        state = self.__dict__.copy()
        if 'yolo_model' in state:
            del state['yolo_model']
        return state

    def __setstate__(self, state):
        # 恢复对象状态，并初始化 yolo_model 为 None
        self.__dict__.update(state)
        self.yolo_model = None  # 子进程不会尝试复制原来的模型

    def _get_raw_data(self):
        """Get all subject directories."""
        data_dirs = glob.glob(os.path.join(self.data_path, "subject*"))
        dirs = [{"index": re.search('subject\\d+', d).group(0), "path": d} for d in data_dirs]
        return dirs

    def _split_data(self, data_dirs):
        """Split data by subject according to split_ratio."""
        begin, end = self.split_ratio
        n = len(data_dirs)
        start_idx = int(begin * n)
        end_idx = int(end * n)
        return data_dirs[start_idx:end_idx]

    @staticmethod
    def read_video(video_file):
        """Reads a video file, returns frames(T, H, W, 3) """
        VidObj = cv2.VideoCapture(video_file)
        VidObj.set(cv2.CAP_PROP_POS_MSEC, 0)
        success, frame = VidObj.read()
        frames = list()
        while success:
            frame = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB)
            frame = np.asarray(frame)
            frames.append(frame)
            success, frame = VidObj.read()
        return np.asarray(frames)

    @staticmethod
    def read_wave(bvp_file):
        """Reads a bvp signal file."""
        with open(bvp_file, "r") as f:
            str1 = f.read()
            str1 = str1.split("\n")
            bvp = [float(x) for x in str1[0].split()]
        return np.asarray(bvp)

    def _save_clips(self, frame_clips, label_clips, filename):
        """Save clips to .npy files."""
        input_paths = []
        label_paths = []
        print(f"Saving {len(frame_clips)} clips for {filename} ...")
        for i in range(len(frame_clips)):
            input_path = os.path.join(self.cached_path, f"{filename}_input{i}.npy")
            label_path = os.path.join(self.cached_path, f"{filename}_label{i}.npy")

            np.save(input_path, frame_clips[i])
            np.save(label_path, label_clips[i])

            input_paths.append(input_path)
            label_paths.append(label_path)
        return input_paths, label_paths

    def face_detection(self, frame, backend, use_larger_box=False, larger_box_coef=1.0):
        """Face detection on a single frame.

        Args:
            frame(np.array): a single frame.
            backend(str): backend to utilize for face detection.
            use_larger_box(bool): whether to use a larger bounding box on face detection.
            larger_box_coef(float): Coef. of larger box.
        Returns:
            face_box_coor(List[int]): coordinates of face bouding box.
        """
        if backend == "HC":
            # Use OpenCV's Haar Cascade algorithm implementation for face detection
            # This should only utilize the CPU
            detector = cv2.CascadeClassifier(
                './data/haarcascade_frontalface_default.xml')

            # Computed face_zone(s) are in the form [x_coord, y_coord, width, height]
            # (x,y) corresponds to the top-left corner of the zone to define using
            # the computed width and height.
            face_zone = detector.detectMultiScale(frame[:, :, :3].astype(np.uint8))

            if len(face_zone) < 1:
                print("ERROR: No Face Detected")
                face_box_coor = [0, 0, frame.shape[0], frame.shape[1]]
            elif len(face_zone) >= 2:
                # Find the index of the largest face zone
                # The face zones are boxes, so the width and height are the same
                max_width_index = np.argmax(face_zone[:, 2])  # Index of maximum width
                face_box_coor = face_zone[max_width_index]
                print("Warning: More than one faces are detected. Only cropping the biggest one.")
            else:
                face_box_coor = face_zone[0]
        elif "YOLO" in backend:
            # Use a YOLO trained on WiderFace dataset
            # This utilizes both the CPU and GPU
            self._init_YOLO()
            results = self.yolo_model(frame[:, :, :3].astype(np.uint8))
            best_box = None
            max_area = 0
            for box in results[0].boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = map(int, box)
                area = (x2 - x1) * (y2 - y1)
                if area > max_area:
                    max_area = area
                    best_box = (x1, y1, x2, y2)
            if best_box != None:
                x_min, y_min, x_max, y_max = best_box
                # Convert to this toolbox's expected format
                # Expected format: [x_coord, y_coord, width, height]
                x = x_min
                y = y_min
                width = x_max - x_min
                height = y_max - y_min

                # Find the center of the face zone
                center_x = x + width // 2
                center_y = y + height // 2

                # Determine the size of the square (use the maximum of width and height)
                square_size = max(width, height)

                # Calculate the new coordinates for a square face zone
                new_x = center_x - (square_size // 2)
                new_y = center_y - (square_size // 2)
                face_box_coor = [new_x, new_y, square_size, square_size]

            else:
                print("ERROR: No Face Detected")
                face_box_coor = [0, 0, frame.shape[0], frame.shape[1]]
        else:
            raise ValueError("Unsupported face detection backend!")

        if use_larger_box:
            face_box_coor[0] = max(0, face_box_coor[0] - (larger_box_coef - 1.0) / 2 * face_box_coor[2])
            face_box_coor[1] = max(0, face_box_coor[1] - (larger_box_coef - 1.0) / 2 * face_box_coor[3])
            face_box_coor[2] = larger_box_coef * face_box_coor[2]
            face_box_coor[3] = larger_box_coef * face_box_coor[3]
        return face_box_coor

    def crop_face_resize(self, frames, backend, use_larger_box, larger_box_coef, width, height,
                         use_face_detection=True):
        """Crop face and resize frames.

        Args:
            frames(np.array): Video frames.
            width(int): Target width for resizing.
            height(int): Target height for resizing.
            use_larger_box(bool): Whether enlarge the detected bouding box from face detection.
            larger_box_coef(float): the coefficient of the larger region(height and weight),
                                the middle point of the detected region will stay still during the process of enlarging.
            use_face_detection(bool): Whether to use face detection to crop the face region.
        Returns:
            resized_frames(list[np.array(float)]): Resized and cropped frames
        """
        total_frames, _, _, channels = frames.shape
        if self.crop_face:
            # Face Cropping
            face_region_all = []
            if use_face_detection:
                # 对每个帧进行人脸检测
                for i in range(0, total_frames):
                    frame = frames[i]
                    face_region_all.append(self.face_detection(frame, backend, use_larger_box, larger_box_coef))
            else:
                # 只对第一个帧进行人脸检测
                face_region_all.append(self.face_detection(frames[0], backend, use_larger_box, larger_box_coef))
            face_region_all = np.asarray(face_region_all, dtype='int')

        # Frame Resizing
        resized_frames = np.zeros((total_frames, height, width, channels))
        for i in range(0, total_frames):
            frame = frames[i]
            if self.crop_face:
                if use_face_detection:
                    assert len(face_region_all) == total_frames, "Face region detection failed!"
                    # 获得当前帧的人脸区域
                    reference_index = i
                else:
                    # use the first region obtrained from the first frame.
                    reference_index = 0
                face_region = face_region_all[reference_index]
                frame = frame[max(face_region[1], 0):min(face_region[1] + face_region[3], frame.shape[0]),
                        max(face_region[0], 0):min(face_region[0] + face_region[2], frame.shape[1])]

            resized_frames[i] = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        return resized_frames

    def chunk(self, frames, bvps, chunk_length):
        """Chunk the data into small chunks.

        Args:
            frames(np.array): video frames.
            bvps(np.array): blood volumne pulse (PPG) labels.
            chunk_length(int): the length of each chunk.
        Returns:
            frames_clips: all chunks of face cropped frames
            bvp_clips: all chunks of bvp frames
        """

        clip_num = frames.shape[0] // chunk_length
        frames_clips = [frames[i * chunk_length:(i + 1) * chunk_length] for i in range(clip_num)]
        bvps_clips = [bvps[i * chunk_length:(i + 1) * chunk_length] for i in range(clip_num)]
        return np.array(frames_clips), np.array(bvps_clips)

    def preprocess_data_and_bvps(self, frames, bvps):
        """Preprocesses a pair of data.

        Args:
            frames(np.array): Frames in a video.
            bvps(np.array): Blood volumne pulse (PPG) signal labels for a video.
        Returns:
            frame_clips(np.array): processed video data by frames
            bvps_clips(np.array): processed bvp (ppg) labels by frames
        """
        # resize frames and crop for face region
        frames = self.crop_face_resize(
            frames,
            backend=self.backend,
            use_larger_box=True,
            larger_box_coef=self.larger_box_coef,
            width=self.re_size,
            height=self.re_size,
            use_face_detection=self.use_face_detection)
        # Check data transformation type
        data = list()  # Video data
        for data_type in ['DiffNormalized', 'Standardized']:
            f_c = frames.copy()
            if data_type == "Raw":
                data.append(f_c)
            elif data_type == "DiffNormalized":
                data.append(UBFCrPPGDataset.diff_normalize_data(f_c))
            elif data_type == "Standardized":
                data.append(UBFCrPPGDataset.standardized_data(f_c))
            else:
                raise ValueError("Unsupported data type!")
        data = np.concatenate(data, axis=-1)  # concatenate all channels
        if self.label_type == "DiffNormalized":
            # 差分归一化标签
            bvps = UBFCrPPGDataset.diff_normalize_label(bvps)
        elif self.label_type == "Standardized":
            # 标准化标签
            bvps = UBFCrPPGDataset.standardized_label(bvps)
        else:
            raise ValueError("Unsupported label type!")
        # chunk data
        frames_clips, bvps_clips = self.chunk(data, bvps, self.chunk_length)
        return frames_clips, bvps_clips

    def _preprocess(self):
        """Main preprocessing function."""
        print("Preprocessing dataset...")

        # Step 1: Get and split data
        all_subjects = self._get_raw_data()
        subjects_subset = self._split_data(all_subjects)

        # Step 2: Process each subject
        input_files = []
        label_files = []

        for subject in subjects_subset:
            print(f"Processing subject: {subject['index']}")
            subject_id = subject["index"]
            subject_path = subject["path"]

            # Read video and label
            video_path = os.path.join(subject_path, "vid.avi")
            label_path = os.path.join(subject_path, "ground_truth.txt")

            frames = UBFCrPPGDataset.read_video(video_path)
            bvps = UBFCrPPGDataset.read_wave(label_path)

            # Chunk and save
            frame_clips, label_clips = self.preprocess_data_and_bvps(frames, bvps)
            input_paths, label_paths = self._save_clips(frame_clips, label_clips, subject_id)

            input_files.extend(input_paths)
            label_files.extend(label_paths)

        # Step 3: Save file list
        df = pd.DataFrame({"input_files": input_files})
        df.to_csv(self.file_list_path, index=False)

        # Step 4: Load paths into memory
        self.inputs = sorted(input_files)
        self.labels = sorted(label_files)
        print(f"Preprocessing done. Total clips: {len(self.inputs)}")

    def _load_file_list(self):
        """Load existing file list from CSV."""
        import pandas as pd
        df = pd.read_csv(self.file_list_path)
        inputs = df['input_files'].tolist()
        self.inputs = sorted(inputs)
        labels = [f.replace("_input", "_label") for f in self.inputs]
        self.labels = sorted(labels)
        print(f"Loaded {len(self.inputs)} clips from cache.")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        """Load a clip from .npy file."""
        video = np.load(self.inputs[idx])  # shape: (T, H, W, 6)
        label = np.load(self.labels[idx])  # shape: (T, )

        # 转换为N, D, C, H, W
        video = np.transpose(video, (0, 3, 1, 2))  # shape: (T, 6, H, W)

        # Convert to float32
        video = np.float32(video)
        label = np.float32(label)

        # Shape: (T, 6, H, W) -> ToTensor-like normalization
        return video, label

    @staticmethod
    def diff_normalize_data(data):
        """Calculate discrete difference in video data along the time-axis and nornamize by its standard deviation."""
        n, h, w, c = data.shape
        diffnormalized_len = n - 1
        diffnormalized_data = np.zeros((diffnormalized_len, h, w, c), dtype=np.float32)
        diffnormalized_data_padding = np.zeros((1, h, w, c), dtype=np.float32)
        for j in range(diffnormalized_len):
            diffnormalized_data[j, :, :, :] = (data[j + 1, :, :, :] - data[j, :, :, :]) / (
                    data[j + 1, :, :, :] + data[j, :, :, :] + 1e-7)
        diffnormalized_data = diffnormalized_data / np.std(diffnormalized_data)
        diffnormalized_data = np.append(diffnormalized_data, diffnormalized_data_padding, axis=0)
        diffnormalized_data[np.isnan(diffnormalized_data)] = 0
        return diffnormalized_data

    @staticmethod
    def diff_normalize_label(label):
        """Calculate discrete difference in labels along the time-axis and normalize by its standard deviation."""
        diff_label = np.diff(label, axis=0)
        diffnormalized_label = diff_label / np.std(diff_label)
        diffnormalized_label = np.append(diffnormalized_label, np.zeros(1), axis=0)
        diffnormalized_label[np.isnan(diffnormalized_label)] = 0
        return diffnormalized_label

    @staticmethod
    def standardized_data(data):
        """Z-score standardization for video data."""
        data = data - np.mean(data)
        data = data / np.std(data)
        data[np.isnan(data)] = 0
        return data

    @staticmethod
    def standardized_label(label):
        """Z-score standardization for label signal."""
        label = label - np.mean(label)
        label = label / np.std(label)
        label[np.isnan(label)] = 0
        return label
