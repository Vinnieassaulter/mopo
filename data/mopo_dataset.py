import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import numpy as np

class MarkerDataset(Dataset):
    def __init__(self,
                marker_path,
                noise_path,
                # pose_path,
                n_max_markers,
                n_noised_frames,
                n_max_noised_markers,
                window_size=50,
                ):
        super().__init__()

        self.window_size = window_size

        # marker_data
        self.marker_data = torch.load(marker_path)
        self.n_max_markers = n_max_markers
        self.marker_data = self.marker_data.reshape(-1, 1, self.n_max_markers, 3)
        # unzip
        self.marker_data = self.marker_data[:, 0, :, :]

        # noise_data
        self.noise_data = np.load(noise_path)
        self.noise_data = torch.from_numpy(self.noise_data['amass_marker_noise_model'])
        self.noise_data = self.noise_data[:, 0, :, :]

        # marker加噪声
        # 确定要替换的帧和marker数量
        num_frames_to_replace = n_noised_frames  # 替换的帧数
        num_markers_to_replace = n_max_noised_markers  # 每帧替换的marker数

        # 随机选择要替换的帧索引
        marker_frame_indices = np.random.choice(self.marker_data.shape[0], num_frames_to_replace, replace=False)
        noise_frame_indices = np.random.choice(self.noise_data.shape[0], num_frames_to_replace, replace=False)

        self.noised_data = self.marker_data
        # 替换marker_data中的部分marker坐标
        for i in range(num_frames_to_replace):
            # 随机选择要替换的marker索引
            marker_indices = np.random.choice(self.marker_data.shape[1], num_markers_to_replace, replace=False)
            for marker_idx in marker_indices:
                self.noised_data[marker_frame_indices[i], marker_idx, :] = self.noise_data[noise_frame_indices[i], marker_idx, :]

        # # pose_data
        # self.pose_data = torch.load(pose_path)
        # self.pose_data = self.pose_data.reshape(-1, 1, 63 // 3, 3)
        # self.pose_data = self.pose_data[:, 0, :, :]

    def __len__(self):
        return self.marker_data.shape[0]

    def __getitem__(self, idx):
        # 获取长为window_size的marker和pose
        start_idx = idx
        end_idx = start_idx + self.window_size
        if end_idx > len(self.marker_data):
            end_idx = len(self.marker_data)
            start_idx = end_idx - self.window_size
        # marker_data = self.marker_data[start_idx:end_idx]
        # pose_data = self.pose_data[start_idx:end_idx]       
        noised_seq = self.noised_data[start_idx:end_idx]    # (50, 53, 3)
        return noised_seq
