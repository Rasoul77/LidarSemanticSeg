# This file is covered by the LICENSE file in the root of this project.

import glob
import logging
import os
from pathlib import Path
from typing import List, Tuple

import torch
import numpy as np
from torch.utils.data import Dataset

from utility.laserscan_unfolding import SemLaserScan


class SemanticKittiDataset(Dataset):
    """
    PyTorch Dataset class for SemanticKITTI LiDAR scans with optional semantic label processing.

    This dataset supports loading multiple sequences, projecting LiDAR scans to range images,
    applying semantic labels (if available), and optionally returning unprojected ground truth
    data for auxiliary tasks.

    Each returned sample includes:
        - A 2D "image" with 5 channels: [range, x, y, z, remission].
        - A 2D "mask" with semantic labels (remapped using the learning map).
        - If `include_full_gt` is True, also includes:
            - "proj_idx": Indices for reprojecting to original point cloud.
            - "flat_label": Unprojected labels with zero-padding.
            - "unproj_range": Unprojected range values.
            - "p_x", "p_y": Projected x and y pixel coordinates.
            - "num_points": Number of valid points before padding.

    Args:
        config (dict): Configuration dictionary with keys:
            - 'data.kitti.seq_dir' (str): Path to the root of SemanticKITTI sequences.
            - 'data.kitti.learning_map' (dict): Label remapping dictionary.
            - 'data.kitti.max_num_points_per_frame' (int): Padding length for unprojected data.
        sequences (List[int]): Sequence numbers to load (e.g., [0, 1, 2]).
        include_full_gt (bool): If True, returns additional unprojected GT fields.

    Attributes:
        frames (List[Tuple[str, str]]): List of (point cloud path, label path) pairs.
        sem_laser_scan (SemLaserScan): Helper class for range projection and label operations.
    """

    def __init__(
        self,
        config: dict,
        sequences: List[int],
        include_full_gt: bool = False,        
    ):
        self.sequences = sequences
        self.include_full_gt = include_full_gt
        self.config = config

        self.frames: List[Tuple[str, str]] = []
        for sequence in self.sequences:
            sequence_dir = os.path.join(config['data']['kitti']['seq_dir'], f"{sequence:02}")
            sequence_lidar_dir = os.path.join(sequence_dir, "velodyne")
            sequence_label_dir = os.path.join(sequence_dir, "labels")
            assert os.path.exists(sequence_lidar_dir)
            frames_points_paths = sorted(glob.glob(os.path.join(sequence_lidar_dir, "*.bin")))
            if os.path.exists(sequence_label_dir):
                frames_labels_paths = sorted(glob.glob(os.path.join(sequence_label_dir, "*.label")))
            else:
                frames_labels_paths = [None] * len(frames_points_paths)
            for point_cloud_path, label_path in zip(frames_points_paths, frames_labels_paths):
                frame_number = Path(point_cloud_path).stem
                self.frames.append({
                    "point_cloud_path": point_cloud_path,
                    "label_path": label_path,
                    "sequence": sequence,
                    "frame_number": frame_number,
                })

        self.sem_laser_scan = SemLaserScan(project=False)
        
        logging.info(f'Initialized the Semantic Kitti dataset with {len(self.frames)} samples.')
    
    def __len__(self):
        """
        Returns the total number of frames in the dataset.
        """
        
        return len(self.frames)
    
    def __getitem__(self, idx):
        """
        Loads and processes the LiDAR scan and (optionally) semantic labels at index `idx`.

        Args:
            idx (int): Index of the data sample.

        Returns:
            dict: Dictionary containing:
                - "image": [H, W, 5] tensor with range, xyz, and remission channels
                - "mask": [H, W] tensor of projected semantic labels
                - Optional unprojected ground truth fields if `include_full_gt` is True:
                    - "proj_idx", "flat_label", "unproj_range", "p_x", "p_y", "num_points"
        """

        output = {}

        # Extract the idx-th frame's paths
        point_cloud_path = self.frames[idx]["point_cloud_path"]
        label_path = self.frames[idx]["label_path"]

        # Load point cloud and create range image
        self.sem_laser_scan.open_scan(point_cloud_path)        
        self.sem_laser_scan.do_range_projection()
        
        # We treat -1.0 values as don't cares
        dc_idx = np.where(self.sem_laser_scan.proj_range < 0.0)        
        
        self.apply_dont_care_range_image(dc_idx)        
        self.apply_dont_care_remission_image(dc_idx)
        
        if label_path:
            self.sem_laser_scan.open_label(label_path)
            self.sem_laser_scan.do_label_projection()

            self.apply_learning_map()

            self.apply_dont_care_labels(dc_idx)

        if self.include_full_gt:
            number_of_points = self.sem_laser_scan.points.shape[0]
            if number_of_points > self.config['data']['kitti']['max_num_points_per_frame']:
                number_of_points = self.config['data']['kitti']['max_num_points_per_frame']

            output["proj_idx"] = torch.from_numpy(self.sem_laser_scan.proj_idx).contiguous()            
            output["unproj_range"] = self.get_padded_unproj_data(data=self.sem_laser_scan.unproj_range, num_points=number_of_points)
            output["p_x"] = self.get_padded_unproj_data(data=self.sem_laser_scan.proj_x, num_points=number_of_points)
            output["p_y"] = self.get_padded_unproj_data(data=self.sem_laser_scan.proj_y, num_points=number_of_points)
            output["num_points"] = number_of_points
            output["sequence"] = self.frames[idx]["sequence"]
            output["frame_number"] = self.frames[idx]["frame_number"]

            if label_path:
                output["flat_label"] = self.get_padded_unproj_data(data=self.sem_laser_scan.sem_label, num_points=number_of_points)
                
        image = np.concatenate(
            [
                self.sem_laser_scan.proj_range[..., None],
                self.sem_laser_scan.proj_xyz,
                self.sem_laser_scan.proj_remission[..., None],
            ],
            axis=-1,
        ).astype(np.float32)        
        
        # Create mask
        mask = self.sem_laser_scan.proj_sem_label

        # Add image and mask to ouput dict
        output["image"] = torch.from_numpy(image).contiguous()
        output["mask"] = torch.from_numpy(mask).contiguous()
        
        return output        
        
    def apply_learning_map(self):
        """
        Remaps labels using the class-defined learning map from the KITTI config.
        This reduces label classes to a smaller set used for training.
        """
        
        remapdict = self.config['data']['kitti']['learning_map']
        maxkey = max(remapdict.keys())  # 259
        remap_lut = np.zeros((maxkey + 100), dtype=np.int32)
        remap_lut[list(remapdict.keys())] = list(remapdict.values())
        self.sem_laser_scan.proj_sem_label = remap_lut[self.sem_laser_scan.proj_sem_label]
        self.sem_laser_scan.sem_label = remap_lut[self.sem_laser_scan.sem_label]

    def apply_dont_care_labels(self, dc_idx):
        """
        Sets semantic labels at invalid range locations to -100, which is ignored in loss computation.

        Args:
            dc_idx (np.ndarray): Indices of invalid pixels in the projected image.
        """
        
        self.sem_laser_scan.proj_sem_label[dc_idx] = -100

    def apply_dont_care_range_image(self, dc_idx):
        """
        Sets semantic labels at invalid range locations to -100, which is ignored in loss computation.

        Args:
            dc_idx (np.ndarray): Indices of invalid pixels in the projected image.
        """

        self.sem_laser_scan.proj_range[dc_idx] = 0.0

    def apply_dont_care_remission_image(self, dc_idx):
        """
        Sets remission values at invalid range locations to 0.

        Args:
            dc_idx (np.ndarray): Indices of invalid pixels in the projected image.
        """

        self.sem_laser_scan.proj_remission[dc_idx] = 0.0
            
    def get_padded_unproj_data(self, data: np.ndarray, num_points: int):
        """
        Pads unprojected data to a fixed maximum size for batching.

        Args:
            data (np.ndarray): 1D array of length <= MAX_NUM_POINTS_PER_FRAME.
            num_points (int): Number of valid points to copy into the padded array.

        Returns:
            torch.Tensor: A fixed-length tensor with padded zeros.
        """

        temp = np.zeros((self.config['data']['kitti']['max_num_points_per_frame'],), dtype=data.dtype)
        temp[0:num_points] = data
        return torch.from_numpy(temp).contiguous()

