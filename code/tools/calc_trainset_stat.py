# This file is covered by the LICENSE file in the root of this project.

"""
Compute Mean and Standard Deviation for LiDAR Training Dataset

This script computes the per-channel mean and standard deviation of range images
from a LiDAR dataset using a numerically stable streaming (Welford’s) method.

It expects a configuration file at `./configs/data_config.yaml` which specifies
the dataset location and other parameters. Results are saved as pickled dictionaries.
"""

import os
import glob
import pickle
import yaml

import numpy as np

from typing import List, Tuple, Dict
from pprint import pprint
from tqdm import tqdm

from utility.laserscan_unfolding import LaserScan


def get_train_set_frames(config: dict) -> List[str]:
    """
    Collect all LiDAR frame paths in the training split as defined by the config.

    Args:
        config (dict): Configuration dictionary containing dataset paths and splits.

    Returns:
        List[str]: List of full file paths to LiDAR binary files in the training set.
    """
    train_set_sequences = config['data']['kitti']['train_split']
    train_set_frames = []

    for sequence in train_set_sequences:
        sequence_dir = os.path.join(config['data']['kitti']['seq_dir'], f"{sequence:02}")
        sequence_lidar_dir = os.path.join(sequence_dir, "velodyne")
        frames_data_paths = sorted(glob.glob(os.path.join(sequence_lidar_dir, "*.bin")))
        train_set_frames.extend(frames_data_paths)

    return train_set_frames


def calc_mean_std_using_streaming_method(
    frames: List[str], lidar_loader: LaserScan
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Calculate the mean and standard deviation of the dataset using a streaming method.

    Uses Welford's algorithm to ensure numerical stability when processing large datasets.

    Args:
        frames (List[str]): List of paths to LiDAR frame files.
        lidar_loader (LaserScan): Instance of LaserScan for loading LiDAR data.

    Returns:
        Tuple[Dict[str, float], Dict[str, float]]: Two dictionaries with per-channel mean and std.
    """
    fields = ["range", "x", "y", "z", "remission"]

    n_total = {f: 0 for f in fields}
    mean = {f: 0.0 for f in fields}
    m2 = {f: 0.0 for f in fields}
    std_dev = {f: 0.0 for f in fields}

    for frame_path in tqdm(frames, desc="Computing statistics"):
        lidar_loader.open_scan(frame_path)
        num_data = lidar_loader.remissions.shape[0]
        range_data = np.linalg.norm(lidar_loader.points, axis=-1)

        for i, f in enumerate(fields):
            if f in ["x", "y", "z"]:
                values = lidar_loader.points[:, i - 1]
            elif f == "remission":
                values = lidar_loader.remissions
            elif f == "range":
                values = range_data
            else:
                continue

            delta = values - mean[f]
            mean[f] += delta.sum() / (n_total[f] + num_data)
            m2[f] += np.sum(delta * (values - mean[f]))
            n_total[f] += num_data

    for f in fields:
        variance = m2[f] / (n_total[f] - 1) if n_total[f] > 1 else 0
        std_dev[f] = np.sqrt(variance)

    return mean, std_dev


def main():
    """
    Main function to compute and save dataset mean and standard deviation.

    Mean
    {'range': 11.562711526678434,
    'remission': 0.28691127996350485,
    'x': -0.10725239569448032,
    'y': 0.5018524833806149,
    'z': -1.0599034569481658}

    Standard Deviation
    {'range': 9.997143915740443,
    'remission': 0.14306272748556606,
    'x': 12.18310786558251,
    'y': 9.113931008658172,
    'z': 0.8721091804893368}
    """
    data_config_path = os.path.join("./configs", "data_config.yaml")
    if not os.path.exists(data_config_path):
        print("WARNING! Data configuration file does not exist!")
        exit(1)

    # Load configuration
    config = yaml.load(open(data_config_path, "r"), Loader=yaml.FullLoader)

    # Collect LiDAR training frames
    train_set_frames = get_train_set_frames(config)
    lidar_loader = LaserScan(project=True)

    # Compute mean and standard deviation
    mean, std_dev = calc_mean_std_using_streaming_method(train_set_frames, lidar_loader)

    # Display results
    print("\n--- Mean ---")
    pprint(mean)    
    print("\n--- Std Dev ---")
    pprint(std_dev)

    # Save to disk
    with open(config['data']['kitti']['mean_pkl_path'], 'wb') as handle:
        pickle.dump(mean, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(config['data']['kitti']['std_pkl_path'], 'wb') as handle:
        pickle.dump(std_dev, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise
