# This file is covered by the LICENSE file in the root of this project.

"""
This script extracts instance-level object crops from the SemanticKITTI dataset for training object-centric models.

For each frame in the training split:
- It loads point clouds and corresponding labels.
- Projects 3D points into 2D range images.
- Extracts instance-wise bounding boxes for specified "thing" classes.
- Filters bounding boxes based on area constraints.
- Saves each instance crop (with associated mask and point features) into a per-class object database.

Output:
- Cropped object instances saved as .npy files under ./objects_db/{class_name}/.
"""

import glob
import os
import yaml
from pathlib import Path

import numpy as np
import cv2
from tqdm import tqdm

from utility.laserscan_unfolding import SemLaserScan


# Create a color lookup table for 20 semantic classes (for visualization)
COLOR_TABLE = (np.array([
    [31, 119, 180], [255, 127, 14], [44, 160, 44], [214, 39, 40], [148, 103, 189],
    [140, 86, 75], [227, 119, 194], [127, 127, 127], [188, 189, 34], [23, 190, 207],
    [174, 199, 232], [255, 187, 120], [152, 223, 138], [255, 152, 150], [197, 176, 213],
    [196, 156, 148], [247, 182, 210], [199, 199, 199], [219, 219, 141], [158, 218, 229]
])).astype(np.uint8)


def extract_bounding_boxes_with_visualization(
    sem_label: np.ndarray,
    inst_label: np.ndarray,
    image: np.ndarray,    
    rare_classes: dict,
    save_path: str = None,
    min_area: int = 15 * 15,
    max_area: int = 500 * 64,
):
    """
    Extracts instance-level bounding boxes from semantic and instance labels,
    applies filtering based on area constraints, and optionally visualizes and saves them.

    Args:
        sem_label (np.ndarray): Projected semantic label image.
        inst_label (np.ndarray): Projected instance label image.
        image (np.ndarray): Concatenated feature image (range + xyz + remission).
        rare_classes (dict): Rare classes IDs/names to include.
        save_path (str, optional): Path to save visualization image.
        min_area (int): Minimum bounding box area to include.
        max_area (int): Maximum bounding box area to include.

    Returns:
        dict: Dictionary mapping class names to list of (bbox_coords, crop_image) tuples.
    """
    frame_info = {}
    sem_rgb = COLOR_TABLE[sem_label % 20]
    overlay = sem_rgb.copy()
    to_include = list(rare_classes.keys())

    for sem_idx in to_include:
        sem_mask = (sem_label == sem_idx)
        inst_ids = np.unique(inst_label[sem_mask])
        bboxes = []

        for inst_id in inst_ids:
            instance_mask = (sem_label == sem_idx) & (inst_label == inst_id)
            if not np.any(instance_mask):
                continue

            y_coords, x_coords = np.where(instance_mask)
            x0, y0 = np.min(x_coords), np.min(y_coords)
            x1, y1 = np.max(x_coords), np.max(y_coords)

            if x0 == x1 or y0 > y1 - 2 or x1 - x0 > 2048 // 4:
                continue

            area = (x1 - x0 + 1) * (y1 - y0 + 1)
            if area < min_area or area > max_area:
                continue

            instance_mask_crop = instance_mask[y0:y1, x0:x1]
            crop_image = np.concatenate(
                [image[y0:y1, x0:x1, :], instance_mask_crop[..., None]],
                axis=-1
            ).astype(np.float32)

            bboxes.append(([x0, y0, x1, y1], crop_image))

            if save_path:
                box_color = COLOR_TABLE[sem_idx % 20].tolist()
                cv2.rectangle(overlay, (x0, y0), (x1, y1), box_color, thickness=-1)
                cv2.rectangle(sem_rgb, (x0, y0), (x1, y1), (0, 0, 0), thickness=1)

        if bboxes:
            frame_info[rare_classes[sem_idx]] = bboxes

    if save_path:
        alpha = 0.3
        sem_rgb = cv2.addWeighted(overlay, alpha, sem_rgb, 1 - alpha, 0)
        cv2.imwrite(save_path, cv2.cvtColor(sem_rgb, cv2.COLOR_RGB2BGR))

    return frame_info


def apply_learning_map(sem_laser_scan: SemLaserScan, learning_map: dict):
    """
    Applies the SemanticKITTI learning map to remap original class IDs.

    Args:
        sem_laser_scan (SemLaserScan): Laser scan object with semantic labels.
    """

    maxkey = max(learning_map.keys())
    remap_lut = np.zeros((maxkey + 100), dtype=np.int32)
    remap_lut[list(learning_map.keys())] = list(learning_map.values())
    sem_laser_scan.proj_sem_label = remap_lut[sem_laser_scan.proj_sem_label]
    sem_laser_scan.sem_label = remap_lut[sem_laser_scan.sem_label]


def main():
    """
    Main execution function that iterates over the training set sequences,
    extracts valid object instances, and saves them as NumPy arrays.
    """

    # Load configuration
    data_config_path = os.path.join("./configs", "data_config.yaml")
    if not os.path.exists(data_config_path):
        print("WARNING! Data configuration file does not exist!")
        exit(1)    
    config = yaml.load(open(data_config_path, "r"), Loader=yaml.FullLoader)

    sem_laser_scan = SemLaserScan(project=False)
    sequences = config['data']['kitti']['train_split']
    objects_db_dir = config['data']['kitti']['object_db_dir']
    os.makedirs(objects_db_dir, exist_ok=True)

    counter = {}
    for name in config['data']['kitti']['rare_classes'].values():
        os.makedirs(os.path.join(objects_db_dir, name), exist_ok=True)
        counter[name] = 0

    for sequence in tqdm(sequences, desc="Train Sequence"):
        sequence_dir = os.path.join(config['data']['kitti']['seq_dir'], f"{sequence:02}")
        points_paths = sorted(glob.glob(os.path.join(sequence_dir, "velodyne", "*.bin")))
        labels_paths = sorted(glob.glob(os.path.join(sequence_dir, "labels", "*.label")))

        for point_path, label_path in tqdm(zip(points_paths, labels_paths), total=len(points_paths)):
            file_stem = Path(point_path).stem

            sem_laser_scan.open_scan(point_path)
            sem_laser_scan.do_range_projection()
            sem_laser_scan.open_label(label_path)
            sem_laser_scan.do_label_projection()
            apply_learning_map(sem_laser_scan, config['data']['kitti']['learning_map'])

            image = np.concatenate(
                [
                    sem_laser_scan.proj_range[..., None],
                    sem_laser_scan.proj_xyz,
                    sem_laser_scan.proj_remission[..., None],
                ],
                axis=-1
            ).astype(np.float32)

            frame_info = extract_bounding_boxes_with_visualization(
                sem_label=sem_laser_scan.proj_sem_label,
                inst_label=sem_laser_scan.proj_inst_label,
                image=image,
                rare_classes=config['data']['kitti']['rare_classes'],
            )

            for name, bboxes in frame_info.items():
                for (coords, crop) in bboxes:
                    x0, y0, x1, y1 = coords
                    coords_str = f"{x0}-{y0}-{x1}-{y1}"
                    area = (x1 - x0 + 1) * (y1 - y0 + 1)
                    save_path = os.path.join(objects_db_dir, name, f"s{sequence:02}_f{file_stem}_i{counter[name]:06}_a{area}_c{coords_str}.npy")
                    counter[name] += 1
                    np.save(save_path, crop)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise
