# This file is covered by the LICENSE file in the root of this project.

import glob
import os
import pickle
import random
import time
import yaml

from pathlib import Path
from collections.abc import Mapping
from typing import Any, Tuple, Union

import cv2
import torch
import numpy as np

from loss.dice_loss import DiceLoss
from loss.jaccard_loss import JaccardLoss


class DotDict:
    """A dictionary wrapper that allows dot notation access to keys."""

    def __init__(self, dictionary: dict):
        self._data = {k: self._convert(v) for k, v in dictionary.items()}

    def _convert(self, obj: Any):
        if isinstance(obj, Mapping):
            return DotDict(obj)
        elif isinstance(obj, list):
            return [self._convert(item) for item in obj]
        return obj

    def __getattr__(self, name: str):
        if name in self._data:
            return self._data[name]
        raise AttributeError(f"'DotDict' object has no attribute '{name}'")

    def __getitem__(self, key):
        return self._data[key]

    def __repr__(self):
        return f"DotDict({self._data})"

    def to_dict(self):
        """Recursively convert back to a plain dictionary."""
        def _convert_back(obj):
            if isinstance(obj, DotDict):
                return {k: _convert_back(v) for k, v in obj._data.items()}
            elif isinstance(obj, list):
                return [_convert_back(item) for item in obj]
            else:
                return obj
        return _convert_back(self)


def load_configurations(config_name: Union[str, None], root_dir: str = "configs") -> dict:
    """
    Load and merge main configuration and data configuration YAML files.
    """
    data_config_path = os.path.join(root_dir, "data_config.yaml")

    if config_name is None:
        all_yaml_files = glob.glob(os.path.join(root_dir, "*.yaml"))
        config_path = list(set(all_yaml_files) - set([data_config_path]))[0]
    else:
        config_path = os.path.join(root_dir, config_name)

    if not os.path.exists(config_path):
        print("WARNING! Configuration file does not exist!")
        exit(1)
    if not os.path.exists(data_config_path):
        print("WARNING! Data configuration file does not exist!")
        exit(1)

    config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)
    data_config = yaml.load(open(data_config_path, "r"), Loader=yaml.FullLoader)
    config.update(data_config)

    config['config_path'] = config_path
    config['data_config_path'] = data_config_path

    return config


def get_seed(config: dict) -> int:
    """
    Get a random seed from config, or generate one if not provided.
    """
    random_seed = config.get('seed', None)
    return random.randint(0, 4294967295) if random_seed is None or random_seed < 0 else random_seed


def set_random_seed(seed: int, deterministic: bool) -> None:
    """
    Set random seed for reproducibility across common libraries.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic


def save_model(model: torch.nn.Module, save_file_name: str, output_path: str) -> None:
    """
    Save model parameters to file.
    """
    torch.save(model.state_dict(), os.path.join(output_path, save_file_name))


def get_loss_function(config: dict):
    """
    Construct a loss function (or a list of weighted loss functions) based on config.
    """
    config_loss_function = config['train']['loss']['function']
    ignore_index = config['train']['loss']['ignore_idx']

    if not isinstance(config_loss_function, list) or len(config_loss_function) % 2 != 0:
        print("WARNING! Loss function format in yaml file is not valid!")
        print("Expected format: [<name>, <weight>, ...]")
        exit(1)

    loss_map = {
        "jaccard": JaccardLoss(ignore_index=ignore_index),
        "dice": DiceLoss(ignore_index=ignore_index),
        "xentropy": torch.nn.CrossEntropyLoss(ignore_index=ignore_index),
    }

    losses = [
        (loss_map[config_loss_function[i]], config_loss_function[i + 1])
        for i in range(0, len(config_loss_function), 2)
    ]

    return losses[0][0] if len(losses) == 1 else losses


def get_objects_database(config: dict) -> dict:
    """
    Load object database paths from disk.
    """
    objects_db_dir = config['data']['kitti']['object_db_dir']
    if objects_db_dir and not os.path.exists(objects_db_dir):
        print("Rare objects directory not found! Please run 'tools/calc_trainset_stat.py'.")
        exit(1)

    object_db = {}
    for name in os.listdir(objects_db_dir):
        full_path = os.path.join(objects_db_dir, name)
        if os.path.isdir(full_path):
            object_db[name] = glob.glob(os.path.join(full_path, "*.npy"))

    print(f"Rare object classes: {list(object_db.keys())}")
    return object_db


def get_mean_std_dev_train_set(config: dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load mean and standard deviation statistics for training features.
    """
    mean_path = config['data']['kitti']['mean_pkl_path']
    std_path = config['data']['kitti']['std_pkl_path']

    if not os.path.exists(mean_path) or not os.path.exists(std_path):
        print("Mean or std file not found! Run 'tools/calc_statistics.py' to generate them.")
        exit(1)

    try:
        mean = pickle.load(open(mean_path, 'rb'))
        std = pickle.load(open(std_path, 'rb'))
    except Exception as e:
        print("WARNING! Failed to load pickle files:", e)
        exit(1)

    mean_arr = np.array([mean[k] for k in ["range", "x", "y", "z", "remission"]], dtype=np.float32)
    std_arr = np.array([std[k] for k in ["range", "x", "y", "z", "remission"]], dtype=np.float32)
    return mean_arr, std_arr


def save_per_class_score(path: str, per_class_score: np.ndarray, config: DotDict) -> None:
    """
    Save and optionally print per-class scores from evaluation.
    """
    to_save = {}
    class_names = config.data.kitti.class_names.to_dict()

    for class_idx, class_name in class_names.items():
        to_save[class_name] = per_class_score[class_idx]

    pickle.dump(to_save, open(os.path.join(path, "per_class_score_dict.pkl"), "wb"))

    if config.eval.print_per_class_score:
        max_len = max(len(label) for label in to_save)
        print("Per Class Score:")
        for label, value in to_save.items():
            print(f"{label:<{max_len}}  {value*100.0:.3f}%")


def get_per_point_pred(pred_mask: np.ndarray, point_index_mask: np.ndarray, num_points: int) -> np.ndarray:
    """
    Convert voxel-level predictions to point-level predictions.
    """
    per_point_pred = np.zeros(shape=(num_points), dtype=np.int32)

    point_index_mask = point_index_mask.flatten()
    pred_mask = pred_mask.flatten()

    valid_indices = point_index_mask >= 0
    per_point_pred[point_index_mask[valid_indices]] = pred_mask[valid_indices]

    return per_point_pred


def get_formatted_time_execution(start_time: float) -> str:
    """
    Return execution time as a formatted string mm:ss.
    """
    execution_time = time.time() - start_time
    minutes = int(execution_time // 60)
    seconds = int(execution_time % 60)
    return f"{minutes:02d}:{seconds:02d}"


def read_number_from_file(variable_name: str, root_path: str, default: float) -> float:
    """
    Read a float number from a text file or return the default value.
    """
    file_path = os.path.join(root_path, f"{variable_name}.txt")
    if os.path.exists(file_path):
        with open(file_path, "r") as infile:
            return float(infile.readline())
    return default


def save_number_to_file(variable_name: str, root_path: str, value: float) -> None:
    """
    Save a float number to a text file.
    """
    file_path = os.path.join(root_path, f"{variable_name}.txt")
    with open(file_path, "w") as outfile:
        outfile.write(str(value))


def create_semantic_color_map(color_map: dict, learning_map: dict, learning_map_inv: dict):
    """
    Creates a color map for semantic segmentation predictions, mapping each training label
    to an RGB color based on the original dataset's label-color associations.

    Args:
        color_map (dict): A dictionary mapping original label IDs to RGB color values.
        learning_map (dict): A dictionary mapping original label IDs to training label IDs.
        learning_map_inv (dict): A dictionary mapping training label IDs back to original label IDs.

    Returns:
        np.ndarray: An array of shape (num_classes, 3) where each row is an RGB color
                    corresponding to a training label.
    """

    num_classes = max(learning_map.values()) + 1

    semantic_color_map = np.zeros((num_classes, 3), dtype=np.uint8)

    for train_label in range(num_classes):

        original_label = learning_map_inv.get(train_label, None)
        if original_label is not None:

            color = color_map.get(original_label, [0, 0, 0])
        else:
            original_labels = [k for k, v in learning_map.items() if v == train_label]
            if original_labels:
                color = color_map.get(original_labels[0], [0, 0, 0])
            else:
                color = [0, 0, 0]
        semantic_color_map[train_label] = color

    return semantic_color_map


def save_proj_pred(pred: np.ndarray, save_path: str, semantic_color_map: np.ndarray):    
    """
    Saves a projected prediction mask as a colorized PNG image.

    Args:
        pred (np.ndarray): A 2D array of predicted class indices (shape: H x W).
        save_path (str): Path to save the colorized PNG image.
        semantic_color_map (np.ndarray): Array mapping class indices to RGB color values
                                         (shape: num_classes x 3).
    """    
    pred_colorized = semantic_color_map[pred]
    cv2.imwrite(save_path, pred_colorized)