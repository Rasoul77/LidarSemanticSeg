# This file is covered by the LICENSE file in the root of this project.

"""
LiDAR Semantic Segmentation Training Script

This script trains a deep learning model for semantic segmentation on LiDAR range images.
It supports features such as automatic mixed precision (AMP), optional post-processing using
k-Nearest Neighbors (KNN), and saving intermediate visualizations and final predictions in
KITTI-compatible format.

Key Features:
- Configurable model architecture and training parameters via YAML configuration file
- Efficient training with optional AMP for faster computation and lower memory usage
- Optional KNN-based refinement of segmentation predictions
- Automatic saving of both projected (image-space) and per-point (3D-space) predictions

Usage:
    python train.py --config_name <config_file.yaml> [--other_options]
"""

import argparse
from argparse import Namespace

from trainer import Trainer
from utility.utils import load_configurations


def get_args():
    parser = argparse.ArgumentParser(description='Train Script')
    parser.add_argument('--config_name', default=None, help='The configuration yaml file name.')
    parser.add_argument('--use-wandb', action='store_true', default=False, help='Use WandB for web logging?')
    parser.add_argument('--name', default=None, help='A descriptive name for wandb logging.')

    return parser.parse_args()


def main(args: Namespace) -> None:
    # Load configurations
    config = load_configurations(config_name=args.config_name)
    config['wandb'] = {
        "enabled": args.use_wandb,
        "project_name": args.name,
    }

    # Run training
    trainer = Trainer(config) 
    trainer.run()


if __name__ == "__main__":    
    args = get_args()

    try:
        main(args)
    except KeyboardInterrupt:
        raise    
