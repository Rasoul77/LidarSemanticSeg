# This file is covered by the LICENSE file in the root of this project.

"""
Inference Script for LiDAR Semantic Segmentation

This script performs semantic segmentation inference on LiDAR range images using a trained deep learning model.
It supports automatic mixed precision, post-processing using k-Nearest Neighbors (KNN), and saves both projected
and per-point predictions to disk in KITTI-compatible format.

Usage:
    python infer.py --checkpoint_path path/to/checkpoint_dir
"""

import argparse
import os
from argparse import Namespace

from collections import OrderedDict
import torch
from tqdm import tqdm

from data.data_loader import get_infer_dataloader
from eval.KNN import KNN
from model.model_creator import get_segmentation_model
from utility.utils import *


class Inference:
    """
    Encapsulates the inference pipeline for LiDAR range image semantic segmentation.

    Attributes:
        config (DotDict): Configuration dictionary for inference.
        device (torch.device): Device used for model inference.
        model (torch.nn.Module): Trained segmentation model.
        dataloader (DataLoader): Inference data loader.
        data_mean (np.ndarray): Mean values for normalization.
        data_std (np.ndarray): Standard deviation values for normalization.
        post_process (KNN): Optional KNN-based post-processor.
        color_map (np.ndarray): Colormap for visualizing predictions.
        projected_predictions_dir (str): Directory to save projected predictions.
    """
    
    def __init__(self, config: dict):
        """
        Initialize the Inference pipeline.

        Args:
            config (dict): Configuration dictionary for inference.
        """
        
        self.config = DotDict(config)

        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Segmentaiton Model
        self.model = get_segmentation_model(
            base_model=self.config.model.base,
            encoder_name=self.config.model.encoder,
            encoder_weights=self.config.model.weights,
            num_classes=self.config.model.num_classes,
            in_channels=self.config.model.in_channels,
            device=self.device,
        )
        self.model.load_state_dict(torch.load(self.config.model.load))
        self.model.eval()

        # Projected Predictions Directory
        self.projected_predictions_dir = None
        if self.config.eval.save_projected_pred:
            self.projected_predictions_dir = os.path.join(self.config.infer.output_dir, "visualization")
            os.makedirs(self.projected_predictions_dir, exist_ok=True)

        # Automatic Mixed Precision
        self.autocast = torch.cuda.amp.autocast(enabled=True, dtype=torch.float16)
        self.scaler = torch.cuda.amp.GradScaler(enabled=True)

        # Dataset
        self.dataloader = get_infer_dataloader(config)
        self.data_mean, self.data_std = get_mean_std_dev_train_set(config)

        # Post processing
        self.post_process = None
        if self.config.infer.post_process.enabled:
            self.post_process = KNN(
                nclasses=self.config.model.num_classes,
                knn=self.config.infer.knn.num_nearest,
                search=self.config.infer.knn.search,
                sigma=self.config.infer.knn.sigma,
                cutoff=self.config.infer.knn.cut_off,
            )
        
        # Create inference output directory structure
        self.create_output_dir()

        # Color map
        learning_map_inv = config['data']['kitti']['learning_map_inv']
        learning_map = config['data']['kitti']['learning_map']
        self.color_map = create_semantic_color_map(
            self.config.data.kitti.color_map.to_dict(), learning_map, learning_map_inv
        )

    def infer(self) -> None:
        """
        Perform inference over the dataset using the trained model.
        Outputs per-point label files and optionally saves visualizations.
        """

        with tqdm(self.dataloader, leave=False, desc="Inference") as eval_bar:
            with torch.no_grad():
                for data in eval_bar:
                    batchsize = int(data['image'].shape[0])
                    images = data['image']

                    if self.config.infer.post_process:
                        proj_range = images[..., 0].cpu()

                    images = (images - self.data_mean[None, None, None, :]) / self.data_std[None, None, None, :]

                    images = images.permute(0, 3, 1, 2).to(self.device)
                    
                    with self.autocast:
                        logits = self.model(images)

                    eval_bar.set_postfix(
                        OrderedDict(
                            sequence=int(data['sequence'][0]), frame_number=str(data['frame_number'][0])
                        )
                    )

                    pred_mask = torch.argmax(logits, dim=1).cpu()
                    
                    for i in range(batchsize):
                        sequence = f"{int(data['sequence'][i]):02}"
                        frame_number = data['frame_number'][i]
                        num_points = int(data['num_points'][i].numpy())
                        if self.config.infer.post_process.enabled:
                            per_point_pred = self.post_process(
                                proj_range = proj_range[i, ...],
                                unproj_range = data['unproj_range'][i, :num_points],
                                proj_argmax=pred_mask[i, ...],
                                px=data['p_x'][i, :num_points],
                                py=data['p_y'][i, :num_points],
                            ).numpy()
                        else:
                            point_index_mask = data['proj_idx'][i, :].numpy()                    
                            per_point_pred = get_per_point_pred(
                                pred_mask=pred_mask[i, ...].numpy(),
                                point_index_mask=point_index_mask,
                                num_points=num_points,
                            )

                        if self.config.infer.save_projected_pred:
                            save_proj_pred(
                                pred_mask[i, ...],
                                os.path.join(
                                    self.projected_predictions_dir, f"{sequence}_{frame_number}.png"
                                ),
                                semantic_color_map=self.color_map
                            )

                        self.create_and_save_predictions(
                            per_point_label=per_point_pred,
                            sequence=sequence,
                            frame_number=frame_number,
                        )

    def create_output_dir(self):
        """
        Create KITTI-format output directory structure for storing per-frame label files.

            sequences
            ├── 11
            │   └── predictions
            │         ├ 000000.label
            │         ├ 000001.label
            │         ├ ...
            ├── 12
            │   └── predictions
            │         ├ 000000.label
            │         ├ 000001.label
            │         ├ ...
            ├── 13
            .
            .
            .
            └── 21
        """
        sequences = os.path.join(self.config.infer.output_dir, "sequences")
        os.makedirs(sequences, exist_ok=True)
        for scene_number in self.config.data.kitti.test_split:
            scene_number_dir = os.path.join(sequences, str(scene_number))
            os.makedirs(scene_number_dir, exist_ok=True)
            predictions = os.path.join(scene_number_dir, "predictions")
            os.makedirs(predictions, exist_ok=True)

    def create_and_save_predictions(
        self,
        per_point_label: np.ndarray,
        sequence: str,
        frame_number: str,
    ):
        """
        Save the final per-point semantic predictions to a KITTI-style .label file.

        Args:
            per_point_label (np.ndarray): Array of predicted labels for each point.
            sequence (str): Sequence ID (e.g., "11").
            frame_number (str): Frame number (e.g., "000123").
        """

        output_file = os.path.join(
            self.config.infer.output_dir, "sequences", sequence, "predictions", frame_number + ".label"
        )
        per_point_label.tofile(output_file)


def get_args():
    """
    Parse command-line arguments.

    Returns:
        Namespace: Parsed arguments.
    """
    
    parser = argparse.ArgumentParser(description='Inference Script')
    parser.add_argument('--checkpoint_path', default=None, help='Path to the checkpoint directory.')

    return parser.parse_args()


def main(args: Namespace) -> None:
    """
    Main function to run the inference pipeline.

    Args:
        args (Namespace): Command-line arguments.
    """
    
    # Load configurations
    config = load_configurations(root_dir=args.checkpoint_path, config_name=None)
    config['train']['root_chkpt'] = args.checkpoint_path
    config['model']['load'] = os.path.join(args.checkpoint_path, "best_val_score_model.pt")

    # Run inference
    trainer = Inference(config) 
    trainer.infer()


if __name__ == "__main__":    
    args = get_args()

    try:
        main(args)
    except KeyboardInterrupt:
        raise    
