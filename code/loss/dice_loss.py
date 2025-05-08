# This file is covered by the LICENSE file in the root of this project.

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Computes the Dice Loss, commonly used for training segmentation models.

    Dice Loss is a measure of overlap between two samples. It ranges from 0 (no overlap)
    to 1 (perfect overlap), and is particularly useful for imbalanced datasets.

    This implementation supports multi-class segmentation and can optionally ignore
    a specific label (e.g., for background or unlabeled regions).

    Args:
        weight (list or Tensor, optional): Class weights of shape [C], where C is 
            the number of classes. Used to weight each class differently in the final loss.
        ignore_index (int, optional): Class index to ignore when computing the loss
            (e.g., -100). Pixels with this label are excluded from both prediction
            and target tensors during loss computation.
    """
    
    def __init__(self, weight=None, ignore_index=None):
        super(DiceLoss, self).__init__()
        if weight is not None:
            weight = torch.Tensor(weight)
            self.weight = weight / torch.sum(weight)  # Normalized weight
        self.ignore_index = ignore_index
        self.smooth = 1e-5

    def forward(self, predict, target):
        """
        Args:
            predict (Tensor): Raw logits from the model of shape [N, C, *], where
                N is batch size, C is number of classes, and * are spatial dimensions.
            target (Tensor): Ground-truth labels of shape [N, *], with each value
                in the range [0, C-1] or `ignore_index`.

        Returns:
            loss (Tensor): Scalar tensor representing the Dice loss (1 - mean Dice coefficient).
        """
        
        N, C = predict.size()[:2]
        predict = predict.view(N, C, -1)  # (N, C, *)
        target = target.view(N, -1)       # (N, *)

        # Apply softmax to get probabilities
        predict = F.softmax(predict, dim=1)  # (N, C, *)

        # Ignore mask
        if self.ignore_index is not None:
            mask = target != self.ignore_index  # (N, *)
            mask = mask.unsqueeze(1).expand(-1, C, -1)  # (N, C, *)
            predict = predict * mask
            target = target * mask[:, 0, :]

        # One-hot encode target
        target_onehot = torch.zeros_like(predict)  # (N, C, *)
        valid = (target >= 0) & (target < C)
        target_onehot.scatter_(1, target.unsqueeze(1) * valid.unsqueeze(1), 1)

        # Dice calculation
        intersection = torch.sum(predict * target_onehot, dim=2)  # (N, C)
        union = torch.sum(predict.pow(2), dim=2) + torch.sum(target_onehot, dim=2)  # (N, C)
        dice_coef = (2 * intersection + self.smooth) / (union + self.smooth)  # (N, C)

        # Weighted average
        if hasattr(self, 'weight'):
            if self.weight.type() != predict.type():
                self.weight = self.weight.type_as(predict)
            dice_coef = dice_coef * self.weight.unsqueeze(0) * C  # (N, C)

        return 1 - dice_coef.mean()
