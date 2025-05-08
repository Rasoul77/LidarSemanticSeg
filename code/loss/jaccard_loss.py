# This file is covered by the LICENSE file in the root of this project.

import torch
import torch.nn as nn
import torch.nn.functional as F


class JaccardLoss(nn.Module):
    """
    Computes the Jaccard Loss (also known as Intersection-over-Union or IoU Loss) 
    for multi-class semantic segmentation tasks.

    Jaccard Loss is defined as:
        IoU = intersection / (union - intersection)
    and the loss is computed as:
        Loss = 1 - mean(IoU)

    This implementation supports multi-class predictions and can optionally ignore
    specific target labels (e.g., for background or unlabeled regions). It is useful
    in handling class imbalance and improving boundary quality in segmentation outputs.

    Args:
        weight (list or Tensor, optional): Class weights of shape [C], where C is the
            number of classes. Used to weigh each class's contribution to the final loss.
        ignore_index (int, optional): Label to ignore during loss computation (e.g., -100).
            Pixels with this label are excluded from both the predictions and targets.
    """

    def __init__(self, weight=None, ignore_index=None):
        super(JaccardLoss, self).__init__()
        self.ignore_index = ignore_index
        self.smooth = 1e-5

        if weight is not None:
            weight = torch.Tensor(weight)
            self.weight = weight / torch.sum(weight)
        else:
            self.weight = None

    def forward(self, predict, target):
        """
        Computes the Jaccard loss between predicted logits and ground-truth labels.

        Args:
            predict (Tensor): Raw model outputs (logits) of shape [N, C, *], where:
                N = batch size,
                C = number of classes,
                * = spatial dimensions (e.g., H, W).
            target (Tensor): Ground-truth labels of shape [N, *], with integer class
                indices in [0, C-1] or equal to `ignore_index`.

        Returns:
            loss (Tensor): A scalar tensor representing the Jaccard loss.
        """
        
        N, C = predict.size()[:2]
        predict = predict.view(N, C, -1)  # [N, C, *] → [N, C, H*W]
        target = target.view(N, -1)       # [N, H, W] → [N, H*W]

        if self.ignore_index is not None:
            mask = target != self.ignore_index
            predict = predict * mask.unsqueeze(1)  # Broadcast over channel dim
            target = target * mask

        probs = F.softmax(predict, dim=1)
        target_onehot = torch.zeros_like(probs).scatter_(1, target.unsqueeze(1), 1)

        intersection = torch.sum(probs * target_onehot, dim=2)
        union = torch.sum(probs + target_onehot, dim=2) - intersection

        jaccard = (intersection + self.smooth) / (union + self.smooth)

        if self.weight is not None:
            if self.weight.type() != predict.type():
                self.weight = self.weight.type_as(predict)
            jaccard = jaccard * self.weight.unsqueeze(0) * C

        loss = 1 - torch.mean(jaccard)
        return loss
