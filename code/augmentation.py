# This file is covered by the LICENSE file in the root of this project.

import random
import torch

from pathlib import Path
from itertools import combinations
from typing import Tuple

import numpy as np


def _apply_cutmix(
    images: torch.Tensor,
    masks: torch.Tensor,
    alpha: float = 0.2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies the CutMix data augmentation technique to a batch of images and masks.

    CutMix replaces a random patch of one image with the corresponding region from another image 
    in the batch and updates the semantic mask accordingly. The patch size is sampled from a 
    Beta distribution.

    Args:
        images (torch.Tensor): Batch of input images of shape (B, C, H, W).
        masks (torch.Tensor): Corresponding masks of shape (B, H, W).
        alpha (float, optional): Beta distribution parameter controlling the CutMix patch size. Defaults to 0.2.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The augmented images and masks.
    """
    
    batch_size, _, _, _ = images.shape

    # Initialize augmented images and one-hot masks
    augmented_images = images.clone()
    augmented_masks = masks.clone()

    # Generate random pairs for mixing up
    pairs = list(combinations([i for i in range(batch_size)], 2))
    random.shuffle(pairs)

    # Process pairs of data-points
    for i in range(min(len(pairs), batch_size)):
        pair = pairs[i]
        
        img1, img2 = images[pair[0]], images[pair[1]]
        mask1, mask2 = masks[pair[0]], masks[pair[1]]

        # Apply cutmix
        h, w, _ = img1.shape
        lam = torch.distributions.Beta(alpha, alpha).sample().item()
        lam = max(0.2, min(0.8, lam))  # Constrain lam for effective mixing

        cut_x = torch.randint(w, (1,)).item()
        cut_y = torch.randint(h, (1,)).item()
        cut_w = int(np.sqrt(1 - lam) * w)
        cut_h = int(np.sqrt(1 - lam) * h)

        x1 = max(0, cut_x - cut_w // 2)
        x2 = min(w, cut_x + cut_w // 2)
        y1 = max(0, cut_y - cut_h // 2)
        y2 = min(h, cut_y + cut_h // 2)

        augmented_images[i] = img1.clone()
        augmented_images[i, y1:y2, x1:x2, :] = img2[y1:y2, x1:x2, :]

        augmented_masks[i] = mask1.clone()
        augmented_masks[i, y1:y2, x1:x2] = mask2[y1:y2, x1:x2]

    return augmented_images, augmented_masks
    

def _replace_image_and_mask_foreground(
    image, image_mask, data, mask, x_0, y_0, x_1, y_1, foreground_index
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Injects a foreground object into a specified region of an image and updates the semantic mask.

    This function overlays a selected region from a source object into a destination image at the given coordinates,
    using a binary mask to define the foreground. The corresponding mask values are updated with a given class index.

    Args:
        image (np.ndarray): Destination image of shape (H, W, C).
        image_mask (np.ndarray): Destination semantic mask of shape (H, W).
        data (np.ndarray): Source object data of shape (h, w, C).
        mask (np.ndarray): Binary mask of shape (h, w) indicating foreground in the source.
        x_0, y_0, x_1, y_1 (int): Coordinates in the destination image where the object should be inserted.
        foreground_index (int): Index to assign in the semantic mask for foreground pixels.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Updated image and semantic mask.
    """
    
    h, w = mask.shape
    H, W = image_mask.shape  # Get shape from image_mask

    # Ensure the target region in 'image' and 'image_mask' is within bounds
    image_x_start = max(0, x_0)
    image_y_start = max(0, y_0)
    image_x_end = min(W, x_1)
    image_y_end = min(H, y_1)

    # Calculate the corresponding region in 'data' and 'mask'
    data_x_start = max(0, -x_0)
    data_y_start = max(0, -y_0)
    data_x_end = w - max(0, x_1 - W)
    data_y_end = h - max(0, y_1 - H)

    # Extract the relevant portions of 'data' and 'mask'
    data_roi = data[data_y_start:data_y_end, data_x_start:data_x_end]
    mask_roi = mask[data_y_start:data_y_end, data_x_start:data_x_end]

    # Extract the target region in 'image' and 'image_mask'
    image_roi = image[image_y_start:image_y_end, image_x_start:image_x_end]
    image_mask_roi = image_mask[image_y_start:image_y_end, image_x_start:image_x_end]

    # Apply the mask to select the foreground from 'data_roi'
    foreground = data_roi[mask_roi]

    # Create a boolean mask for the target region in 'image' based on the 'mask_roi'
    image_mask_bool = np.tile(mask_roi[:, :, np.newaxis], (1, 1, image.shape[2]))

    # Replace the corresponding pixels in 'image_roi' with the foreground
    image_roi[image_mask_bool] = foreground.flatten()

    # Update the 'image_mask_roi' with the foreground index where the mask is True
    image_mask_roi[mask_roi] = foreground_index

    # Update the original 'image' and 'image_mask' with the modified regions
    image[image_y_start:image_y_end, image_x_start:image_x_end] = image_roi
    image_mask[image_y_start:image_y_end, image_x_start:image_x_end] = image_mask_roi

    return image, image_mask


def _apply_instance_injection(
    images: torch.Tensor,
    masks: torch.Tensor,
    objects_db: dict,
    rare_classes: dict,
    num_objects_per_cat: Tuple[int, int] = (2, 10),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Performs instance-level object injection into semantic segmentation data.

    For each image in the batch, a random subset of object instances from the object database
    is inserted into the image at predefined coordinates. The corresponding semantic mask is updated.

    Args:
        images (torch.Tensor): Batch of images of shape (B, C, H, W).
        masks (torch.Tensor): Batch of semantic masks of shape (B, H, W).
        objects_db (dict): Mapping from class names to lists of file paths of object instances.
        rare_classes (dict): Mapping from class indices to class names for rare categories.
        num_objects_per_cat (Tuple[int, int], optional): Min and max number of objects per category to inject.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The modified images and masks as PyTorch tensors.
    """
    
    names = list(objects_db.keys())
    selected_names = np.random.choice(names, size=len(names) - 1, replace=False)

    images = images.numpy()
    masks = masks.numpy()

    batch_size = images.shape[0]

    name_to_index = {v:k for k, v in rare_classes.items()}
    for index in range(batch_size):
        image = images[index, ...]
        image_mask = masks[index, ...]

        for name in selected_names:
            num_objects = np.random.randint(num_objects_per_cat[0], num_objects_per_cat[1], size=1)
            selected_objects_paths = np.random.choice(objects_db[name], size=num_objects, replace=False)
            foreground_index = name_to_index[name]
            for object_path in selected_objects_paths:
                x0y0x1y1 = Path(object_path).stem.split("_")[-1]
                x0, y0, x1, y1 = x0y0x1y1[1:].split("-")
                x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
                data = np.load(object_path)
                object_data = data[..., 0:5]
                object_mask = data[..., -1].astype('bool')
                
                image, image_mask = _replace_image_and_mask_foreground(
                    image, image_mask, object_data, object_mask, x0, y0, x1, y1, foreground_index
                )

        images[index, ...] = image
        masks[index, ...] = image_mask
    
    return torch.from_numpy(images), torch.from_numpy(masks)


def apply_augmentation(
    images: torch.Tensor,
    masks: torch.Tensor,
    objects_db: dict,
    rng: np.random.Generator,
    rare_classes: dict,
    num_objects_per_cat: Tuple[int, int] = (2, 10),
    aug_prob: float = 0.5,
    instance_injection_prob: float = 0.5,
    cut_mix_prob: float = 0.5,       
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies a combination of instance injection and CutMix augmentations to a batch of images and masks.

    This function probabilistically applies instance injection and CutMix based on input probabilities.
    It is designed to increase data diversity for semantic segmentation tasks, especially in low-resource settings.

    Args:
        images (torch.Tensor): Batch of input images of shape (B, C, H, W).
        masks (torch.Tensor): Batch of corresponding semantic masks of shape (B, H, W).
        objects_db (dict): Dictionary mapping class names to object instance file paths.
        rng (np.random.Generator): NumPy random number generator for reproducibility.
        rare_classes (dict): Mapping from class indices to class names for rare categories.
        num_objects_per_cat (Tuple[int, int], optional): Range of object count per class for injection.
        aug_prob (float, optional): Probability of applying any augmentation at all.
        instance_injection_prob (float, optional): Probability of applying instance injection.
        cut_mix_prob (float, optional): Probability of applying CutMix.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The augmented images and masks.
    """
    
    if rng.random() < aug_prob:
        if rng.random() < instance_injection_prob:
            images, masks = _apply_instance_injection(
                images, masks, objects_db, rare_classes, num_objects_per_cat
            )
        
        if rng.random() < cut_mix_prob:
            images, masks = _apply_cutmix(images, masks)

    return images, masks
        
