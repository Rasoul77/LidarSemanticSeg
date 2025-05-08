# This file is covered by the LICENSE file in the root of this project.

from typing import Union

import torch

import torch.nn as nn
import segmentation_models_pytorch as smp


def get_segmentation_model(
    base_model: str,
    encoder_name: str,
    num_classes: int,
    in_channels: int,
    device: Union[torch.device, str],
    encoder_weights: str = 'imagenet',
) -> nn.Module:
    """
    Creates a semantic segmentation model using the segmentation_models_pytorch library.

    Args:
        base_model (str): The name of the segmentation model architecture to use.
            Supported architectures include: Unet, Unet++, MAnet, Linknet, FPN, PSPNet,
            PAN, DeepLabV3, DeepLabV3+, UperNet.
        encoder_name (str): The name of the encoder (backbone) to use for feature extraction.
            Supported encoders include: ResNet, ResNeXt, ResNeSt, ResNe(X)t, RegNet(x/y),
            SE-Net, SK-ResNe(X)t, DenseNet, Inception, EfficientNet, MobileNet, DPN, VGG,
            Mix Vision Transformer, MobileOne.
        num_classes (int): The number of classes in the segmentation task. This determines the
            number of output channels of the model.
        in_channels (int): The number of input channels of the image data. For example, 3 for RGB
            images, 1 for grayscale images, or more for multi-spectral images.
        device (torch.device or str): The device to which the model will be moved
            (e.g., 'cuda' for GPU, 'cpu' for CPU).
        encoder_weights (str, optional): The pre-trained weights to use for the encoder.
            Defaults to 'imagenet'.  Other possible values are 'imagenet', 'ssl', 'swsl', or None.
            If None, the encoder will be initialized with random weights.

    Returns:
        torch.nn.Module: The created segmentation model.

    Example:
        >>> import torch
        >>> model = get_segmentation_model(
        ...     base_model='Unet',
        ...     encoder_name='resnet34',
        ...     num_classes=10,
        ...     in_channels=3,
        ...     device='cuda',
        ...     encoder_weights='imagenet'
        ... )
        >>> print(model)
        Unet(
          (encoder): ResNetEncoder(...)
          (decoder): UnetDecoder(...)
          (segmentation_head): SegmentationHead(...)
        )
    """
    model = smp.create_model(
        arch=base_model,
        encoder_name=encoder_name,
        in_channels=in_channels,
        classes=num_classes,
        encoder_weights=encoder_weights,
    ).to(device)

    return model
