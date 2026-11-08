"""ResNet18 model definition for Dataset B ultrasound classification.

Architecture follows the original CNN experiment:
- ImageNet-pretrained ResNet18 backbone
- One-channel (grayscale) input
- First convolution initialized by averaging pretrained RGB filters
- Four output classes (Heckmatt Grades 1–4)
"""

from typing import Optional

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


def build_resnet18(
    num_classes: int = 4,
    pretrained: bool = True,
    weights: Optional[ResNet18_Weights] = None,
) -> nn.Module:
    """Build the grayscale ResNet18 classifier.

    Args:
        num_classes: Number of output classes. The experiment uses four.
        pretrained: Load ImageNet weights when ``weights`` is not supplied.
        weights: Optional explicit torchvision ResNet18 weights. If provided,
            this takes precedence over ``pretrained``.

    Returns:
        A ResNet18 model accepting tensors shaped (batch, 1, height, width).
    """
    if weights is None and pretrained:
        weights = ResNet18_Weights.DEFAULT

    model = resnet18(weights=weights)

    # Convert the original 3-channel convolution to a 1-channel convolution.
    original_conv = model.conv1
    grayscale_conv = nn.Conv2d(
        in_channels=1,
        out_channels=original_conv.out_channels,
        kernel_size=original_conv.kernel_size,
        stride=original_conv.stride,
        padding=original_conv.padding,
        bias=False,
    )

    # Preserve pretrained initialization by averaging the RGB input filters.
    with torch.no_grad():
        grayscale_conv.weight.copy_(original_conv.weight.mean(dim=1, keepdim=True))

    model.conv1 = grayscale_conv

    # Replace the ImageNet classifier with the four-grade classifier.
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model
