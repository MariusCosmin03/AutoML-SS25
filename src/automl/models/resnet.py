import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
from torchvision.models import (
    ResNet18_Weights, ResNet34_Weights, ResNet50_Weights,
    ResNet101_Weights, ResNet152_Weights
)

from enum import Enum
import logging


class ResNetType(Enum):
    RESNET18 = 'resnet18'
    RESNET34 = 'resnet34'
    RESNET50 = 'resnet50'
    RESNET101 = 'resnet101'
    RESNET152 = 'resnet152'


class ResNetBackbone(nn.Module):
    def __init__(self, resnet_type: ResNetType, pretrained: bool, frozen: bool) -> None:
        self.resnet_type = resnet_type
        super(ResNetBackbone, self).__init__()
        match resnet_type:
            case ResNetType.RESNET18:
                weights = ResNet18_Weights.DEFAULT if pretrained else None
                orig = resnet18(weights=weights)
            case ResNetType.RESNET34:
                weights = ResNet34_Weights.DEFAULT if pretrained else None
                orig = resnet34(weights=weights)
            case ResNetType.RESNET50:
                weights = ResNet50_Weights.DEFAULT if pretrained else None
                orig = resnet50(weights=weights)
            case ResNetType.RESNET101:
                weights = ResNet101_Weights.DEFAULT if pretrained else None
                orig = resnet101(weights=weights)
            case ResNetType.RESNET152:
                weights = ResNet152_Weights.DEFAULT if pretrained else None
                orig = resnet152(weights=weights)
            case _:
                raise ValueError(f"Unsupported ResNet type: {resnet_type}")

        self.model = nn.Sequential(*list(orig.children())[:-1])

        for param in self.model.parameters():
            param.requires_grad = not frozen

        logging.info(f"Initialized {resnet_type.value} backbone with pretrained={pretrained} and frozen={frozen}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the ResNet backbone.

        Args:
            x (torch.Tensor): Input tensor of shape (N, 3, H, W).

        Returns:
            torch.Tensor: Image Features of shape (N, C_out), where C_out is the number of output channels.
        """
        features = self.model(x)
        return torch.flatten(features, 1)


if __name__ == "__main__":
    model = ResNetBackbone(ResNetType.RESNET50, pretrained=True, frozen=True)
