import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
from torchvision.models import (
    ResNet18_Weights, ResNet34_Weights, ResNet50_Weights,
    ResNet101_Weights, ResNet152_Weights
)

from typing import Literal
import logging


class ResNetBackbone(nn.Module):
    def __init__(self, resnet_type: Literal[18, 34, 50, 101, 152], pretrained: bool, frozen: bool) -> None:
        self.resnet_type = resnet_type
        self.pretrained = pretrained
        self.frozen = frozen
        self.in_shape = (3, 224, 224)
        super(ResNetBackbone, self).__init__()
        match resnet_type:
            case 18:
                weights = ResNet18_Weights.DEFAULT if pretrained else None
                orig = resnet18(weights=weights)
                self.out_channels = 512
            case 34:
                weights = ResNet34_Weights.DEFAULT if pretrained else None
                orig = resnet34(weights=weights)
                self.out_channels = 512
            case 50:
                weights = ResNet50_Weights.DEFAULT if pretrained else None
                orig = resnet50(weights=weights)
                self.out_channels = 2048
            case 101:
                weights = ResNet101_Weights.DEFAULT if pretrained else None
                orig = resnet101(weights=weights)
                self.out_channels = 2048
            case 152:
                weights = ResNet152_Weights.DEFAULT if pretrained else None
                orig = resnet152(weights=weights)
                self.out_channels = 2048
            case _:
                raise ValueError(f"Unsupported ResNet type: {resnet_type}")

        self.model = nn.Sequential(*list(orig.children())[:-1])

        for param in self.model.parameters():
            param.requires_grad = not frozen

        logging.info(f"Initialized {resnet_type} backbone with pretrained={pretrained} and frozen={frozen}")

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
    model = ResNetBackbone(50, pretrained=True, frozen=True)
