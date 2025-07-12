import torch
from torchvision.models import (
    vit_b_16, ViT_B_16_Weights,
    vit_b_32, ViT_B_32_Weights,
    vit_l_16, ViT_L_16_Weights,
    vit_l_32, ViT_L_32_Weights,
    vit_h_14, ViT_H_14_Weights
)

from enum import Enum
import logging


class ViTType(Enum):
    VIT_B_16 = 'vit_b_16'
    VIT_B_32 = 'vit_b_32'
    VIT_L_16 = 'vit_l_16'
    VIT_L_32 = 'vit_l_32'
    VIT_H_14 = 'vit_h_14'


class ViTBackbone(torch.nn.Module):
    def __init__(self, vit_type: ViTType, pretrained: bool, frozen: bool) -> None:
        self.vit_type = vit_type
        super(ViTBackbone, self).__init__()
        match vit_type:
            case ViTType.VIT_B_16:
                weights = ViT_B_16_Weights.DEFAULT if pretrained else None
                self.model = vit_b_16(weights=weights)
                self.out_channels = 768
            case ViTType.VIT_B_32:
                weights = ViT_B_32_Weights.DEFAULT if pretrained else None
                self.model = vit_b_32(weights=weights)
                self.out_channels = 768
            case ViTType.VIT_L_16:
                weights = ViT_L_16_Weights.DEFAULT if pretrained else None
                self.model = vit_l_16(weights=weights)
                self.out_channels = 1024
            case ViTType.VIT_L_32:
                weights = ViT_L_32_Weights.DEFAULT if pretrained else None
                self.model = vit_l_32(weights=weights)
                self.out_channels = 1024
            case ViTType.VIT_H_14:
                weights = ViT_H_14_Weights.DEFAULT if pretrained else None
                self.model = vit_h_14(weights=weights)
                self.out_channels = 1280
            case _:
                raise ValueError(f"Unsupported ViT type: {vit_type}")

        for param in self.model.parameters():
            param.requires_grad = not frozen

        logging.info(f"Initialized {vit_type.value} backbone with pretrained={pretrained} and frozen={frozen}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the ViT backbone.

        Args:
            x (torch.Tensor): Input tensor of shape (N, 3, H, W).

        Returns:
            torch.Tensor: Image Features of shape (N, C_out), where C_out is the number of output channels.
        """
        x = self.model._process_input(x)
        n = x.size(0)
        cls_tok = self.model.class_token.expand(n, -1, -1)
        tokens = torch.cat([cls_tok, x], dim=1)
        feat_seq = self.model.encoder(tokens)
        cls_embed = feat_seq[:, 0]
        return cls_embed


if __name__ == "__main__":
    model = ViTBackbone(ViTType.VIT_B_16, pretrained=True, frozen=False).cuda()
    d2 = torch.randn(1, 3, 224, 224).cuda()
    print(model(d2).shape)
    print(model(d1).shape)
