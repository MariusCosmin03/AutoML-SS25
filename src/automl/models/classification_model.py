import torch
import torch.nn as nn
import logging

from automl.models.resnet import ResNetBackbone, ResNetType
from automl.models.vit import ViTBackbone, ViTType
from automl.models.head import PredictionHead, HeadActivationType

type ImageBackbone = ResNetBackbone | ViTBackbone


class ClassificationModel(nn.Module):
    def __init__(self,
                 backbone: ImageBackbone,
                 head: PredictionHead) -> None:
        super(ClassificationModel, self).__init__()
        self.backbone = backbone
        self.head = head

        logging.info(f"Initialized ClassificationModel with backbone={type(backbone).__name__} "
                     f"and head={type(head).__name__}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Example usage
    resnet_backbone = ResNetBackbone(resnet_type=ResNetType.RESNET50, pretrained=True, frozen=False)
    vit_backbone = ViTBackbone(vit_type=ViTType.VIT_B_16, pretrained=True, frozen=False)

    prediction_head = PredictionHead(in_features=2048, n_classes=10, n_hidden=2,
                                     d_hidden=512, activation=HeadActivationType.RELU)

    model = ClassificationModel(backbone=resnet_backbone, head=prediction_head)
    print(model)
