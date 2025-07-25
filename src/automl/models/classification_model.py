import torch.nn as nn
import logging

from automl.models.resnet import ResNetBackbone
from automl.models.vit import ViTBackbone, ViTType
from automl.models.head import PredictionHead

ImageBackbone = ResNetBackbone | ViTBackbone


class ClassificationModel(nn.Module):
    def __init__(self,
                 backbone: ImageBackbone,
                 head: PredictionHead) -> None:
        super(ClassificationModel, self).__init__()
        self.backbone = backbone
        self.head = head

        logging.info(f"Initialized ClassificationModel with backbone={type(backbone).__name__} "
                     f"and head={type(head).__name__}")


def make_model(model_family: str, resnet_type: int | None, backbone_frozen: bool,
               in_features: int, n_classes: int, head_activation: str, head_n_hidden: int,
               head_d_hidden: int, dropout: float) -> ClassificationModel:
    match model_family:
        case "resnet":
            if resnet_type is None:
                raise ValueError("resnet_type must be specified for ResNet backbone.")
            backbone = ResNetBackbone(resnet_type=resnet_type, pretrained=True, frozen=backbone_frozen)
        case _:
            raise ValueError(f"Unsupported model family: {model_family}")
    
    


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Example usage
    resnet_backbone = ResNetBackbone(resnet_type=ResNetType.RESNET50, pretrained=True, frozen=False)
    vit_backbone = ViTBackbone(vit_type=ViTType.VIT_B_16, pretrained=True, frozen=False)

    prediction_head = PredictionHead(in_features=2048, n_classes=10, n_hidden=2,
                                     d_hidden=512, activation=HeadActivationType.RELU)

    model = ClassificationModel(backbone=resnet_backbone, head=prediction_head)
    print(model)
