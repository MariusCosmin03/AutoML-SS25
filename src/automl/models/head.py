import torch
import torch.nn as nn
from enum import Enum
import logging


class HeadActivationType(Enum):
    RELU = 'relu'
    SIGMOID = 'sigmoid'
    # TODO: What else?


class PredictionHead(nn.Module):
    def __init__(self,
                 in_features: int, n_classes: int,
                 n_hidden: int, d_hidden: int, activation: HeadActivationType) -> None:
        super(PredictionHead, self).__init__()
        self.in_features = in_features
        self.n_classes = n_classes
        self.n_hidden = n_hidden
        self.d_hidden = d_hidden
        self.activation = activation

        match activation:
            case HeadActivationType.RELU:
                activation_fn = nn.ReLU()
            case HeadActivationType.SIGMOID:
                activation_fn = nn.Sigmoid()
            case _:
                raise ValueError(f"Unsupported activation type: {activation}")

        hidden_blocks = [nn.Linear(d_hidden, d_hidden), activation_fn] * n_hidden

        self.model = nn.Sequential(
            nn.Linear(in_features, n_hidden),
            activation_fn,
            *hidden_blocks,
            nn.Linear(d_hidden, n_classes)
        )

        logging.info(f"Initialized PredictionHead with in_features={in_features}, "
                     f"n_classes={n_classes}, n_hidden={n_hidden}, d_hidden={d_hidden}, "
                     f"activation={activation.value}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the prediction head.

        Args:
            x (torch.Tensor): Input tensor of shape (N, in_features).

        Returns:
            torch.Tensor: Output tensor of shape (N, n_classes).
        """
        return self.model(x)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    head = PredictionHead(in_features=512, n_classes=10, n_hidden=2, d_hidden=256, activation=HeadActivationType.RELU)
    head.cuda()
