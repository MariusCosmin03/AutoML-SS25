import torch
import torch.nn as nn
import logging
from typing import Literal


class PredictionHead(nn.Module):
    def __init__(self,
                 in_features: int, n_classes: int,
                 n_hidden: int, d_hidden: int, activation: Literal["relu", "gelu", "sigmoid", "tanh"], dropout: float) -> None:
        super(PredictionHead, self).__init__()
        self.in_features = in_features
        self.n_classes = n_classes
        self.n_hidden = n_hidden
        self.d_hidden = d_hidden
        self.activation = activation
        self.dropout = dropout

        match activation:
            case "relu":
                activation_fn = nn.ReLU()
            case "gelu":
                activation_fn = nn.GELU()
            case "tanh":
                activation_fn = nn.Tanh()
            case "sigmoid":
                activation_fn = nn.Sigmoid()
            case _:
                raise ValueError(f"Unsupported activation type: {activation}")

        if n_hidden == 0:
            self.model = nn.Linear(in_features, n_classes)
        else:
            hidden_blocks = [nn.Linear(d_hidden, d_hidden), activation_fn, nn.Dropout(dropout)] * n_hidden
            self.model = nn.Sequential(
                nn.Linear(in_features, n_hidden),
                nn.Dropout(dropout),
                activation_fn,
                *hidden_blocks,
                nn.Linear(d_hidden, n_classes)
            )

        logging.info(f"Initialized PredictionHead with in_features={in_features}, "
                     f"n_classes={n_classes}, n_hidden={n_hidden}, d_hidden={d_hidden}, "
                     f"activation={activation}")

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
    head = PredictionHead(in_features=512, n_classes=10, n_hidden=2, d_hidden=256, activation="relu", dropout=0.2)
