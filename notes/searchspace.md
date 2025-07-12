# Search Space for modality II

## Space partitioning by image resolution

* **Idea**: Split search space into three partitions: Small (<28x28 images), Medium (28x28 to 224x224), Large (>224x224) images
- For each partition, define a set of backbone architectures that are suitable for the image size and complexity.

* **Motivation/Justification**:
Because model capacity should match data complexity, we explicitly gate our architecture search by image resolution. Small images (≤32×32) only trigger tiny CNNs and micro-ResNets, medium images (33–128px) only trigger mid-size backbones (ResNet-18/34, EfficientNet-B0/B1), and large images (>128px) unlock deeper networks or ViTs. This biases our AutoML toward the right inductive regime, reduces the search space, and dramatically improves convergence under a fixed compute budget, while still leaving room for full flexibility when data warrant it.

## TODO: Adjust bottom part to match partitioning

## Backbone Architectures
* **ResNet**
  * Types: 18, 34, 50
  * Weights: Random, ImageNet, DiNO
  * Number of trainable Layers

* **ViT**
  * Types: Small, Base
  * Weights: Random, ImageNet, DiNOv2
  * Number of trainable Layers

## Prediction Head

* **MLP**
  * Hidden Layers: 0, 1, 2, 3
  * Hidden Units: 128, 256, 512
  * Activation Functions: ReLU, GELU

## Data & Preprocessing

* **Normalization**
  * Standardization: Mean and standard deviation normalization
  * Min-Max Scaling: Rescaling to [0, 1]

* **Augmentations**:
  * Flip probability
  * Rotation range (degrees)
  * Scale/crop ranges
  * Color jitter (brightness, contrast, saturation, hue)
  * CutMix / MixUp probabilities & α parameters
  * Random erasing probability & area fraction
  * Policy: AutoAugment, RandAugment

## Training Parameters

* LR-Range
* LR-Scheduler
* Batch Size
* Optimizers
* Weight Decay
* Momentum (if SGD)
* Epochs
* CE-Loss: label smoothing, class imbalance handling
* Gradient clipping (max norm)
