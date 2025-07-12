# Search Space for modality II

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
