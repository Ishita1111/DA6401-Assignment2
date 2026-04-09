"""Model package exports for Assignment-2 skeleton.

Import from this package in training/inference scripts to keep paths stable.
"""

from .layers import CustomDropout
from .localization import VGG11Localizer
from .classification import VGG11Classifier
from .segmentation import VGG11UNet
from .vgg11 import VGG11Encoder

# Optional: `MultiTaskPerceptionModel` depends on `gdown` for checkpoint download.
# Keep core model imports usable even when `gdown` isn't installed.
try:
    from .multitask import MultiTaskPerceptionModel
except ModuleNotFoundError:
    MultiTaskPerceptionModel = None

__all__ = [
    "CustomDropout",
    "VGG11Classifier",
    "VGG11Encoder",
    "VGG11Localizer",
    "VGG11UNet",
    "MultiTaskPerceptionModel",
]
