"""
Patch feature extractor.

Wraps a pretrained CNN or pathology foundation model.
Supported: ResNet-50 (ImageNet), UNI (pathology ViT), CONCH.
Owner: Luke Zhao

TODO: add UNI / CONCH loading once access is granted.
"""

import torch
import torch.nn as nn
import torchvision.models as tv_models
from typing import Literal


EncoderName = Literal["resnet50", "uni", "conch"]


def get_encoder(name: EncoderName = "resnet50", device: str = "cuda") -> nn.Module:
    if name == "resnet50":
        model = tv_models.resnet50(weights=tv_models.ResNet50_Weights.IMAGENET1K_V2)
        model.fc = nn.Identity()  # output: (B, 2048)
    elif name in ("uni", "conch"):
        raise NotImplementedError(f"{name} loader to be implemented by Luke Zhao.")
    else:
        raise ValueError(f"Unknown encoder: {name}")

    return model.eval().to(device)
