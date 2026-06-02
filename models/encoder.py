"""
Patch feature extractors.

Each encoder returns a frozen module with a matching PIL transform and a stable
feature dimension. Downstream MIL code only sees tensors of shape (B, D).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Dict, Literal, Optional

import torch
import torch.nn as nn
import torchvision.models as tv_models


EncoderName = Literal["resnet50", "uni", "conch"]

ENCODER_DIMS: Dict[str, int] = {
    "resnet50": 2048,
    "uni": 1024,
    "conch": 512,
}


class EncoderBundle(nn.Module):
    """
    Frozen patch encoder plus the preprocessing transform it expects.

    Args:
        model: torch module that maps a batch of image tensors to (B, D).
        transform: callable that maps a PIL RGB image to a tensor.
        feature_dim: output feature dimension D.
        name: canonical encoder name.
        checkpoint_source: human-readable weight source for HDF5 metadata.
    """

    def __init__(
        self,
        model: nn.Module,
        transform: Callable,
        feature_dim: int,
        name: str,
        checkpoint_source: str,
    ):
        super().__init__()
        self.model = model
        self.transform = transform
        self.feature_dim = feature_dim
        self.name = name
        self.checkpoint_source = checkpoint_source

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        out = self.model(batch)
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out


class _ConchImageEncoder(nn.Module):
    """Adapter so CONCH presents the same forward(batch) API as other encoders."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.model.encode_image(batch, proj_contrast=False, normalize=False)


def list_encoders() -> Dict[str, int]:
    """Return supported encoder names and feature dimensions."""
    return dict(ENCODER_DIMS)


def _resolve_hf_token(hf_token: Optional[str], hf_token_env: str) -> Optional[str]:
    return hf_token or os.environ.get(hf_token_env)


def _maybe_login_hf(token: Optional[str]) -> None:
    if not token:
        return
    try:
        from huggingface_hub import login
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required when passing an HF token. "
            "Install dependencies with `pip install -r requirements.txt`."
        ) from exc
    login(token=token, add_to_git_credential=False)


def _load_state_dict(model: nn.Module, checkpoint_path: str, device: str) -> None:
    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            if key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                break
    if not isinstance(ckpt, dict):
        raise ValueError(f"Checkpoint does not contain a state dict: {checkpoint_path}")

    ckpt = {
        k.removeprefix("module.").removeprefix("model."): v
        for k, v in ckpt.items()
    }
    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    if unexpected:
        print(f"[encoder] Ignored {len(unexpected)} unexpected checkpoint key(s).")
    if missing:
        print(f"[encoder] Missing {len(missing)} checkpoint key(s) during load.")


def _load_resnet50(device: str) -> EncoderBundle:
    weights = tv_models.ResNet50_Weights.IMAGENET1K_V2
    model = tv_models.resnet50(weights=weights)
    model.fc = nn.Identity()
    return EncoderBundle(
        model=model.to(device),
        transform=weights.transforms(),
        feature_dim=ENCODER_DIMS["resnet50"],
        name="resnet50",
        checkpoint_source="torchvision:ResNet50_Weights.IMAGENET1K_V2",
    )


def _load_uni(
    device: str,
    checkpoint_path: Optional[str],
    hf_token: Optional[str],
    hf_token_env: str,
) -> EncoderBundle:
    try:
        import timm
        from timm.data import create_transform, resolve_data_config
    except ImportError as exc:
        raise ImportError("UNI requires `timm`; install dependencies with `pip install -r requirements.txt`.") from exc

    token = _resolve_hf_token(hf_token, hf_token_env)
    _maybe_login_hf(token)

    if checkpoint_path:
        model = timm.create_model(
            "vit_large_patch16_224",
            img_size=224,
            patch_size=16,
            init_values=1e-5,
            num_classes=0,
            dynamic_img_size=True,
        )
        _load_state_dict(model, checkpoint_path, device)
        checkpoint_source = str(Path(checkpoint_path))
    else:
        try:
            model = timm.create_model(
                "hf-hub:MahmoodLab/UNI",
                pretrained=True,
                init_values=1e-5,
                dynamic_img_size=True,
                num_classes=0,
            )
        except Exception as exc:
            raise RuntimeError(
                "Could not load gated UNI weights from Hugging Face. Request access "
                "to MahmoodLab/UNI, log in with HF_TOKEN or pass --hf_token, or "
                "provide --checkpoint_path."
            ) from exc
        checkpoint_source = "hf-hub:MahmoodLab/UNI"

    transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    return EncoderBundle(
        model=model.to(device),
        transform=transform,
        feature_dim=ENCODER_DIMS["uni"],
        name="uni",
        checkpoint_source=checkpoint_source,
    )


def _load_conch(
    device: str,
    checkpoint_path: Optional[str],
    hf_token: Optional[str],
    hf_token_env: str,
) -> EncoderBundle:
    try:
        from conch.open_clip_custom import create_model_from_pretrained
    except ImportError as exc:
        raise ImportError(
            "CONCH requires the CONCH package. Install it with "
            "`pip install git+https://github.com/mahmoodlab/CONCH.git`."
        ) from exc

    token = _resolve_hf_token(hf_token, hf_token_env)
    source = checkpoint_path or "hf_hub:MahmoodLab/conch"
    try:
        model, preprocess = create_model_from_pretrained(
            "conch_ViT-B-16",
            source,
            hf_auth_token=token,
        )
    except Exception as exc:
        raise RuntimeError(
            "Could not load gated CONCH weights. Request access to MahmoodLab/conch, "
            "log in with HF_TOKEN or pass --hf_token, or provide --checkpoint_path."
        ) from exc

    return EncoderBundle(
        model=_ConchImageEncoder(model).to(device),
        transform=preprocess,
        feature_dim=ENCODER_DIMS["conch"],
        name="conch",
        checkpoint_source=source,
    )


def get_encoder(
    name: EncoderName = "resnet50",
    device: str = "cuda",
    checkpoint_path: Optional[str] = None,
    hf_token: Optional[str] = None,
    hf_token_env: str = "HF_TOKEN",
) -> EncoderBundle:
    """
    Load a frozen patch encoder bundle.

    UNI and CONCH are gated models. If Hugging Face loading fails, request
    access on Hugging Face and authenticate, or pass a local checkpoint path.
    """
    name = name.lower()
    if name == "resnet50":
        if checkpoint_path:
            raise ValueError("resnet50 uses torchvision weights and does not accept --checkpoint_path.")
        return _load_resnet50(device)
    if name == "uni":
        return _load_uni(device, checkpoint_path, hf_token, hf_token_env)
    if name == "conch":
        return _load_conch(device, checkpoint_path, hf_token, hf_token_env)
    raise ValueError(f"Unknown encoder: {name}. Supported encoders: {sorted(ENCODER_DIMS)}")
