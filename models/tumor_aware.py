"""
Tumor-aware MIL wrapper.

The key modification over vanilla CLAM: before attention aggregation, a
lightweight first-pass scorer (trained jointly or with pseudo-labels from a
tile-level classifier) up-weights patches in likely invasive tumor regions.
This reduces confounding from benign stroma, fat, and slide artifacts.

Owner: Jen Ho
TODO: implement TumorAwareMIL.
"""

import torch
import torch.nn as nn


class TumorAwareMIL(nn.Module):
    """
    Wraps a CLAM_SB backbone with a patch-level tumor prior.

    Forward pass:
        1. Score each patch with tumor_scorer -> tumor_logits (N, 2).
        2. Compute tumor probability p_tumor = softmax(tumor_logits)[:, 1].
        3. Gate the patch features: h_gated = h * p_tumor.unsqueeze(-1).
        4. Pass h_gated through the CLAM attention + classifier.
    """

    def __init__(self, backbone: nn.Module, in_dim: int = 1024):
        super().__init__()
        self.backbone = backbone
        self.tumor_scorer = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(), nn.Linear(128, 2)
        )

    def forward(self, h: torch.Tensor):
        raise NotImplementedError("TumorAwareMIL forward pass to be implemented by Jen Ho.")
