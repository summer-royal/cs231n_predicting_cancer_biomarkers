"""
Tumor-aware MIL wrapper.

The key modification over vanilla CLAM: before attention aggregation, a
lightweight first-pass scorer up-weights patches in likely invasive tumor
regions. This reduces confounding from benign stroma, fat, and slide
artifacts.

Owner: Jen Ho

The tumor scorer is trained jointly with the CLAM backbone via the
slide-level loss — gradients flow back through the gated features. No
patch-level tumor labels are required; the scorer learns a coarse
tumor-vs-not prior implicitly from the slide-level signal.
"""

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn


class TumorAwareMIL(nn.Module):
    """
    Wraps a CLAM (SB or MB) backbone with a patch-level tumor prior.

    Forward pass:
        1. Score each patch with tumor_scorer -> tumor_logits (K, 2).
        2. Compute tumor probability p_tumor = softmax(tumor_logits)[:, 1:2].
           Shape (K, 1), in [0, 1], differentiable wrt scorer params.
        3. Gate the patch features: h_gated = h * p_tumor (broadcast over D).
        4. Pass h_gated through the CLAM backbone.

    Shapes:
        h            : (K, in_dim)         -- bag of patch embeddings
        tumor_probs  : (K, 1)              -- per-patch tumor probability
        logits       : (1, n_classes)      -- from backbone
        A            : backbone-dependent  -- (K, 1) for CLAM_SB, (C, K) for CLAM_MB
    """

    def __init__(self, backbone: nn.Module, in_dim: int = 1024):
        super().__init__()
        self.backbone = backbone
        # Two-output scorer kept for backward compatibility with the original
        # docstring (softmax over [non-tumor, tumor]). Equivalent in capacity
        # to a 1-output sigmoid scorer and avoids changing the existing API.
        self.tumor_scorer = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(), nn.Linear(128, 2)
        )

    def forward(
        self,
        h: torch.Tensor,
        label: Optional[torch.Tensor] = None,
        instance_eval: bool = False,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        # h: (K, in_dim)   (also accepts (1, K, in_dim))
        if h.dim() == 3 and h.size(0) == 1:
            h = h.squeeze(0)

        tumor_logits = self.tumor_scorer(h)                          # (K, 2)
        tumor_probs = torch.softmax(tumor_logits, dim=-1)[:, 1:2]    # (K, 1)
        h_gated = h * tumor_probs                                    # (K, in_dim)

        backbone_out = self.backbone(h_gated, label=label, instance_eval=instance_eval)

        # Tack the tumor probs onto whatever the backbone returns. Callers can
        # inspect them for interpretability without retraining.
        if len(backbone_out) == 2:
            logits, A = backbone_out
            return logits, A, tumor_probs
        logits, A, instance_loss = backbone_out
        return logits, A, instance_loss, tumor_probs
