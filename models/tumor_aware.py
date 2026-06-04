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

gate options
  gate_mode  "mult"     -> h * p            (og)
             "residual" -> h * (1 + a*p)    (keeps non-tumor at full strength,
                                             amplifies tumor patches by up to a)
  learn_temp           divide scorer logits by a learnable temperature
  reg_mode   "entropy" push p toward confident 0/1 values
             "budget"  push mean(p) toward `budget`
             "both"    entropy + budget
"""

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn


class TumorAwareMIL(nn.Module):
    """
    Wraps a CLAM (SB or MB) backbone with a patch-level tumor prior.

    Forward pass:
        1. Score each patch with tumor_scorer -> tumor_logits (K, 2).
        2. p_tumor = softmax(tumor_logits / temp)[:, 1:2], shape (K, 1) in [0, 1].
        3. Gate the patch features (mult or residual).
        4. Pass gated features through the CLAM backbone.

    Shapes:
        h            : (K, in_dim)         -- bag of patch embeddings
        tumor_probs  : (K, 1)              -- per-patch tumor probability
        logits       : (1, n_classes)      -- from backbone
        A            : backbone-dependent  -- (K, 1) for CLAM_SB, (C, K) for CLAM_MB
    """

    def __init__(
        self,
        backbone: nn.Module,
        in_dim: int = 1024,
        gate_mode: str = "mult",
        alpha: float = 1.0,
        learn_temp: bool = False,
        reg_mode: str = "none",
        reg_weight: float = 0.0,
        budget: float = 0.25,
    ):
        super().__init__()
        assert gate_mode in ("mult", "residual")
        assert reg_mode in ("none", "entropy", "budget", "both")
        self.backbone = backbone
        self.gate_mode = gate_mode
        self.alpha = alpha
        self.reg_mode = reg_mode
        self.reg_weight = reg_weight
        self.budget = budget

        # tiny neural net outputs [non-tumor, tumor] score -> probability (later))
        # Two-logit scorer kept (softmax over [non-tumor, tumor]).
        # temp on 2-logit softmax = sigmoid of the logit gap / temp
        #                         = single-logit sigmoid gate
        self.tumor_scorer = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(), nn.Linear(128, 2)
        )

        # learnable temp in log space
        self.learn_temp = learn_temp
        if learn_temp:
            self.log_temp = nn.Parameter(torch.zeros(1))

        # last gate reg
        self._reg_loss: Optional[torch.Tensor] = None


    # to avoid lazy 0.5 tumor scorer scores
    # entropy reg pushes tumor probs away from 0.5
    # budget reg keeps active fraction near `budget`
    def _gate_reg(self, p: torch.Tensor) -> torch.Tensor:
        # p: (K,) tumor probs returns scalar reg (alr scaled by reg_weight)
        reg = p.new_zeros(())
        if self.reg_mode in ("none",) or self.reg_weight == 0.0:
            return reg
        eps = 1e-6
        p = p.clamp(eps, 1.0 - eps)
        if self.reg_mode in ("entropy", "both"):
            # bin entropy, minimized -> p commits to 0 or 1
            reg = reg + (-(p * p.log() + (1 - p) * (1 - p).log())).mean()
        if self.reg_mode in ("budget", "both"):
            # keep active fraction near `budget` so can't flag every patch
            reg = reg + (p.mean() - self.budget) ** 2
        return self.reg_weight * reg

    def regularization_loss(self) -> torch.Tensor:
        # scalar tensor from most recent forward (0 if reg disabled)
        if self._reg_loss is None:
            return torch.zeros((), device=next(self.parameters()).device)
        return self._reg_loss

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
        # tumor probability
        tumor_logits = self.tumor_scorer(h)                          # (K, 2)
        if self.learn_temp:
            tumor_logits = tumor_logits / self.log_temp.exp()
        tumor_probs = torch.softmax(tumor_logits, dim=-1)[:, 1:2]    # (K, 1)

        if self.gate_mode == "residual":
            # residual gate preserves weak patches while amplifying high-score ones
            h_gated = h * (1.0 + self.alpha * tumor_probs)
        else:
            h_gated = h * tumor_probs                                # (K, in_dim)

        self._reg_loss = self._gate_reg(tumor_probs.squeeze(-1))

        backbone_out = self.backbone(h_gated, label=label, instance_eval=instance_eval)

        # Tack the tumor probs onto whatever the backbone returns. Callers can
        # inspect them for interpretability without retraining.
        if len(backbone_out) == 2:
            logits, A = backbone_out
            return logits, A, tumor_probs
        logits, A, instance_loss = backbone_out
        return logits, A, instance_loss, tumor_probs
