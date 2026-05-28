"""
CLAM: Clustering-constrained Attention Multiple Instance Learning.

Reference: Lu et al., Nature Biomedical Engineering 2021.
Owner: Jen Ho

CLAM is NOT a Transformer. Each WSI is a bag of patch embeddings of shape
[K, D]. We project to a hidden dim, score each patch with a gated attention
network, softmax over patches, take a weighted sum to get one slide-level
embedding (per class for the multi-branch variant), and classify.

Shapes used throughout this file:
    K           = number of patches in the bag (variable per slide)
    D = in_dim  = encoder feature dim (e.g., 2048 for ResNet-50, 1024 for UNI)
    H = hidden_dim
    C = n_classes
"""

from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Attention modules
# ---------------------------------------------------------------------------


class AttentionNet(nn.Module):
    """Single-branch gated attention. One attention score per patch."""

    def __init__(self, in_dim: int = 1024, hidden_dim: int = 256):
        super().__init__()
        self.attention_V = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.Sigmoid())
        self.attention_W = nn.Linear(hidden_dim, 1)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # h: (K, in_dim)
        A = self.attention_W(self.attention_V(h) * self.attention_U(h))  # (K, 1)
        A = torch.softmax(A, dim=0)                                       # softmax over patches
        M = (A * h).sum(dim=0, keepdim=True)                              # (1, in_dim)
        return M, A


class AttentionNetMultiBranch(nn.Module):
    """
    Multi-branch gated attention: one attention distribution per class.

    The V and U gates are shared across classes (cheap, common in CLAM_MB);
    only the final W projection has per-class outputs.
    """

    def __init__(self, in_dim: int = 1024, hidden_dim: int = 256, n_classes: int = 5):
        super().__init__()
        self.attention_V = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.Sigmoid())
        self.attention_W = nn.Linear(hidden_dim, n_classes)  # one logit per class per patch

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # h: (K, in_dim)
        A = self.attention_W(self.attention_V(h) * self.attention_U(h))  # (K, C)
        A = A.transpose(0, 1)                                             # (C, K)
        A = torch.softmax(A, dim=1)                                       # softmax over patches per class
        M = A @ h                                                         # (C, in_dim)
        return M, A


# ---------------------------------------------------------------------------
# Helpers for instance-level clustering loss
# ---------------------------------------------------------------------------


def _safe_topk_indices(scores: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return (top_idx, bot_idx) of length k each from a 1-D `scores` tensor.

    `k` is clamped to floor(K/2) so top and bottom selections never overlap,
    and to at least 1 (caller is responsible for skipping when K < 2).
    """
    K = scores.numel()
    k_eff = max(1, min(k, K // 2))
    top_idx = torch.topk(scores, k_eff, largest=True).indices
    bot_idx = torch.topk(scores, k_eff, largest=False).indices
    return top_idx, bot_idx


# ---------------------------------------------------------------------------
# CLAM single-branch
# ---------------------------------------------------------------------------


class CLAM_SB(nn.Module):
    """
    Single-branch CLAM for binary biomarker prediction (e.g., ER status).

    Forward returns `(logits, A)` by default for backward compatibility.
    Pass `instance_eval=True` along with a `label` to also receive the
    instance-clustering auxiliary loss as a third tuple element.

    Args:
        in_dim:         encoder feature dim D (2048 for ResNet-50, 1024 for UNI).
        hidden_dim:     projected feature dim H.
        n_classes:      number of slide-level classes (default 2 for binary).
        dropout:        dropout after the projection MLP.
        k_sample:       number of pseudo-pos / pseudo-neg patches per class
                        for the instance clustering loss.
        subtyping:      if True, also penalize other classes' top-k as
                        pseudo-negatives (CLAM "subtyping" mode).
    """

    def __init__(
        self,
        in_dim: int = 1024,
        hidden_dim: int = 256,
        n_classes: int = 2,
        dropout: float = 0.25,
        k_sample: int = 8,
        subtyping: bool = False,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.k_sample = k_sample
        self.subtyping = subtyping

        self.feature_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)
        )
        self.attention = AttentionNet(hidden_dim, hidden_dim // 2)
        self.classifier = nn.Linear(hidden_dim, n_classes)
        # One binary instance-level head per slide-level class. Operates on
        # projected (hidden_dim) features, not raw encoder features.
        self.instance_classifiers = nn.ModuleList(
            [nn.Linear(hidden_dim, 2) for _ in range(n_classes)]
        )

    def forward(
        self,
        h: torch.Tensor,
        label: Optional[torch.Tensor] = None,
        instance_eval: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor],
               Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        # h: (K, in_dim)   (callers may also pass (1, K, in_dim); we squeeze.)
        if h.dim() == 3 and h.size(0) == 1:
            h = h.squeeze(0)

        h_proj = self.feature_proj(h)                       # (K, H)
        M, A = self.attention(h_proj)                       # M: (1, H),  A: (K, 1)
        logits = self.classifier(M)                         # (1, n_classes)

        if not instance_eval:
            return logits, A

        instance_loss = self._instance_loss_sb(h_proj, A, label)
        return logits, A, instance_loss

    def _instance_loss_sb(
        self, h_proj: torch.Tensor, A: torch.Tensor, label: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Top-k / bottom-k pseudo-supervision on the projected features."""
        device = h_proj.device
        if label is None or h_proj.size(0) < 2:
            return torch.zeros((), device=device)

        cls = int(label.item()) if label.numel() == 1 else int(label[0].item())
        scores = A.squeeze(-1)                              # (K,)
        top_idx, bot_idx = _safe_topk_indices(scores, self.k_sample)

        clf = self.instance_classifiers[cls]
        pos_logits = clf(h_proj[top_idx])                   # (k, 2)
        neg_logits = clf(h_proj[bot_idx])                   # (k, 2)

        targets = torch.cat(
            [torch.ones(top_idx.numel(), dtype=torch.long, device=device),
             torch.zeros(bot_idx.numel(), dtype=torch.long, device=device)]
        )
        logits = torch.cat([pos_logits, neg_logits], dim=0)
        loss = F.cross_entropy(logits, targets)

        if self.subtyping:
            # For every other class, treat its own top-k as pseudo-negatives
            # under that class's instance head. Used when SB is run on a
            # multi-class label (rare; included for symmetry with MB).
            for other in range(self.n_classes):
                if other == cls:
                    continue
                top_other, _ = _safe_topk_indices(scores, self.k_sample)
                other_logits = self.instance_classifiers[other](h_proj[top_other])
                other_targets = torch.zeros(
                    top_other.numel(), dtype=torch.long, device=device
                )
                loss = loss + F.cross_entropy(other_logits, other_targets)
        return loss


# ---------------------------------------------------------------------------
# CLAM multi-branch
# ---------------------------------------------------------------------------


class CLAM_MB(nn.Module):
    """
    Multi-branch CLAM for multi-class slide-level tasks (e.g., PAM50 subtype).

    One attention distribution per class, one slide-level embedding per class,
    one binary linear classifier per class. The C class-specific logits are
    stacked into a single slide-level logit vector of shape (1, C).

    Forward signature mirrors `CLAM_SB`:
        logits, A = model(h)
        logits, A, instance_loss = model(h, label=y, instance_eval=True)

    Shapes:
        h          : (K, in_dim)              -- bag of patch embeddings
        logits     : (1, n_classes)           -- slide-level logits
        A          : (n_classes, K)           -- per-class attention,
                                                 rows sum to 1 across patches
    """

    def __init__(
        self,
        in_dim: int = 1024,
        hidden_dim: int = 256,
        n_classes: int = 5,
        dropout: float = 0.25,
        k_sample: int = 8,
        subtyping: bool = True,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.k_sample = k_sample
        self.subtyping = subtyping

        self.feature_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)
        )
        self.attention = AttentionNetMultiBranch(
            in_dim=hidden_dim, hidden_dim=hidden_dim // 2, n_classes=n_classes
        )
        # One scalar linear classifier per class branch; concatenated -> (1, C).
        self.classifiers = nn.ModuleList(
            [nn.Linear(hidden_dim, 1) for _ in range(n_classes)]
        )
        self.instance_classifiers = nn.ModuleList(
            [nn.Linear(hidden_dim, 2) for _ in range(n_classes)]
        )

    def forward(
        self,
        h: torch.Tensor,
        label: Optional[torch.Tensor] = None,
        instance_eval: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor],
               Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        # h: (K, in_dim)   (also accepts (1, K, in_dim))
        if h.dim() == 3 and h.size(0) == 1:
            h = h.squeeze(0)

        h_proj = self.feature_proj(h)                       # (K, H)
        M, A = self.attention(h_proj)                       # M: (C, H),  A: (C, K)

        # One slide-level logit per class branch (each classifier sees its own M row).
        per_class_logits = [
            self.classifiers[c](M[c : c + 1]) for c in range(self.n_classes)
        ]                                                   # list of (1, 1)
        logits = torch.cat(per_class_logits, dim=1)         # (1, n_classes)

        if not instance_eval:
            return logits, A

        instance_loss = self._instance_loss_mb(h_proj, A, label)
        return logits, A, instance_loss

    def _instance_loss_mb(
        self, h_proj: torch.Tensor, A: torch.Tensor, label: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        For the ground-truth class c*:
          - top-k by A[c*] are pseudo-positives for instance_classifiers[c*]
          - bottom-k by A[c*] are pseudo-negatives for instance_classifiers[c*]

        For subtyping, for every other class c != c*:
          - top-k by A[c] are pseudo-negatives for instance_classifiers[c]
        """
        device = h_proj.device
        if label is None or h_proj.size(0) < 2:
            return torch.zeros((), device=device)

        cls = int(label.item()) if label.numel() == 1 else int(label[0].item())
        loss = torch.zeros((), device=device)

        # In-class: positives + negatives.
        scores_true = A[cls]                                # (K,)
        top_idx, bot_idx = _safe_topk_indices(scores_true, self.k_sample)
        clf_true = self.instance_classifiers[cls]
        in_logits = torch.cat(
            [clf_true(h_proj[top_idx]), clf_true(h_proj[bot_idx])], dim=0
        )
        in_targets = torch.cat(
            [torch.ones(top_idx.numel(), dtype=torch.long, device=device),
             torch.zeros(bot_idx.numel(), dtype=torch.long, device=device)]
        )
        loss = loss + F.cross_entropy(in_logits, in_targets)

        if self.subtyping:
            for other in range(self.n_classes):
                if other == cls:
                    continue
                top_other, _ = _safe_topk_indices(A[other], self.k_sample)
                other_logits = self.instance_classifiers[other](h_proj[top_other])
                other_targets = torch.zeros(
                    top_other.numel(), dtype=torch.long, device=device
                )
                loss = loss + F.cross_entropy(other_logits, other_targets)
        return loss
