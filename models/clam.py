"""
CLAM: Clustering-constrained Attention Multiple Instance Learning.

Reference: Lu et al., Nature Biomedical Engineering 2021.
Owner: Jen Ho

TODO: implement CLAM_SB (single branch) and CLAM_MB (multi-branch).
"""

import torch
import torch.nn as nn
from typing import Tuple


class AttentionNet(nn.Module):
    """Gated attention network that scores each patch."""

    def __init__(self, in_dim: int = 1024, hidden_dim: int = 256):
        super().__init__()
        self.attention_V = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.Sigmoid())
        self.attention_W = nn.Linear(hidden_dim, 1)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # h: (N, in_dim)
        A = self.attention_W(self.attention_V(h) * self.attention_U(h))  # (N, 1)
        A = torch.softmax(A, dim=0)
        M = (A * h).sum(dim=0, keepdim=True)  # (1, in_dim)
        return M, A


class CLAM_SB(nn.Module):
    """Single-branch CLAM for binary biomarker prediction."""

    def __init__(self, in_dim: int = 1024, hidden_dim: int = 256, n_classes: int = 2):
        super().__init__()
        self.feature_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.25)
        )
        self.attention = AttentionNet(hidden_dim, hidden_dim // 2)
        self.classifier = nn.Linear(hidden_dim, n_classes)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.feature_proj(h)
        M, A = self.attention(h)
        logits = self.classifier(M)
        return logits, A


class CLAM_MB(nn.Module):
    """Multi-branch CLAM for multi-class (PAM50 subtype) prediction. TODO."""

    def __init__(self, in_dim: int = 1024, hidden_dim: int = 256, n_classes: int = 5):
        super().__init__()
        raise NotImplementedError("CLAM_MB to be implemented by Jen Ho.")
