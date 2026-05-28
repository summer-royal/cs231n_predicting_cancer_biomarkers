"""
Synthetic shape-and-sanity smoke test for CLAM_SB, CLAM_MB, and TumorAwareMIL.

No WSIs or local data required. Run from the repo root:

    python scripts/smoke_test_clam.py

Exits with code 0 on success and prints a summary table of the shapes
each model produces. Failures raise AssertionError.
"""

import sys
from pathlib import Path

import torch

# Allow running as `python scripts/smoke_test_clam.py` from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models import CLAM_MB, CLAM_SB, TumorAwareMIL


def _check_attention_sums(A: torch.Tensor, expected_n_dists: int, tag: str) -> None:
    """A is either (K, 1) [SB] or (C, K) [MB]. Check each distribution ~ 1."""
    if A.dim() == 2 and A.size(-1) == 1:
        sums = A.sum(dim=0)                  # (1,)
    else:
        sums = A.sum(dim=1)                  # (C,)
    assert sums.numel() == expected_n_dists, f"{tag}: expected {expected_n_dists} dists, got {sums.numel()}"
    for i, s in enumerate(sums.tolist()):
        assert abs(s - 1.0) < 1e-4, f"{tag}: attention dist {i} sums to {s:.6f}, not 1.0"


# ---------------------------------------------------------------------------
# Test 1 -- CLAM_SB binary
# ---------------------------------------------------------------------------
def test_clam_sb_binary() -> None:
    K, D = 128, 2048
    x = torch.randn(K, D)
    model = CLAM_SB(in_dim=D, hidden_dim=256, n_classes=2)
    model.eval()
    with torch.no_grad():
        logits, A = model(x)
    assert logits.shape == (1, 2), f"CLAM_SB logits {tuple(logits.shape)} != (1, 2)"
    assert A.shape == (K, 1), f"CLAM_SB attention {tuple(A.shape)} != (K, 1)"
    _check_attention_sums(A, expected_n_dists=1, tag="CLAM_SB")
    print(f"[ok] CLAM_SB       logits={tuple(logits.shape)}  A={tuple(A.shape)}  sum(A)={float(A.sum()):.4f}")


# ---------------------------------------------------------------------------
# Test 2 -- CLAM_MB multi-class (PAM50)
# ---------------------------------------------------------------------------
def test_clam_mb_multiclass() -> None:
    K, D, C = 128, 2048, 5
    x = torch.randn(K, D)
    model = CLAM_MB(in_dim=D, hidden_dim=256, n_classes=C)
    model.eval()
    with torch.no_grad():
        logits, A = model(x)
    assert logits.shape == (1, C), f"CLAM_MB logits {tuple(logits.shape)} != (1, {C})"
    assert A.shape == (C, K), f"CLAM_MB attention {tuple(A.shape)} != ({C}, {K})"
    _check_attention_sums(A, expected_n_dists=C, tag="CLAM_MB")
    print(f"[ok] CLAM_MB       logits={tuple(logits.shape)}  A={tuple(A.shape)}  rowsum~1 for {C} classes")


# ---------------------------------------------------------------------------
# Test 3 -- Small bag edge case
# ---------------------------------------------------------------------------
def test_small_bag() -> None:
    K, D, C = 4, 2048, 5
    x = torch.randn(K, D)

    sb = CLAM_SB(in_dim=D, n_classes=2, k_sample=32)
    mb = CLAM_MB(in_dim=D, n_classes=C, k_sample=32)
    label = torch.tensor([1])

    sb.train(); mb.train()
    # instance_eval=True with k_sample > K // 2 should clamp safely, not crash.
    sb_out = sb(x, label=label, instance_eval=True)
    mb_out = mb(x, label=label, instance_eval=True)
    assert len(sb_out) == 3 and len(mb_out) == 3
    sb_logits, sb_A, sb_inst = sb_out
    mb_logits, mb_A, mb_inst = mb_out
    assert sb_logits.shape == (1, 2)
    assert mb_logits.shape == (1, C)
    assert torch.isfinite(sb_inst).all() and torch.isfinite(mb_inst).all(), \
        "instance loss should be finite even with K=4"
    print(f"[ok] small_bag K=4 sb_inst={sb_inst.item():.4f}  mb_inst={mb_inst.item():.4f}")


# ---------------------------------------------------------------------------
# Test 4 -- TumorAwareMIL forward returns tumor probs
# ---------------------------------------------------------------------------
def test_tumor_aware() -> None:
    K, D, C = 128, 2048, 5
    x = torch.randn(K, D)

    # SB backbone
    sb = CLAM_SB(in_dim=D, n_classes=2)
    wrapped_sb = TumorAwareMIL(backbone=sb, in_dim=D).eval()
    with torch.no_grad():
        out = wrapped_sb(x)
    assert len(out) == 3, f"TumorAware(SB) returned {len(out)}-tuple, want 3"
    logits, A, tumor_probs = out
    assert logits.shape == (1, 2)
    assert A.shape == (K, 1)
    assert tumor_probs.shape == (K, 1), f"tumor_probs {tuple(tumor_probs.shape)} != (K, 1)"
    assert ((tumor_probs >= 0) & (tumor_probs <= 1)).all(), "tumor_probs not in [0, 1]"

    # MB backbone
    mb = CLAM_MB(in_dim=D, n_classes=C)
    wrapped_mb = TumorAwareMIL(backbone=mb, in_dim=D).eval()
    with torch.no_grad():
        logits, A, tumor_probs = wrapped_mb(x)
    assert logits.shape == (1, C)
    assert A.shape == (C, K)
    assert tumor_probs.shape == (K, 1)
    print(f"[ok] TumorAware    p_tumor range=[{float(tumor_probs.min()):.3f}, {float(tumor_probs.max()):.3f}]")


# ---------------------------------------------------------------------------
# Test 5 -- Gradients flow through tumor scorer and instance losses
# ---------------------------------------------------------------------------
def test_gradients() -> None:
    K, D, C = 32, 2048, 5
    x = torch.randn(K, D)
    label = torch.tensor([2])

    mb = CLAM_MB(in_dim=D, n_classes=C, k_sample=4)
    wrapped = TumorAwareMIL(backbone=mb, in_dim=D)
    wrapped.train()

    out = wrapped(x, label=label, instance_eval=True)
    assert len(out) == 4
    logits, _, inst_loss, _ = out

    slide_loss = torch.nn.functional.cross_entropy(logits, label)
    loss = 0.7 * slide_loss + 0.3 * inst_loss
    loss.backward()

    # Scorer must receive non-zero gradient.
    scorer_grads = [p.grad for p in wrapped.tumor_scorer.parameters() if p.grad is not None]
    assert len(scorer_grads) > 0, "tumor_scorer parameters got no gradients"
    grad_norm = sum(float(g.abs().sum()) for g in scorer_grads)
    assert grad_norm > 0, f"tumor_scorer gradients are exactly zero ({grad_norm})"
    print(f"[ok] gradients     tumor_scorer grad_l1={grad_norm:.4f}  inst_loss={inst_loss.item():.4f}")


def main() -> int:
    torch.manual_seed(0)
    test_clam_sb_binary()
    test_clam_mb_multiclass()
    test_small_bag()
    test_tumor_aware()
    test_gradients()
    print("\nAll CLAM smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
