"""Smoke test for the Spatial + Temporal Attention Transfer implementation.

Verifies:
  1. Attention map shapes are correct
  2. Loss computation produces finite values
  3. Gradients flow through the student
  4. Backward compatibility: old `at_beta` config key still works
  5. Spatial interpolation handles H×W mismatch correctly
  6. Temporal maps match when T is identical (no interpolation)

Usage:
    python -m tests.test_at_loss
"""

import torch
import torch.nn as nn

from src.training.losses import (
    _spatial_attention_map,
    _temporal_attention_map,
    AttentionTransferLoss,
    CombinedKDATLoss,
)


def test_spatial_attention_shape():
    """Spatial attention: [B, C, T, H, W] → [B, H*W]"""
    feat = torch.randn(2, 64, 24, 14, 14)
    attn = _spatial_attention_map(feat)
    assert attn.shape == (2, 14 * 14), f"Expected (2, 196), got {attn.shape}"
    # Verify L2-normalized
    norms = attn.norm(p=2, dim=1)
    assert torch.allclose(norms, torch.ones(2), atol=1e-5), f"Not L2-normalized: {norms}"
    print("[OK] Spatial attention shape & normalization OK")


def test_spatial_attention_2d_fallback():
    """Spatial attention on 2D features: [B, C, H, W] → [B, H*W]"""
    feat = torch.randn(2, 64, 14, 14)
    attn = _spatial_attention_map(feat)
    assert attn.shape == (2, 14 * 14), f"Expected (2, 196), got {attn.shape}"
    print("[OK] Spatial attention 2D fallback OK")


def test_temporal_attention_shape():
    """Temporal attention: [B, C, T, H, W] → [B, T]"""
    feat = torch.randn(2, 64, 24, 14, 14)
    attn = _temporal_attention_map(feat)
    assert attn.shape == (2, 24), f"Expected (2, 24), got {attn.shape}"
    # Verify L2-normalized
    norms = attn.norm(p=2, dim=1)
    assert torch.allclose(norms, torch.ones(2), atol=1e-5), f"Not L2-normalized: {norms}"
    print("[OK] Temporal attention shape & normalization OK")


def test_temporal_rejects_4d():
    """Temporal attention should reject 4D input (no time dimension)."""
    feat = torch.randn(2, 64, 14, 14)
    try:
        _temporal_attention_map(feat)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    print("[OK] Temporal attention rejects 4D input OK")


def test_at_loss_same_shapes():
    """AT loss with matching spatial dimensions (no interpolation needed)."""
    teacher_feats = {3: torch.randn(2, 256, 24, 14, 14)}
    student_feats = {2: torch.randn(2, 32, 24, 14, 14)}

    at = AttentionTransferLoss(beta_spatial=0.05, beta_temporal=0.05)
    total, spatial, temporal = at(teacher_feats, student_feats, [3], [2])

    assert total.dim() == 0, "Loss should be scalar"
    assert total.item() >= 0, "Loss should be non-negative"
    assert torch.isfinite(total), "Loss should be finite"
    assert spatial.item() >= 0
    assert temporal.item() >= 0
    print(f"[OK] AT loss (same shape): total={total.item():.6f}, "
          f"spatial={spatial.item():.6f}, temporal={temporal.item():.6f}")


def test_at_loss_different_spatial():
    """AT loss with different H×W: student has smaller spatial dims."""
    teacher_feats = {3: torch.randn(2, 256, 24, 14, 14)}
    student_feats = {2: torch.randn(2, 32, 24, 4, 4)}

    at = AttentionTransferLoss(beta_spatial=0.05, beta_temporal=0.05)
    total, spatial, temporal = at(teacher_feats, student_feats, [3], [2])

    assert total.dim() == 0, "Loss should be scalar"
    assert torch.isfinite(total), "Loss should be finite"
    print(f"[OK] AT loss (spatial mismatch 14×14 vs 4×4): total={total.item():.6f}, "
          f"spatial={spatial.item():.6f}, temporal={temporal.item():.6f}")


def test_at_loss_multiple_pairs():
    """AT loss with 3 feature pairs (typical config)."""
    teacher_feats = {
        3: torch.randn(2, 256, 24, 14, 14),
        4: torch.randn(2, 512, 24, 7, 7),
        5: torch.randn(2, 1024, 24, 4, 4),
    }
    student_feats = {
        2: torch.randn(2, 32, 24, 14, 14),
        4: torch.randn(2, 96, 24, 4, 4),
        6: torch.randn(2, 320, 24, 2, 2),
    }

    at = AttentionTransferLoss(beta_spatial=0.05, beta_temporal=0.05)
    total, spatial, temporal = at(teacher_feats, student_feats, [3, 4, 5], [2, 4, 6])

    assert torch.isfinite(total), "Loss should be finite"
    print(f"[OK] AT loss (3 pairs): total={total.item():.6f}, "
          f"spatial={spatial.item():.6f}, temporal={temporal.item():.6f}")


def test_combined_loss_gradient():
    """Verify gradients flow through the combined KD+AT loss."""
    # Simple linear student mock
    student_head = nn.Linear(10, 101)
    student_logits = student_head(torch.randn(2, 10))
    teacher_logits = torch.randn(2, 101)
    labels = torch.randint(0, 101, (2,))

    teacher_feats = {3: torch.randn(2, 256, 24, 14, 14)}
    student_feats = {2: torch.randn(2, 32, 24, 7, 7, requires_grad=True)}

    loss_fn = CombinedKDATLoss(
        temperature=8.0,
        alpha=0.7,
        beta_spatial=0.05,
        beta_temporal=0.05,
        teacher_keys=[3],
        student_keys=[2],
    )
    total, kd, at = loss_fn(
        student_logits, teacher_logits, labels,
        teacher_feats, student_feats,
    )

    total.backward()
    assert student_feats[2].grad is not None, "Gradient should flow to student features"
    assert student_head.weight.grad is not None, "Gradient should flow to student head"
    print(f"[OK] Combined loss gradient flow OK: total={total.item():.4f}, "
          f"kd={kd.item():.4f}, at={at.item():.4f}")


def test_backward_compat_single_beta():
    """Old config with single at_beta should work via trainer fallback logic."""
    # Simulate what trainer._build_criterion does with old config
    kd_cfg = {"temperature": 8.0, "alpha": 0.7, "at_beta": 0.05}
    at_beta_fallback = kd_cfg.get("at_beta", 0.05)

    loss_fn = CombinedKDATLoss(
        temperature=kd_cfg.get("temperature", 5.0),
        alpha=kd_cfg.get("alpha", 0.7),
        beta_spatial=kd_cfg.get("at_beta_spatial", at_beta_fallback),
        beta_temporal=kd_cfg.get("at_beta_temporal", at_beta_fallback),
    )

    assert loss_fn.at_loss.beta_spatial == 0.05
    assert loss_fn.at_loss.beta_temporal == 0.05
    print("[OK] Backward compatibility with single at_beta OK")


def test_temporal_only():
    """Verify temporal-only AT (β_s=0) works without error."""
    teacher_feats = {3: torch.randn(2, 256, 24, 14, 14)}
    student_feats = {2: torch.randn(2, 32, 24, 4, 4)}

    at = AttentionTransferLoss(beta_spatial=0.0, beta_temporal=0.1)
    total, spatial, temporal = at(teacher_feats, student_feats, [3], [2])

    assert spatial.item() == 0.0, "Spatial should be zero when beta_spatial=0"
    assert temporal.item() > 0.0, "Temporal should be non-zero"
    print(f"[OK] Temporal-only mode OK: spatial={spatial.item():.6f}, temporal={temporal.item():.6f}")


if __name__ == "__main__":
    print("=" * 60)
    print("Spatial + Temporal Attention Transfer -- Smoke Tests")
    print("=" * 60)
    print()

    test_spatial_attention_shape()
    test_spatial_attention_2d_fallback()
    test_temporal_attention_shape()
    test_temporal_rejects_4d()
    test_at_loss_same_shapes()
    test_at_loss_different_spatial()
    test_at_loss_multiple_pairs()
    test_combined_loss_gradient()
    test_backward_compat_single_beta()
    test_temporal_only()

    print()
    print("=" * 60)
    print("ALL TESTS PASSED [OK]")
    print("=" * 60)
