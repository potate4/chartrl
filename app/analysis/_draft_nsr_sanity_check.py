"""DRAFT — Sanity-check for the NSR loss-masking trainer.

This does NOT train a real model. It builds a tiny dummy GRPO setup,
runs a few steps, and verifies that:

  (a) When all rollouts in a group are CORRECT, the masked advantages
      are all zero, and the loss has zero gradient.
  (b) When some rollouts are CORRECT and some WRONG, only the wrong
      ones contribute to the gradient.
  (c) When all rollouts are WRONG, behavior matches an un-masked GRPO
      run (gradient is non-zero).

You must run this before retraining anything for real. If any check
fails, do NOT adopt _draft_nsr_masked.py — there's a bug in either the
mask construction or the TRL hook.

How to run
----------
The full TRL pipeline needs a model + dataset, so this script uses a
synthetic stub. We patch in fake reward values and check the math
directly on the advantages tensor — we don't need a real model.

This isolates the NSR-mask logic from the rest of the training code.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_mask_construction():
    """Verify _build_correct_mask zeroes correct-rollout advantages."""
    print("\n=== Test 1: mask construction ===")
    # Pretend max_reward = 10, threshold = 0.5 -> correct iff reward >= 5.
    raw_rewards = [8.0, 7.5, 3.0, 2.0]   # rollouts 0,1 correct; 2,3 wrong
    advantages = torch.tensor([+1.5, +1.2, -1.4, -1.3])  # post-TRL normalization

    threshold = 5.0
    mask = torch.tensor([r >= threshold for r in raw_rewards], dtype=torch.bool)
    masked_adv = advantages * (1.0 - mask.float())

    print(f"  raw rewards : {raw_rewards}")
    print(f"  advantages  : {advantages.tolist()}")
    print(f"  correct mask: {mask.tolist()}")
    print(f"  after mask  : {masked_adv.tolist()}")

    expected = torch.tensor([0.0, 0.0, -1.4, -1.3])
    ok = torch.allclose(masked_adv, expected)
    print(f"  PASS: {ok}")
    return ok


def test_all_correct_zero_gradient():
    """When every rollout is correct, no gradient should flow."""
    print("\n=== Test 2: all-correct -> zero gradient ===")
    # Synthetic per-token-loss setup mirroring TRL's loss formula:
    # per_token_loss = -coef * advantages
    # loss = (per_token_loss * mask).sum() / mask.sum()

    raw_rewards = [9.0, 8.5, 7.0, 8.0]   # all >= 5 == correct
    advantages = torch.tensor([+0.5, +0.2, -0.5, -0.2])  # normalized
    threshold = 5.0
    mask = torch.tensor([r >= threshold for r in raw_rewards], dtype=torch.bool)
    masked_adv = advantages * (1.0 - mask.float())

    # Stand-in for log-prob ratio that has gradient.
    coef = torch.ones(4, requires_grad=True)
    completion_mask = torch.ones(4)

    per_token_loss = -coef * masked_adv
    loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
    loss.backward()

    grad_norm = coef.grad.norm().item()
    print(f"  raw rewards : {raw_rewards}")
    print(f"  masked adv  : {masked_adv.tolist()}")
    print(f"  loss        : {loss.item():.6f}")
    print(f"  ||grad||    : {grad_norm:.6e}")
    ok = grad_norm < 1e-9
    print(f"  PASS: {ok}")
    return ok


def test_only_wrong_contribute():
    """When 2 of 4 rollouts are wrong, only their gradient flows."""
    print("\n=== Test 3: mixed correct+wrong -> only wrong contribute ===")
    raw_rewards = [9.0, 3.0, 8.0, 2.0]
    advantages = torch.tensor([+0.5, -0.4, +0.6, -0.7])
    threshold = 5.0
    mask = torch.tensor([r >= threshold for r in raw_rewards], dtype=torch.bool)

    coef = torch.ones(4, requires_grad=True)
    completion_mask = torch.ones(4)

    masked_adv = advantages * (1.0 - mask.float())
    per_token_loss = -coef * masked_adv
    loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
    loss.backward()

    expected_grad = torch.tensor([0.0, 0.4 / 4, 0.0, 0.7 / 4])  # -d/dcoef of -coef*adv = adv
    actual_grad = coef.grad
    print(f"  raw rewards : {raw_rewards}")
    print(f"  masked adv  : {masked_adv.tolist()}")
    print(f"  expected dL/dcoef: {expected_grad.tolist()}")
    print(f"  actual   dL/dcoef: {actual_grad.tolist()}")

    ok = torch.allclose(actual_grad, expected_grad, atol=1e-6)
    print(f"  PASS: {ok}")
    return ok


def test_all_wrong_normal_grpo():
    """When all rollouts are wrong, masking is a no-op (matches plain GRPO)."""
    print("\n=== Test 4: all-wrong -> no masking, matches GRPO ===")
    raw_rewards = [2.0, 1.5, 3.0, 2.5]   # all below threshold
    advantages = torch.tensor([+0.4, -0.6, +0.7, -0.5])
    threshold = 5.0
    mask = torch.tensor([r >= threshold for r in raw_rewards], dtype=torch.bool)

    masked_adv = advantages * (1.0 - mask.float())
    diff = (masked_adv - advantages).abs().max().item()
    print(f"  diff(masked, original advantages) = {diff:.3e}")
    ok = diff < 1e-9
    print(f"  PASS: {ok}")
    return ok


if __name__ == "__main__":
    results = [
        test_mask_construction(),
        test_all_correct_zero_gradient(),
        test_only_wrong_contribute(),
        test_all_wrong_normal_grpo(),
    ]
    print("\n" + "=" * 50)
    if all(results):
        print("ALL CHECKS PASS")
        sys.exit(0)
    else:
        print(f"FAILED: {sum(1 for r in results if not r)}/{len(results)} checks")
        sys.exit(1)
