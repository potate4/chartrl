"""DRAFT — Real NSR loss-masking on top of TRL's GRPOTrainer.

This is a draft. It does NOT replace any existing trainer.
Review, run the sanity check, then decide whether to adopt.

Why this is needed
==================
The current `app/trainers/nsr_trainer.py` implements NSR via a `compute_advantages`
method, but that method is never called: TRL's GRPOTrainer takes the reward
function output and applies its own group-mean+std normalization (see
`scale_rewards: "group"` in our training_args.bin). There is no hook that
lets a subclass swap in its own advantage values.

Verified: trainer_state.json from the existing nsr_baseline run shows
`reward: 8.79, reward_std: 1.06` at step 500 — those are raw stack-of-rewards
magnitudes, not NSR's zero/negative advantage values. NSR's logic never ran.

What this file does
===================
1. Wraps TRL's GRPOTrainer.
2. Wraps the reward function so we capture the raw per-rollout reward vector
   the moment it's produced (before TRL group-normalizes it).
3. In `compute_loss`, builds a binary `correct_mask` from those raw rewards,
   then ZEROS OUT advantages on correct rollouts BEFORE TRL's loss formula
   uses them.

Result: gradient is zero on correct rollouts. Per the NSR paper this is
the canonical implementation (see §3.1 footnote of arXiv:2506.01347 —
"PSR and NSR are implemented by selectively updating the policy model
using only correct or incorrect responses").

How a "correct" rollout is determined
=====================================
We use a reward threshold: rollout i is correct iff
  raw_reward[i] >= reward_threshold * max_reward_estimate

`max_reward_estimate` follows `get_max_reward()` in `base_trainer.py`
(format=2 + accuracy=1 + length=2 + token=2 + type=1 + table=2 +
process=1 + optional HCPC + optional CLC = ~11 base, +HCPC if enabled).

`reward_threshold` is the existing `config.reward_threshold` (default 0.5).

This matches the threshold used in the existing (dead) nsr_trainer.compute_advantages.

Where to wire it in (not yet done)
==================================
After review, the way to adopt this is:
  - In `app/trainers/__init__.py`, route `policy_method="nsr"` to this class
    instead of NSRTrainer (or replace NSRTrainer's body with this).
  - Set `apply_advantages_in_reward_fn=False` (already the default).
  - Run the sanity check below.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
from trl import GRPOTrainer as TRLGRPOTrainer

from .base_trainer import BaseTrainer, PolicyMethodMixin


class NSRMaskedTrainer(BaseTrainer, PolicyMethodMixin):
    """Negative Sample Reinforcement via per-rollout loss masking.

    Unlike the legacy NSRTrainer (which tried and failed to override
    advantages through the reward function), this subclass overrides
    TRL's `compute_loss` directly. It works regardless of TRL's
    `scale_rewards` setting — we modify advantages AFTER TRL has
    normalized them, on whichever subset of rollouts we choose.
    """

    # ------------------------------------------------------------------
    # 1) Capture raw per-rollout rewards from the reward function.
    # ------------------------------------------------------------------
    def _create_reward_function(self):
        """Wrap base reward function to stash raw rewards on `self`.

        This is the only place we can see the per-rollout raw reward
        vector before TRL's group normalization mangles it.
        """
        base_fn = super()._create_reward_function()

        def reward_fn_capture(completions, **kwargs):
            rewards = base_fn(completions, **kwargs)
            # Stash for compute_loss. TRL invokes reward_fn once per
            # generate-and-score batch in batch order, so the order matches
            # the advantages tensor we'll see later.
            try:
                self._last_raw_rewards = list(map(float, rewards))
            except Exception:
                self._last_raw_rewards = None
            return rewards

        return reward_fn_capture

    # ------------------------------------------------------------------
    # 2) Build a custom TRL trainer subclass that masks advantages.
    # ------------------------------------------------------------------
    def _create_trl_trainer(self):
        # First let BaseTrainer build the TRL trainer normally.
        super()._create_trl_trainer()

        # Then monkey-patch its compute_loss to apply NSR masking.
        # We patch the instance method (not the class) so the change is
        # local to this trainer.
        trl_trainer = self._trl_trainer
        outer_self = self

        original_compute_loss = trl_trainer.compute_loss

        def nsr_compute_loss(model, inputs, return_outputs=False, num_items_in_batch=None):
            # Build the correct/wrong mask from raw rewards captured by
            # the wrapped reward function.
            correct_mask = outer_self._build_correct_mask(inputs.get("advantages"))
            if correct_mask is not None:
                advantages = inputs["advantages"]
                # Multiply by (1 - correct_mask) — zero on correct rollouts,
                # unchanged on wrong rollouts. Shape: (B,) or (B, T).
                if advantages.dim() == 1:
                    inputs["advantages"] = advantages * (1.0 - correct_mask.to(advantages.dtype).to(advantages.device))
                else:
                    # (B, T) — broadcast along time dim.
                    m = correct_mask.to(advantages.dtype).to(advantages.device)
                    inputs["advantages"] = advantages * (1.0 - m).unsqueeze(-1)

                # Log how much we masked, for the train.log.
                n = correct_mask.numel()
                k = int(correct_mask.sum().item())
                outer_self.logger.info(
                    f"[NSR-mask] rollouts={n}  correct(masked)={k}  wrong(kept)={n - k}"
                )

            return original_compute_loss(
                model, inputs, return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )

        trl_trainer.compute_loss = nsr_compute_loss

    # ------------------------------------------------------------------
    # 3) Determine which rollouts are "correct".
    # ------------------------------------------------------------------
    def _build_correct_mask(self, advantages_tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Returns a boolean tensor of shape (B,) where True == correct rollout.

        Uses the raw rewards stashed by the wrapped reward function.
        """
        raw = getattr(self, "_last_raw_rewards", None)
        if raw is None or advantages_tensor is None:
            return None

        # Length sanity: advantages_tensor is (B,) or (B, T) where B = num_rollouts.
        b = advantages_tensor.size(0)
        if len(raw) != b:
            self.logger.warning(
                f"[NSR-mask] raw_rewards length {len(raw)} != advantages B {b}; skipping mask"
            )
            return None

        max_reward = self.get_max_reward(self.config)
        threshold = self.config.reward_threshold * max_reward

        return torch.tensor([r >= threshold for r in raw], dtype=torch.bool)

    # ------------------------------------------------------------------
    # The legacy compute_advantages stub (kept for interface compatibility,
    # never called).
    # ------------------------------------------------------------------
    def compute_advantages(self, rewards: List[float]) -> List[float]:
        return list(rewards)
