"""NSR via per-rollout loss masking in TRL's compute_loss.

Why this file exists
====================
The legacy NSRTrainer (nsr_trainer.py) tried to implement NSR by returning
modified values from compute_advantages(). That method is never called:
apply_advantages_in_reward_fn=False in _SHARED, so TRL computes its own
group-normalized advantages regardless of what we return from the reward fn.

This trainer overrides TRL's compute_loss directly and zeroes out advantages
on rollouts whose raw reward >= threshold AFTER TRL has normalized them.
That is the canonical NSR implementation (Zhu et al. 2025, §3.1).

How correct/wrong is determined
================================
A rollout is "correct" iff its raw reward (before TRL normalization) satisfies:
    raw_reward >= config.reward_threshold * get_max_reward(config)

The raw reward vector is captured from the reward function before TRL sees it.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch

from .base_trainer import BaseTrainer, PolicyMethodMixin


class NSRMaskedTrainer(BaseTrainer, PolicyMethodMixin):
    """Negative Sample Reinforcement via per-rollout loss masking."""

    # ------------------------------------------------------------------
    # 1) Wrap the reward function to capture raw per-rollout rewards.
    # ------------------------------------------------------------------
    def _create_reward_function(self):
        base_fn = super()._create_reward_function()

        def reward_fn_capture(completions, **kwargs):
            rewards = base_fn(completions, **kwargs)
            try:
                self._last_raw_rewards = list(map(float, rewards))
            except Exception:
                self._last_raw_rewards = None
            return rewards

        return reward_fn_capture

    # ------------------------------------------------------------------
    # 2) After BaseTrainer creates the TRL trainer, patch compute_loss.
    # ------------------------------------------------------------------
    def _create_trl_trainer(self):
        super()._create_trl_trainer()          # sets self._trl_trainer
        trl_trainer = self._trl_trainer
        outer_self = self
        original_compute_loss = trl_trainer.compute_loss

        def nsr_compute_loss(model, inputs, return_outputs=False, num_items_in_batch=None):
            correct_mask = outer_self._build_correct_mask(inputs.get("advantages"))
            if correct_mask is not None:
                advantages = inputs["advantages"]
                inv = (1.0 - correct_mask.to(advantages.dtype).to(advantages.device))
                if advantages.dim() == 1:
                    inputs["advantages"] = advantages * inv
                else:
                    # (B, T) — broadcast along token dimension
                    inputs["advantages"] = advantages * inv.unsqueeze(-1)

                n = correct_mask.numel()
                k = int(correct_mask.sum().item())
                outer_self.logger.info(
                    f"[NSR-mask] rollouts={n}  correct(masked)={k}  wrong(kept)={n - k}"
                )

            return original_compute_loss(
                model, inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )

        trl_trainer.compute_loss = nsr_compute_loss

    # ------------------------------------------------------------------
    # 3) Build the correct-rollout boolean mask from captured raw rewards.
    # ------------------------------------------------------------------
    def _build_correct_mask(
        self, advantages_tensor: Optional[torch.Tensor]
    ) -> Optional[torch.Tensor]:
        """True where rollout is correct (gradient should be zeroed)."""
        raw = getattr(self, "_last_raw_rewards", None)
        if raw is None or advantages_tensor is None:
            return None

        b = advantages_tensor.size(0)
        if len(raw) != b:
            self.logger.warning(
                f"[NSR-mask] raw_rewards length {len(raw)} != advantages batch {b}; skipping mask"
            )
            return None

        max_reward = self.get_max_reward(self.config)
        threshold = self.config.reward_threshold * max_reward
        return torch.tensor([r >= threshold for r in raw], dtype=torch.bool)

    # ------------------------------------------------------------------
    # Legacy interface stub — never called but required by ABC.
    # ------------------------------------------------------------------
    def compute_advantages(self, rewards: List[float]) -> List[float]:
        return list(rewards)
