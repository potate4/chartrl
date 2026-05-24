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

How the queue works
===================
TRL calls the reward function ONCE per group (4 rollouts → 4 raw rewards).
It then calls compute_loss ONCE PER ROLLOUT (batch_size=1 per call).
So we store raw rewards in a deque and popleft() one per compute_loss call.
"""
from __future__ import annotations

from collections import deque
from typing import List, Optional

import torch

from .base_trainer import BaseTrainer, PolicyMethodMixin


class NSRMaskedTrainer(BaseTrainer, PolicyMethodMixin):
    """Negative Sample Reinforcement via per-rollout loss masking."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._raw_reward_queue: deque = deque()

    # ------------------------------------------------------------------
    # 1) Wrap the reward function to enqueue raw per-rollout rewards.
    # ------------------------------------------------------------------
    def _create_reward_function(self):
        base_fn = super()._create_reward_function()

        def reward_fn_capture(completions, **kwargs):
            rewards = base_fn(completions, **kwargs)
            try:
                # Load all rewards for this group into the queue.
                # compute_loss will popleft() one per call.
                for r in rewards:
                    self._raw_reward_queue.append(float(r))
            except Exception:
                pass
            return rewards

        return reward_fn_capture

    # ------------------------------------------------------------------
    # 2) After BaseTrainer creates the TRL trainer, patch compute_loss.
    # ------------------------------------------------------------------
    def _create_trl_trainer(self):
        super()._create_trl_trainer()
        trl_trainer = self._trl_trainer
        outer_self = self
        original_compute_loss = trl_trainer.compute_loss

        def nsr_compute_loss(model, inputs, return_outputs=False, num_items_in_batch=None):
            advantages = inputs.get("advantages")
            if advantages is not None and len(outer_self._raw_reward_queue) > 0:
                b = advantages.size(0)
                # Pop exactly b rewards (normally b==1 per call)
                raw = []
                for _ in range(b):
                    if outer_self._raw_reward_queue:
                        raw.append(outer_self._raw_reward_queue.popleft())
                    else:
                        break

                if len(raw) == b:
                    max_reward = outer_self.get_max_reward(outer_self.config)
                    threshold = outer_self.config.reward_threshold * max_reward
                    correct_mask = torch.tensor(
                        [r >= threshold for r in raw], dtype=torch.bool
                    )
                    inv = (1.0 - correct_mask.to(advantages.dtype).to(advantages.device))

                    if advantages.dim() == 1:
                        inputs["advantages"] = advantages * inv
                    else:
                        inputs["advantages"] = advantages * inv.unsqueeze(-1)

                    n_correct = int(correct_mask.sum().item())
                    outer_self.logger.info(
                        f"[NSR-mask] batch={b}  correct(masked)={n_correct}  wrong(kept)={b - n_correct}"
                    )

            return original_compute_loss(
                model, inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )

        trl_trainer.compute_loss = nsr_compute_loss

    # ------------------------------------------------------------------
    # Legacy interface stub — never called but required by ABC.
    # ------------------------------------------------------------------
    def compute_advantages(self, rewards: List[float]) -> List[float]:
        return list(rewards)
