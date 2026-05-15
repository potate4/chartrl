"""NSR + HCPC Trainer.

Implements NSR (zero gradient on correct rollouts) via the same loss-masking
mechanism as NSRMaskedTrainer, with HCPC reward enabled on top.

The HCPC reward is already computed inside RewardAggregator when
config.rewards.use_hcpc=True — no extra code needed here for that part.
What this class adds over NSRMaskedTrainer:

1. Captures rollout texts so _compute_wrong_inconsistency() can read them
   (used only for logging; the penalty multiplier on wrong rollouts is an
   optional enhancement — see compute_advantages docstring).
2. Applies the same loss-masking (inherited via _create_trl_trainer path).

Architecture note
-----------------
compute_advantages() is still dead code (apply_advantages_in_reward_fn=False).
The real NSR effect comes from the compute_loss patch in NSRMaskedTrainer,
which this class inherits by calling super()._create_trl_trainer().
"""
from __future__ import annotations

from typing import List

from .nsr_masked_trainer import NSRMaskedTrainer
from .base_trainer import PolicyMethodMixin


class NSRHCPCTrainer(NSRMaskedTrainer, PolicyMethodMixin):
    """NSR loss-masking + HCPC reward shaping.

    Inherits all NSR loss-masking logic from NSRMaskedTrainer.
    HCPC reward is activated by setting config.rewards.use_hcpc=True,
    which RewardAggregator picks up automatically.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_rollouts: list = []

    # ------------------------------------------------------------------
    # Override reward capture to also stash normalized completion texts
    # for optional inconsistency logging.
    # ------------------------------------------------------------------
    def _create_reward_function(self):
        # NSRMaskedTrainer._create_reward_function wraps BaseTrainer's fn
        # and captures self._last_raw_rewards. We wrap that further to
        # also capture normalized completion strings.
        base_fn = super()._create_reward_function()

        def reward_fn_with_rollout_capture(completions, **kwargs):
            # Normalize to strings — same logic as BaseTrainer._normalize_completion
            normalized = []
            for c in completions:
                if isinstance(c, list):
                    for msg in c:
                        if isinstance(msg, dict) and msg.get("role") == "assistant":
                            content = msg.get("content", "")
                            if isinstance(content, list):
                                text_parts = [p.get("text", "") for p in content if p.get("type") == "text"]
                                normalized.append(" ".join(text_parts))
                            else:
                                normalized.append(str(content))
                            break
                    else:
                        normalized.append(str(c))
                elif isinstance(c, str):
                    normalized.append(c)
                else:
                    normalized.append(str(c))

            self._last_rollouts = normalized

            # Pass original completions — base_fn handles its own normalization
            return base_fn(completions, **kwargs)

        return reward_fn_with_rollout_capture

    # ------------------------------------------------------------------
    # Legacy interface stub — never called, required by ABC.
    # ------------------------------------------------------------------
    def compute_advantages(self, rewards: List[float]) -> List[float]:
        return list(rewards)
