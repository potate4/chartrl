# HCPC-RLVR: Hierarchical Correct-Path Consistency for Robust Chart Reasoning

A reinforcement learning framework for training vision-language models on chart reasoning tasks with improved out-of-distribution generalization and interpretable reasoning.

## Key Features

- **Three Policy Update Methods**: GRPO, NSR, W-REINFORCE
- **HCPC Reward**: Hierarchical Correct-Path Consistency for level-specific objectives
- **Robust Checkpointing**: Resume training from any device/session

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Training

```bash
# Train with GRPO baseline (Chart-RVR reproduction)
python scripts/train.py --experiment grpo_baseline

# Train with NSR + HCPC
python scripts/train.py --experiment nsr_hcpc

# Train with W-REINFORCE + HCPC (Full HCPC-RLVR)
python scripts/train.py --experiment w_reinforce_hcpc

# Quick test with subset
python scripts/train.py --experiment grpo_baseline --subset-size 100
```

### Resume Training

```bash
# Resume from latest checkpoint
python scripts/train.py --experiment grpo_baseline --resume

# Resume from specific checkpoint
python scripts/train.py --experiment grpo_baseline --resume-from outputs/grpo_baseline/run_xxx/checkpoints/step_100

# Start fresh run (keeps old checkpoints)
python scripts/train.py --experiment grpo_baseline --from-scratch
```

### Evaluation

```bash
# Evaluate on EvoChart (OOD)
python scripts/evaluate.py --checkpoint outputs/grpo_baseline/best --dataset evochart

# Evaluate on both ID and OOD
python scripts/evaluate.py --checkpoint outputs/nsr_hcpc/best --id-dataset chartqa --ood-dataset evochart
```

### Run All Experiments

```bash
# Run all 6 experiments
python scripts/run_experiments.py

# Run specific experiments
python scripts/run_experiments.py --experiments nsr_hcpc w_reinforce_hcpc
```

## Experiments

| # | Name | Policy | HCPC | Description |
|---|------|--------|------|-------------|
| 1 | `grpo_baseline` | GRPO | No | Chart-RVR reproduction |
| 2 | `grpo_hcpc` | GRPO | Yes | GRPO with HCPC reward |
| 3 | `nsr_baseline` | NSR | No | Negative Sample Reinforcement |
| 4 | `nsr_hcpc` | NSR | Yes | NSR with HCPC reward |
| 5 | `w_reinforce_baseline` | W-REINFORCE | No | Weighted REINFORCE |
| 6 | `w_reinforce_hcpc` | W-REINFORCE | Yes | **Full HCPC-RLVR** |

## Project Structure

```
app/
├── configs/          # Configuration management
├── data/             # Dataset loading and prompts
├── models/           # Model loading and LoRA
├── rewards/          # Reward functions (HCPC)
├── trainers/         # GRPO, NSR, W-REINFORCE trainers
├── evaluation/       # Metrics and evaluation
├── utils/            # Parsing, similarity, checkpointing
└── scripts/          # Training and evaluation scripts
```

## Checkpoint Management

Checkpoints are saved with timestamped run directories:

```
outputs/
├── grpo_baseline/
│   ├── run_20260208_143052/
│   │   ├── checkpoints/
│   │   │   ├── step_50/
│   │   │   ├── step_100/
│   │   │   └── latest -> step_100/
│   │   ├── config.json
│   │   └── metrics.jsonl
│   └── best/
```

Key features:
- **Never overwrite**: Each run gets a unique timestamp
- **Resume anywhere**: Full state saved (model, optimizer, RNG)
- **Fresh start option**: `--from-scratch` creates new run without deleting old ones
- **Configurable retention**: Keep last N checkpoints

## Configuration

Override any setting via command line:

```bash
python scripts/train.py --experiment grpo_baseline \
    --num-epochs 5 \
    --batch-size 4 \
    --learning-rate 1e-6 \
    --num-generations 8
```

Or modify `configs/experiment.py` for permanent changes.

## Reward System

### Base Rewards (from Chart-RVR)
- Format compliance
- Answer accuracy
- Reasoning length
- Chart type prediction
- Table extraction
- Process alignment

### HCPC Reward (Novel)
Measures cross-rollout properties among ground-truth-correct rollouts:
- **C_type**: Chart type consistency (want: high)
- **C_table**: Table extraction consistency (want: high)
- **D_reason**: Reasoning diversity (want: high)

## Citation

```bibtex
@article{hcpc-rlvr-2026,
  title={HCPC-RLVR: Hierarchical Correct-Path Consistency for Robust Chart Reasoning},
  author={...},
  year={2026}
}
```

app/
├── PLAN.md                          # Detailed implementation plan
├── README.md                        # Usage documentation
├── requirements.txt                 # Dependencies
│
├── configs/
│   ├── __init__.py
│   ├── base.py                      # TrainingConfig, CheckpointConfig, RewardConfig
│   ├── experiment.py                # 6 experiment configurations
│   └── deepspeed_zero3.yaml         # DeepSpeed config
│
├── data/
│   ├── __init__.py
│   ├── dataset.py                   # Dataset loading from HuggingFace
│   ├── preprocessing.py             # Image resizing
│   └── prompts.py                   # System prompt with CoT format
│
├── models/
│   ├── __init__.py
│   ├── loader.py                    # Model loading with checkpoint support
│   └── lora_config.py               # LoRA configurations
│
├── rewards/
│   ├── __init__.py
│   ├── base_rewards.py              # 7 Chart-RVR rewards
│   ├── hcpc_reward.py               # HCPC: Hierarchical Correct-Path Consistency
│   ├── clc_reward.py                # CLC: Cross-Level Coherence
│   └── reward_aggregator.py         # Combines all rewards
│
├── trainers/
│   ├── __init__.py
│   ├── base_trainer.py              # Base trainer with checkpoint management
│   ├── grpo_trainer.py              # Standard GRPO
│   ├── nsr_trainer.py               # Negative Sample Reinforcement
│   └── w_reinforce_trainer.py       # Weighted REINFORCE
│
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py                   # Accuracy, Pass@k
│   ├── diversity_metrics.py         # C_table, D_reason, Coherence
│   └── evaluator.py                 # Full evaluation pipeline
│
├── utils/
│   ├── __init__.py
│   ├── parsing.py                   # Extract type, table, reasoning, answer
│   ├── similarity.py                # Sentence embeddings (MiniLM)
│   ├── checkpointing.py             # Checkpoint save/load/resume
│   └── logging_utils.py             # WandB and console logging
│
└── scripts/
    ├── train.py                     # Main training entry point
    ├── evaluate.py                  # Evaluation script
    └── run_experiments.py           # Run all 6 experiments

Key Features Implemented
-----------------------
1. Checkpoint Management:

- Saves every N steps (configurable)
- Timestamped run directories (never overwrite)
- --resume from latest checkpoint
- --resume-from specific checkpoint
- --from-scratch starts new run without deleting old ones
- Saves full state: model, optimizer, RNG for reproducibility

2. Three Policy Methods:

- GRPO: advantage = reward - mean(rewards)
- NSR: Skip correct samples, only penalize wrong ones
- W-REINFORCE: λ·PSR + NSR where λ=0.1

3. HCPC Reward (your thesis contribution):

- Filters to ground-truth-correct rollouts
- C_type: Chart type consistency
- C_table: Table extraction consistency
- D_reason: Reasoning diversity

4. CLC Reward (your thesis contribution):

- Catches hallucinated reasoning
- Checks if reasoning references table values

5. Usage Examples

# Train
python scripts/train.py --experiment w_reinforce_hcpc_clc

# Resume
python scripts/train.py --experiment w_reinforce_hcpc_clc --resume

# Quick test
python scripts/train.py --experiment grpo_baseline --subset-size 100

# Evaluate
python scripts/evaluate.py --checkpoint outputs/w_reinforce_hcpc_clc/best --dataset evochart

# Run all experiments
python scripts/run_experiments.py











