# HCPC-RLVR Implementation Plan

## Overview

This document outlines the complete implementation plan for **HCPC-RLVR** (Hierarchical Correct-Path Consistency with Reinforcement Learning from Verifiable Rewards). The goal is to create a self-contained, clean codebase inside the `app/` folder that:

1. Implements the HCPC and CLC rewards from your methodology
2. Supports three policy update methods: GRPO, NSR, W-REINFORCE
3. Uses Chart-RVR's CoT dataset for training
4. Evaluates on EvoChart (OOD) and ChartQA (ID)

---

## Understanding the Methodology

### What We're Building

Your thesis proposes that chart reasoning has a **hierarchical structure** with different objectives at each level:

```
Input → [Chart Type] → [Table Extraction] → [Reasoning] → [Answer]
           ↓                ↓                  ↓            ↓
       CONSISTENT       CONSISTENT          DIVERSE     CONSISTENT
```

**Key Insight**: Standard GRPO causes "reasoning collapse" (model converges to single strategy) and doesn't catch "hallucinated reasoning" (correct answer via fabricated steps).

### The Three Components

#### 1. Policy Update Methods (from NSR paper)
- **GRPO**: Standard - reinforces correct, penalizes wrong (causes collapse)
- **NSR**: Only penalizes wrong samples (preserves diversity)
- **W-REINFORCE**: λ·PSR + NSR where λ=0.1 (balanced)

#### 2. HCPC Reward (Hierarchical Correct-Path Consistency)
- Filter to **ground-truth-correct** rollouts (type, table, answer all match)
- Measure **C_type**: Consistency of chart type predictions
- Measure **C_table**: Consistency of extracted tables
- Measure **D_reason**: Diversity of reasoning traces
- Combine: `R_HCPC = (correct_rate) × (w₁·C_type + w₂·C_table + w₃·D_reason)`

#### 3. CLC Reward (Cross-Level Coherence)
- For each rollout, check if reasoning **references values from extracted table**
- `C_coherence = |values_in_reasoning ∩ values_in_table| / |values_in_reasoning|`
- Catches hallucinated reasoning

### How Rewards Combine

```
R_total[i] = R_base[i] + λ_hcpc·R_HCPC + λ_clc·R_CLC[i]

Where R_base = format + accuracy + length + chart_type + table + process (from Chart-RVR)
```

---

## File Structure

```
app/
├── PLAN.md                      # This file
├── README.md                    # Usage documentation
├── requirements.txt             # Dependencies
│
├── configs/
│   ├── __init__.py
│   ├── base.py                  # Base configuration dataclass
│   ├── experiment.py            # Experiment configurations (6 experiments)
│   └── deepspeed_zero3.yaml     # DeepSpeed config
│
├── data/
│   ├── __init__.py
│   ├── dataset.py               # Dataset loading (ChartQA, EvoChart)
│   ├── preprocessing.py         # Image resizing, text processing
│   └── prompts.py               # System prompts for CoT format
│
├── models/
│   ├── __init__.py
│   ├── loader.py                # VLM loading (Qwen2.5-VL-3B)
│   └── lora_config.py           # LoRA configurations
│
├── rewards/
│   ├── __init__.py
│   ├── base_rewards.py          # Chart-RVR rewards (format, accuracy, etc.)
│   ├── hcpc_reward.py           # HCPC: Hierarchical Correct-Path Consistency
│   ├── clc_reward.py            # CLC: Cross-Level Coherence
│   └── reward_aggregator.py     # Combines all rewards
│
├── trainers/
│   ├── __init__.py
│   ├── base_trainer.py          # Base GRPO trainer wrapper
│   ├── grpo_trainer.py          # Standard GRPO
│   ├── nsr_trainer.py           # Negative Sample Reinforcement
│   └── w_reinforce_trainer.py   # Weighted REINFORCE
│
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py               # Accuracy, relaxed match
│   ├── diversity_metrics.py     # C_table, D_reason, Coherence
│   └── evaluator.py             # Full evaluation pipeline
│
├── utils/
│   ├── __init__.py
│   ├── parsing.py               # Extract type, table, reasoning, answer from output
│   ├── similarity.py            # Sentence similarity (MiniLM)
│   ├── checkpointing.py         # Checkpoint save/load/management
│   └── logging_utils.py         # WandB and console logging
│
├── scripts/
│   ├── train.py                 # Main training script
│   ├── evaluate.py              # Evaluation script
│   └── run_experiments.py       # Run all 6 experiments
│
└── tests/
    ├── test_rewards.py          # Unit tests for rewards
    ├── test_parsing.py          # Unit tests for parsing
    └── test_hcpc.py             # Unit tests for HCPC computation
```

---

## Checkpoint Management System

### Design Goals

1. **Frequent saves**: Configurable interval (default every 50 steps)
2. **Resumable anywhere**: Load checkpoint on different device/session
3. **Fresh start option**: `--from-scratch` flag starts new run without deleting old checkpoints
4. **Version history**: Checkpoints are timestamped and never overwritten
5. **Run isolation**: Each experiment run gets a unique ID

### Directory Structure

```
outputs/
├── grpo_baseline/
│   ├── run_20260208_143052/           # Unique run ID (timestamp)
│   │   ├── checkpoints/
│   │   │   ├── step_50/
│   │   │   │   ├── adapter_model.safetensors
│   │   │   │   ├── adapter_config.json
│   │   │   │   ├── trainer_state.json
│   │   │   │   └── training_args.json
│   │   │   ├── step_100/
│   │   │   ├── step_150/
│   │   │   └── latest -> step_150/    # Symlink to latest
│   │   ├── config.json                # Full config snapshot
│   │   ├── metrics.jsonl              # Training metrics log
│   │   └── run_info.json              # Run metadata
│   │
│   ├── run_20260209_091234/           # Another run (fresh start)
│   │   └── ...
│   │
│   └── best/                          # Optional: best checkpoint link
│
├── nsr_hcpc_clc/
│   └── run_20260210_102030/
│       └── ...
```

### Usage Examples

```bash
# Start fresh (creates new run directory)
python scripts/train.py --experiment grpo_baseline

# Resume from latest checkpoint of most recent run
python scripts/train.py --experiment grpo_baseline --resume

# Resume from specific checkpoint
python scripts/train.py --experiment grpo_baseline --resume-from outputs/grpo_baseline/run_20260208_143052/checkpoints/step_100

# Start fresh but keep old runs (default behavior)
python scripts/train.py --experiment grpo_baseline --from-scratch

# List all runs for an experiment
python scripts/train.py --experiment grpo_baseline --list-runs
```

### Checkpoint Contents

Each checkpoint saves:

```python
checkpoint = {
    # Model state
    "adapter_model": lora_adapter_weights,
    "adapter_config": lora_config,

    # Training state
    "trainer_state": {
        "global_step": 150,
        "epoch": 1.5,
        "best_metric": 0.85,
        "total_steps": 1000,
    },

    # Optimizer state (for exact resumption)
    "optimizer_state": optimizer.state_dict(),
    "scheduler_state": scheduler.state_dict(),

    # RNG states (for reproducibility)
    "rng_states": {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    },

    # Data state (resume from exact position)
    "dataloader_state": {
        "epoch": 1,
        "step_in_epoch": 50,
        "indices_seen": [...],
    },
}
```

### Implementation (utils/checkpointing.py)

```python
class CheckpointManager:
    """Manages checkpoint saving, loading, and run directories."""

    def __init__(
        self,
        experiment_name: str,
        output_base: str = "./outputs",
        save_every_n_steps: int = 50,
        keep_last_n: int = 5,  # Keep last N checkpoints, delete older
        keep_best: bool = True,
    ):
        self.experiment_name = experiment_name
        self.output_base = Path(output_base)
        self.save_every_n_steps = save_every_n_steps
        self.keep_last_n = keep_last_n
        self.keep_best = keep_best

        self.run_id = None
        self.run_dir = None

    def create_new_run(self) -> Path:
        """Create a new run directory with timestamp."""
        self.run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        self.run_dir = self.output_base / self.experiment_name / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "checkpoints").mkdir()
        return self.run_dir

    def get_latest_run(self) -> Optional[Path]:
        """Find the most recent run directory."""
        exp_dir = self.output_base / self.experiment_name
        if not exp_dir.exists():
            return None
        runs = sorted(exp_dir.glob("run_*"))
        return runs[-1] if runs else None

    def get_latest_checkpoint(self, run_dir: Path = None) -> Optional[Path]:
        """Get latest checkpoint from a run."""
        run_dir = run_dir or self.run_dir
        if run_dir is None:
            return None
        ckpt_dir = run_dir / "checkpoints"
        if not ckpt_dir.exists():
            return None
        # Check for 'latest' symlink first
        latest_link = ckpt_dir / "latest"
        if latest_link.exists():
            return latest_link.resolve()
        # Otherwise find highest step
        steps = sorted(ckpt_dir.glob("step_*"),
                       key=lambda p: int(p.name.split("_")[1]))
        return steps[-1] if steps else None

    def save_checkpoint(
        self,
        trainer,
        step: int,
        metrics: dict = None,
        is_best: bool = False,
    ):
        """Save checkpoint with all necessary state."""
        ckpt_dir = self.run_dir / "checkpoints" / f"step_{step}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Save adapter weights
        trainer.model.save_pretrained(ckpt_dir)

        # Save trainer state
        trainer_state = {
            "global_step": step,
            "epoch": trainer.state.epoch,
            "metrics": metrics or {},
        }
        with open(ckpt_dir / "trainer_state.json", "w") as f:
            json.dump(trainer_state, f, indent=2)

        # Save optimizer and scheduler
        torch.save({
            "optimizer": trainer.optimizer.state_dict(),
            "scheduler": trainer.lr_scheduler.state_dict() if trainer.lr_scheduler else None,
        }, ckpt_dir / "optimizer.pt")

        # Save RNG states
        torch.save({
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }, ckpt_dir / "rng_states.pt")

        # Update 'latest' symlink
        latest_link = self.run_dir / "checkpoints" / "latest"
        if latest_link.exists():
            latest_link.unlink()
        latest_link.symlink_to(ckpt_dir.name)

        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()

        # Save best if applicable
        if is_best and self.keep_best:
            best_dir = self.output_base / self.experiment_name / "best"
            if best_dir.exists():
                shutil.rmtree(best_dir)
            shutil.copytree(ckpt_dir, best_dir)

    def load_checkpoint(
        self,
        checkpoint_path: Path,
        trainer,
        load_optimizer: bool = True,
    ) -> dict:
        """Load checkpoint and restore all state."""
        # Load adapter
        from peft import PeftModel
        trainer.model = PeftModel.from_pretrained(
            trainer.model.base_model,
            checkpoint_path,
        )

        # Load trainer state
        with open(checkpoint_path / "trainer_state.json") as f:
            trainer_state = json.load(f)

        # Load optimizer
        if load_optimizer and (checkpoint_path / "optimizer.pt").exists():
            opt_state = torch.load(checkpoint_path / "optimizer.pt")
            trainer.optimizer.load_state_dict(opt_state["optimizer"])
            if opt_state["scheduler"] and trainer.lr_scheduler:
                trainer.lr_scheduler.load_state_dict(opt_state["scheduler"])

        # Restore RNG states
        if (checkpoint_path / "rng_states.pt").exists():
            rng_states = torch.load(checkpoint_path / "rng_states.pt")
            random.setstate(rng_states["python"])
            np.random.set_state(rng_states["numpy"])
            torch.set_rng_state(rng_states["torch"])
            if rng_states["cuda"] and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rng_states["cuda"])

        return trainer_state

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping last N."""
        ckpt_dir = self.run_dir / "checkpoints"
        steps = sorted(
            [p for p in ckpt_dir.glob("step_*") if p.is_dir()],
            key=lambda p: int(p.name.split("_")[1])
        )
        # Keep last N
        to_delete = steps[:-self.keep_last_n] if len(steps) > self.keep_last_n else []
        for ckpt in to_delete:
            shutil.rmtree(ckpt)

    def list_runs(self) -> List[dict]:
        """List all runs for this experiment."""
        exp_dir = self.output_base / self.experiment_name
        if not exp_dir.exists():
            return []
        runs = []
        for run_dir in sorted(exp_dir.glob("run_*")):
            info_path = run_dir / "run_info.json"
            if info_path.exists():
                with open(info_path) as f:
                    info = json.load(f)
            else:
                info = {"run_id": run_dir.name}
            # Get latest checkpoint step
            latest = self.get_latest_checkpoint(run_dir)
            if latest:
                info["latest_step"] = int(latest.name.split("_")[1])
            runs.append(info)
        return runs
```

### Config Updates

```python
@dataclass
class TrainingConfig:
    # ... existing fields ...

    # Checkpoint settings
    checkpoint_every_n_steps: int = 50
    keep_last_n_checkpoints: int = 5
    keep_best_checkpoint: bool = True

    # Resume settings
    resume: bool = False                    # Resume from latest
    resume_from: Optional[str] = None       # Resume from specific path
    from_scratch: bool = False              # Force new run
```

---

## Implementation Steps

### Phase 1: Core Infrastructure (Files 1-8)

#### Step 1: Configuration System
**Files**: `configs/base.py`, `configs/experiment.py`

```python
# configs/base.py
@dataclass
class TrainingConfig:
    # Model
    model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"

    # Training
    num_epochs: int = 3
    batch_size: int = 2
    num_generations: int = 8  # K rollouts
    learning_rate: float = 5e-7
    max_prompt_length: int = 4096
    max_completion_length: int = 768
    temperature: float = 0.8

    # Policy method
    policy_method: str = "grpo"  # grpo, nsr, w_reinforce
    lambda_psr: float = 0.1  # For W-REINFORCE

    # HCPC weights
    use_hcpc: bool = True
    w_type: float = 1.0
    w_table: float = 2.0
    w_reason: float = 1.5
    table_sim_threshold: float = 0.8

    # CLC weights
    use_clc: bool = True
    w_clc: float = 1.0

    # Paths
    output_dir: str = "./outputs"
    cache_dir: str = "./cache"
```

#### Step 2: Data Loading
**Files**: `data/dataset.py`, `data/preprocessing.py`, `data/prompts.py`

- Load from HuggingFace: `sanchit97/chart-rvr-grpo-train`
- Handle EvoChart for evaluation
- Image resizing (min/max pixels)
- CoT prompt format with `<think>`, `<type>`, `<table>`, `<answer>` tags

#### Step 3: Model Loading
**Files**: `models/loader.py`, `models/lora_config.py`

- Load Qwen2.5-VL-3B with proper device handling for GRPO
- LoRA config: r=8, alpha=16, target all projection layers

### Phase 2: Reward System (Files 9-13)

#### Step 4: Base Rewards (from Chart-RVR)
**File**: `rewards/base_rewards.py`

Implement the 7 rewards from the original paper:
1. `format_reward`: Check `<think>`, `<type>`, `<table>`, `<answer>` structure
2. `accuracy_reward`: Compare answer with label (numeric tolerance)
3. `length_reward`: Reward appropriate reasoning length
4. `token_count_reward`: Verify tag structure
5. `chart_type_reward`: Match predicted type with ground truth
6. `table_reward`: JSON validity + column/key matching
7. `process_reward`: Reasoning similarity (sentence embeddings)

#### Step 5: HCPC Reward (NEW)
**File**: `rewards/hcpc_reward.py`

```python
def compute_hcpc_reward(
    rollouts: List[Dict],      # All K rollouts
    ground_truth: Dict,        # {type, table, answer}
    config: TrainingConfig
) -> float:
    """
    Hierarchical Correct-Path Consistency Reward

    1. Filter to fully correct rollouts (type + table + answer match GT)
    2. Compute C_type: consistency of chart types
    3. Compute C_table: pairwise similarity of tables
    4. Compute D_reason: 1 - pairwise similarity of reasoning
    5. Return weighted combination
    """

    # Step 1: Filter
    fully_correct = filter_correct_rollouts(rollouts, ground_truth,
                                            table_threshold=config.table_sim_threshold)

    if len(fully_correct) < 2:
        return 0.0  # Need at least 2 for cross-rollout metrics

    # Step 2-4: Compute metrics
    c_type = compute_type_consistency(fully_correct)
    c_table = compute_table_consistency(fully_correct)
    d_reason = compute_reasoning_diversity(fully_correct)

    # Step 5: Combine
    correct_rate = len(fully_correct) / len(rollouts)
    r_hcpc = correct_rate * (
        config.w_type * c_type +
        config.w_table * c_table +
        config.w_reason * d_reason
    )

    return r_hcpc
```

#### Step 6: CLC Reward (NEW)
**File**: `rewards/clc_reward.py`

```python
def compute_clc_reward(
    rollout: Dict,  # Single rollout with {table, reasoning}
) -> float:
    """
    Cross-Level Coherence Reward

    Checks if reasoning references values from extracted table.
    """

    # Extract numeric values from reasoning
    reasoning_values = extract_numbers(rollout['reasoning'])

    # Extract numeric values from table
    table_values = extract_table_values(rollout['table'])

    if len(reasoning_values) == 0:
        return 1.0  # No values to check, assume coherent

    # Compute overlap
    overlap = len(set(reasoning_values) & set(table_values))
    coherence = overlap / len(reasoning_values)

    return coherence
```

#### Step 7: Reward Aggregator
**File**: `rewards/reward_aggregator.py`

```python
def compute_total_rewards(
    rollouts: List[Dict],
    ground_truth: Dict,
    config: TrainingConfig
) -> List[float]:
    """
    Compute total reward for each rollout.

    R_total[i] = R_base[i] + λ_hcpc·R_HCPC + λ_clc·R_CLC[i]

    Note: R_HCPC is a group-level reward applied equally to all rollouts.
    """

    total_rewards = []

    # Compute base rewards for each rollout
    for rollout in rollouts:
        r_base = compute_base_rewards(rollout, ground_truth)
        total_rewards.append(r_base)

    # Compute HCPC (group-level)
    if config.use_hcpc:
        r_hcpc = compute_hcpc_reward(rollouts, ground_truth, config)
        # Add to all rollouts (it's a group property)
        total_rewards = [r + r_hcpc for r in total_rewards]

    # Compute CLC (per-rollout)
    if config.use_clc:
        for i, rollout in enumerate(rollouts):
            r_clc = config.w_clc * compute_clc_reward(rollout)
            total_rewards[i] += r_clc

    return total_rewards
```

### Phase 3: Training Methods (Files 14-18)

#### Step 8: Base Trainer
**File**: `trainers/base_trainer.py`

Wraps TRL's GRPOTrainer with:
- Custom reward function integration
- Checkpoint management
- WandB logging

#### Step 9: GRPO Trainer
**File**: `trainers/grpo_trainer.py`

Standard GRPO with our custom rewards:
```python
class HCPCGRPOTrainer(BaseTrainer):
    """Standard GRPO with HCPC+CLC rewards"""

    def compute_advantages(self, rewards: List[float]) -> List[float]:
        # Standard: advantage = reward - mean(rewards)
        mean_r = sum(rewards) / len(rewards)
        return [r - mean_r for r in rewards]
```

#### Step 10: NSR Trainer
**File**: `trainers/nsr_trainer.py`

```python
class NSRTrainer(BaseTrainer):
    """Negative Sample Reinforcement"""

    def compute_advantages(self, rewards: List[float]) -> List[float]:
        threshold = self.config.reward_threshold
        advantages = []

        for r in rewards:
            if r >= threshold:
                # Skip correct samples (no gradient)
                advantages.append(0.0)
            else:
                # Penalize wrong samples
                advantages.append(-(threshold - r))

        return advantages
```

#### Step 11: W-REINFORCE Trainer
**File**: `trainers/w_reinforce_trainer.py`

```python
class WREINFORCETrainer(BaseTrainer):
    """Weighted REINFORCE: λ·PSR + NSR"""

    def compute_advantages(self, rewards: List[float]) -> List[float]:
        threshold = self.config.reward_threshold
        lambda_psr = self.config.lambda_psr  # 0.1
        advantages = []

        for r in rewards:
            if r >= threshold:
                # Weak positive reinforcement
                advantages.append(lambda_psr * r)
            else:
                # Strong negative reinforcement
                advantages.append(-(threshold - r))

        return advantages
```

### Phase 4: Evaluation (Files 19-22)

#### Step 12: Metrics
**File**: `evaluation/metrics.py`

- `exact_match`: String equality
- `relaxed_accuracy`: 5% numeric tolerance
- `extract_answer`: Parse `<answer>...</answer>`

#### Step 13: Diversity Metrics
**File**: `evaluation/diversity_metrics.py`

- `compute_table_consistency`: Average pairwise table similarity
- `compute_reasoning_diversity`: 1 - average pairwise reasoning similarity
- `compute_coherence`: CLC score

#### Step 14: Evaluator
**File**: `evaluation/evaluator.py`

Full evaluation pipeline:
1. Load checkpoint
2. Generate K responses per sample
3. Compute accuracy (Pass@1, Pass@k)
4. Compute C_table, D_reason, Coherence
5. Report OOD gap (ChartQA - EvoChart)

### Phase 5: Utilities and Scripts (Files 23-28)

#### Step 15: Parsing Utilities
**File**: `utils/parsing.py`

```python
def parse_response(text: str) -> Dict:
    """
    Parse model output into structured components.

    Returns: {
        'type': str,
        'table': dict,
        'reasoning': str,
        'answer': str
    }
    """
    type_match = re.search(r'<type>(.*?)</type>', text, re.DOTALL)
    table_match = re.search(r'<table>(.*?)</table>', text, re.DOTALL)
    think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    answer_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)

    return {
        'type': type_match.group(1).strip() if type_match else '',
        'table': parse_json_table(table_match.group(1)) if table_match else {},
        'reasoning': extract_reasoning(think_match.group(1)) if think_match else '',
        'answer': answer_match.group(1).strip() if answer_match else ''
    }
```

#### Step 16: Similarity Utilities
**File**: `utils/similarity.py`

```python
from sentence_transformers import SentenceTransformer

# Load once, cache globally
_model = None

def get_sentence_model():
    global _model
    if _model is None:
        _model = SentenceTransformer('all-MiniLM-L6-v2')
    return _model

def compute_similarity(text1: str, text2: str) -> float:
    model = get_sentence_model()
    embeddings = model.encode([text1, text2])
    return float(cosine_similarity([embeddings[0]], [embeddings[1]])[0][0])
```

#### Step 17: Training Script
**File**: `scripts/train.py`

```python
"""
Main training script for HCPC-RLVR

Usage:
    python scripts/train.py --config configs/experiment.py:grpo_baseline
    python scripts/train.py --config configs/experiment.py:nsr_hcpc_clc
"""

def main():
    args = parse_args()
    config = load_config(args.config)

    # Setup
    model, processor = load_model(config)
    train_dataset = load_dataset(config)

    # Create trainer based on policy method
    trainer_cls = {
        'grpo': HCPCGRPOTrainer,
        'nsr': NSRTrainer,
        'w_reinforce': WREINFORCETrainer
    }[config.policy_method]

    trainer = trainer_cls(
        model=model,
        processor=processor,
        train_dataset=train_dataset,
        config=config
    )

    # Train
    trainer.train()
    trainer.save()

if __name__ == "__main__":
    main()
```

#### Step 18: Evaluation Script
**File**: `scripts/evaluate.py`

```python
"""
Evaluation script

Usage:
    python scripts/evaluate.py --checkpoint outputs/grpo_baseline --dataset evochart
"""

def main():
    args = parse_args()

    # Load model with checkpoint
    model, processor = load_model_with_checkpoint(args.checkpoint)

    # Load evaluation dataset
    eval_dataset = load_eval_dataset(args.dataset)

    # Run evaluation
    evaluator = Evaluator(model, processor)
    results = evaluator.evaluate(eval_dataset, k_samples=8)

    # Report
    print(f"Accuracy: {results['accuracy']:.2%}")
    print(f"C_table: {results['c_table']:.3f}")
    print(f"D_reason: {results['d_reason']:.3f}")
    print(f"Coherence: {results['coherence']:.3f}")
```

#### Step 19: Experiment Runner
**File**: `scripts/run_experiments.py`

```python
"""
Run all 6 experiments from the methodology

Experiments:
1. GRPO (baseline, no HCPC+CLC)
2. GRPO + HCPC + CLC
3. NSR (no HCPC+CLC)
4. NSR + HCPC + CLC
5. W-REINFORCE (no HCPC+CLC)
6. W-REINFORCE + HCPC + CLC
"""

EXPERIMENTS = [
    ("grpo_baseline", {"policy_method": "grpo", "use_hcpc": False, "use_clc": False}),
    ("grpo_hcpc_clc", {"policy_method": "grpo", "use_hcpc": True, "use_clc": True}),
    ("nsr_baseline", {"policy_method": "nsr", "use_hcpc": False, "use_clc": False}),
    ("nsr_hcpc_clc", {"policy_method": "nsr", "use_hcpc": True, "use_clc": True}),
    ("w_reinforce_baseline", {"policy_method": "w_reinforce", "use_hcpc": False, "use_clc": False}),
    ("w_reinforce_hcpc_clc", {"policy_method": "w_reinforce", "use_hcpc": True, "use_clc": True}),
]
```

---

## Experiment Matrix

| # | Experiment Name | Policy | HCPC | CLC | Expected Outcome |
|---|-----------------|--------|------|-----|------------------|
| 1 | `grpo_baseline` | GRPO | No | No | Chart-RVR baseline, high ID, low OOD |
| 2 | `grpo_hcpc_clc` | GRPO | Yes | Yes | Improved consistency, slightly better OOD |
| 3 | `nsr_baseline` | NSR | No | No | Preserved diversity, better OOD |
| 4 | `nsr_hcpc_clc` | NSR | Yes | Yes | Best OOD with structured diversity |
| 5 | `w_reinforce_baseline` | W-REINFORCE | No | No | Balanced accuracy/diversity |
| 6 | `w_reinforce_hcpc_clc` | W-REINFORCE | Yes | Yes | **Best overall** (hypothesis) |

---

## Key Design Decisions

### 1. HCPC is a Group-Level Reward
Unlike per-sample rewards, HCPC measures properties **across all K rollouts**. The computed value is added to each rollout's reward equally. This encourages the model to produce diverse-yet-consistent sets of responses.

### 2. Ground-Truth Anchoring (Not Majority Voting)
During training, we filter to rollouts that match **ground truth** (type, table, answer), not just agreement with each other. This prevents rewarding "consistent wrong answers."

### 3. CLC is Per-Rollout
Unlike HCPC, CLC is computed for each rollout individually. A rollout with fabricated reasoning (values not in table) gets lower CLC, even if the answer is correct.

### 4. Threshold-Based Correct/Wrong Classification
For NSR and W-REINFORCE, we use a threshold (default 0.8 of max possible reward) to classify responses as "correct" or "wrong". This is more nuanced than binary accuracy.

### 5. Sentence Embeddings for Similarity
We use `all-MiniLM-L6-v2` for reasoning similarity (same as Chart-RVR's process conformity reward). This captures semantic similarity, not just lexical overlap.

---

## Dependencies

```txt
# Core
torch>=2.0.0
transformers>=4.40.0
accelerate>=0.27.0
peft>=0.10.0
trl>=0.12.0
datasets>=2.18.0
deepspeed>=0.14.0

# Vision
qwen-vl-utils>=0.0.8
Pillow>=10.0.0

# Similarity
sentence-transformers>=2.2.0
scikit-learn>=1.3.0

# Utilities
wandb>=0.16.0
tqdm>=4.66.0
numpy>=1.24.0
```

---

## Implementation Order

I will implement files in this order to ensure dependencies are satisfied:

### Day 1: Foundation
1. `configs/base.py` - Configuration dataclass
2. `configs/experiment.py` - Experiment configs
3. `data/prompts.py` - Prompt templates
4. `utils/parsing.py` - Output parsing
5. `utils/similarity.py` - Sentence embeddings

### Day 2: Data & Model
6. `data/preprocessing.py` - Image/text processing
7. `data/dataset.py` - Dataset loading
8. `models/lora_config.py` - LoRA settings
9. `models/loader.py` - Model loading

### Day 3: Rewards
10. `rewards/base_rewards.py` - Chart-RVR rewards
11. `rewards/hcpc_reward.py` - HCPC implementation
12. `rewards/clc_reward.py` - CLC implementation
13. `rewards/reward_aggregator.py` - Combine all rewards

### Day 4: Trainers
14. `trainers/base_trainer.py` - Base wrapper
15. `trainers/grpo_trainer.py` - GRPO
16. `trainers/nsr_trainer.py` - NSR
17. `trainers/w_reinforce_trainer.py` - W-REINFORCE

### Day 5: Evaluation & Scripts
18. `evaluation/metrics.py` - Accuracy metrics
19. `evaluation/diversity_metrics.py` - Diversity metrics
20. `evaluation/evaluator.py` - Full pipeline
21. `scripts/train.py` - Training entry point
22. `scripts/evaluate.py` - Evaluation entry point
23. `scripts/run_experiments.py` - Run all experiments

### Day 6: Testing & Documentation
24. `tests/test_parsing.py`
25. `tests/test_rewards.py`
26. `tests/test_hcpc.py`
27. `README.md`
28. `requirements.txt`

---

## Next Steps

After you approve this plan, I will:

1. Create all the directories
2. Implement each file in order
3. Write unit tests
4. Create documentation

**Questions to confirm before proceeding:**

1. **Dataset**: Use `sanchit97/chart-rvr-grpo-train` from HuggingFace? (This includes ChartQA, PlotQA, ChartFC samples with CoT annotations)

2. **Evaluation**: Use EvoChart from HuggingFace for OOD evaluation?

3. **Model**: Qwen2.5-VL-3B-Instruct (same as Chart-RVR)?

4. **Compute**: Single GPU with DeepSpeed ZeRO-3?

5. **Subset**: Start with a small subset (e.g., 1000 samples) for faster iteration?

---

## Validation Checklist

Before running experiments, verify:

- [ ] Parsing correctly extracts type, table, reasoning, answer
- [ ] Base rewards match Chart-RVR implementation
- [ ] HCPC correctly filters to ground-truth-correct rollouts
- [ ] HCPC computes C_type, C_table, D_reason correctly
- [ ] CLC correctly extracts numeric values and computes overlap
- [ ] NSR zeros out correct sample gradients
- [ ] W-REINFORCE applies λ=0.1 to correct samples
- [ ] Checkpoints save/load correctly
- [ ] WandB logs all metrics

---

*Plan created: 2026-02-08*
*Author: Claude (implementing HCPC-RLVR thesis)*
