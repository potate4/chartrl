# HCPC-RLVR: Google Colab Training Guide

Complete guide to running HCPC-RLVR experiments on Google Colab with automatic checkpoint backup to Google Drive.

---

## Cell 1: Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

---

## Cell 2: Clone Repository and Setup

```python
import os

# Configure your repository (update these)
REPO_URL = "https://github.com/YOUR_USERNAME/chartrl.git"  # Your repo URL
BRANCH = "main"  # or your branch name

# Working directories
WORK_DIR = "/content/chartrl"
DRIVE_BACKUP_DIR = "/content/drive/MyDrive/hcpc_rlvr_checkpoints"

# Clone or pull latest
if os.path.exists(WORK_DIR):
    print("Repository exists, pulling latest...")
    os.chdir(WORK_DIR)
    !git pull origin {BRANCH}
else:
    print("Cloning repository...")
    !git clone -b {BRANCH} {REPO_URL} {WORK_DIR}
    os.chdir(WORK_DIR)

# Create backup directory on Drive
os.makedirs(DRIVE_BACKUP_DIR, exist_ok=True)
print(f"Backup directory: {DRIVE_BACKUP_DIR}")

# Change to app directory
os.chdir(f"{WORK_DIR}/app")
print(f"Working directory: {os.getcwd()}")
```

---

## Cell 3: Install Dependencies

```python
# Install required packages
!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
!pip install -q transformers>=4.45.0 accelerate>=0.34.0 peft>=0.13.0
!pip install -q trl>=0.12.0  # Required for GRPO trainer
!pip install -q datasets pillow tqdm wandb
!pip install -q sentence-transformers deepspeed bitsandbytes
!pip install -q qwen-vl-utils

# Verify GPU
import torch
print(f"GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

### Alternative: Install from requirements.txt

```python
!pip install -q -r requirements.txt
```

---

## Cell 4: Setup Automatic Checkpoint Backup to Drive

```python
import threading
import time
import shutil
from datetime import datetime
from pathlib import Path

class DriveBackupManager:
    """Automatically backs up checkpoints to Google Drive."""

    def __init__(
        self,
        source_dir: str = "./outputs",
        backup_dir: str = "/content/drive/MyDrive/hcpc_rlvr_checkpoints",
        interval_minutes: int = 5,
    ):
        self.source_dir = Path(source_dir)
        self.backup_dir = Path(backup_dir)
        self.interval = interval_minutes * 60  # Convert to seconds
        self.running = False
        self.thread = None
        self.last_backup = None
        self.backup_count = 0

    def _get_checkpoint_dirs(self):
        """Find all checkpoint directories."""
        checkpoints = []
        if self.source_dir.exists():
            for exp_dir in self.source_dir.iterdir():
                if exp_dir.is_dir():
                    for run_dir in exp_dir.glob("run_*"):
                        ckpt_dir = run_dir / "checkpoints"
                        if ckpt_dir.exists():
                            checkpoints.append(ckpt_dir)
        return checkpoints

    def _backup_once(self):
        """Perform a single backup."""
        checkpoint_dirs = self._get_checkpoint_dirs()

        if not checkpoint_dirs:
            return False

        for ckpt_dir in checkpoint_dirs:
            # Get relative path from outputs
            rel_path = ckpt_dir.relative_to(self.source_dir)
            dest_dir = self.backup_dir / rel_path

            # Copy new checkpoint files
            for ckpt in ckpt_dir.iterdir():
                if ckpt.is_dir() and ckpt.name.startswith("step_"):
                    dest_ckpt = dest_dir / ckpt.name
                    if not dest_ckpt.exists():
                        print(f"[Backup] Copying {ckpt.name} to Drive...")
                        shutil.copytree(ckpt, dest_ckpt)
                        self.backup_count += 1

            # Also copy config and metrics
            run_dir = ckpt_dir.parent
            for file in ["config.json", "metrics.jsonl"]:
                src_file = run_dir / file
                if src_file.exists():
                    dest_file = dest_dir.parent / file
                    dest_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src_file, dest_file)

        self.last_backup = datetime.now()
        return True

    def _backup_loop(self):
        """Background backup loop."""
        while self.running:
            try:
                self._backup_once()
            except Exception as e:
                print(f"[Backup] Error: {e}")
            time.sleep(self.interval)

    def start(self):
        """Start automatic backup."""
        if self.running:
            print("[Backup] Already running")
            return

        self.running = True
        self.thread = threading.Thread(target=self._backup_loop, daemon=True)
        self.thread.start()
        print(f"[Backup] Started - backing up every {self.interval // 60} minutes")
        print(f"[Backup] Source: {self.source_dir}")
        print(f"[Backup] Destination: {self.backup_dir}")

    def stop(self):
        """Stop automatic backup."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        print(f"[Backup] Stopped - {self.backup_count} checkpoints backed up")

    def backup_now(self):
        """Trigger immediate backup."""
        print("[Backup] Manual backup triggered...")
        success = self._backup_once()
        if success:
            print(f"[Backup] Complete at {self.last_backup}")
        else:
            print("[Backup] No checkpoints found yet")

    def status(self):
        """Print backup status."""
        print(f"[Backup] Running: {self.running}")
        print(f"[Backup] Last backup: {self.last_backup}")
        print(f"[Backup] Total backed up: {self.backup_count}")


# Create and start backup manager
backup_manager = DriveBackupManager(
    source_dir="./outputs",
    backup_dir=DRIVE_BACKUP_DIR,
    interval_minutes=5,  # Backup every 5 minutes
)
backup_manager.start()
```

---

## Cell 5: Restore Checkpoints from Drive (Optional)

Run this cell if you're resuming from a previous session:

```python
import shutil
from pathlib import Path

def restore_from_drive(
    drive_dir: str = "/content/drive/MyDrive/hcpc_rlvr_checkpoints",
    local_dir: str = "./outputs",
):
    """Restore checkpoints from Google Drive."""
    drive_path = Path(drive_dir)
    local_path = Path(local_dir)

    if not drive_path.exists():
        print("No backup found on Drive")
        return

    # Find all experiments in backup
    for exp_dir in drive_path.iterdir():
        if exp_dir.is_dir():
            dest_exp = local_path / exp_dir.name

            for run_dir in exp_dir.glob("run_*"):
                dest_run = dest_exp / run_dir.name

                if dest_run.exists():
                    print(f"Skipping {run_dir.name} (already exists)")
                    continue

                print(f"Restoring {exp_dir.name}/{run_dir.name}...")
                shutil.copytree(run_dir, dest_run)

    print("Restore complete!")

# Uncomment to restore:
# restore_from_drive()
```

---

## Cell 6: Configure Weights & Biases (Optional)

```python
import wandb

# Login to WandB (optional but recommended for tracking)
# Get your API key from: https://wandb.ai/authorize
wandb.login()

# Or disable WandB by adding --no-wandb to training commands
```

---

## Cell 7: Verify Setup

```python
import sys
sys.path.insert(0, '.')

# Test imports
from configs.experiment import EXPERIMENTS, list_experiments
from data import load_training_dataset
from models import load_model_for_training

print("All imports successful!")
print()
list_experiments()
```

---

## Cell 8: Quick Test Run (Recommended First)

```python
# Run a quick test with small subset to verify everything works
!python scripts/train.py \
    --experiment grpo_baseline \
    --subset-size 10 \
    --num-epochs 1 \
    --no-wandb

print("\nTest complete! If no errors, proceed to full training.")
```

---

## Cell 9: Run Training

Choose one of the experiments below:

### Option A: GRPO Baseline (Chart-RVR Reproduction)

```python
!python scripts/train.py \
    --experiment grpo_baseline \
    --num-epochs 3

# Trigger backup after training
backup_manager.backup_now()
```

### Option B: NSR + HCPC

```python
!python scripts/train.py \
    --experiment nsr_hcpc \
    --num-epochs 3

backup_manager.backup_now()
```

### Option C: W-REINFORCE + HCPC (Full HCPC-RLVR)

```python
!python scripts/train.py \
    --experiment w_reinforce_hcpc \
    --num-epochs 3

backup_manager.backup_now()
```

### Option D: Run All 6 Experiments

```python
# This will take a long time - run overnight or on Colab Pro
!python scripts/run_experiments.py

backup_manager.backup_now()
```

---

## Cell 10: Resume Training (If Disconnected)

```python
# First, restore checkpoints from Drive
from pathlib import Path
import shutil

drive_dir = Path("/content/drive/MyDrive/hcpc_rlvr_checkpoints")
local_dir = Path("./outputs")

for exp_dir in drive_dir.iterdir():
    if exp_dir.is_dir():
        dest_exp = local_dir / exp_dir.name
        for run_dir in exp_dir.glob("run_*"):
            dest_run = dest_exp / run_dir.name
            if not dest_run.exists():
                print(f"Restoring {exp_dir.name}/{run_dir.name}...")
                shutil.copytree(run_dir, dest_run)

# Then resume training
!python scripts/train.py \
    --experiment grpo_baseline \
    --resume

backup_manager.backup_now()
```

---

## Cell 11: List Available Runs

```python
!python scripts/train.py --experiment grpo_baseline --list-runs
!python scripts/train.py --experiment nsr_hcpc --list-runs
!python scripts/train.py --experiment w_reinforce_hcpc --list-runs
```

---

## Cell 12: Evaluation

```python
# Evaluate on in-distribution (ChartQA) and out-of-distribution (EvoChart)
!python scripts/evaluate.py \
    --checkpoint outputs/grpo_baseline/best \
    --id-dataset chartqa \
    --ood-dataset evochart

# Copy results to Drive
!cp outputs/*/eval_results.json {DRIVE_BACKUP_DIR}/
```

---

## Cell 13: Download Results

```python
from google.colab import files
import json

# Zip and download results
!zip -r results.zip outputs/*/eval_results.json outputs/*/metrics.jsonl

files.download('results.zip')
```

---

## Cell 14: Monitor Training (Run in Separate Tab)

```python
# Watch training progress
import time
from pathlib import Path
from IPython.display import clear_output

def monitor_training(experiment="grpo_baseline", refresh_seconds=30):
    """Monitor training progress in real-time."""
    while True:
        clear_output(wait=True)

        # Find latest run
        exp_dir = Path(f"./outputs/{experiment}")
        if exp_dir.exists():
            runs = sorted(exp_dir.glob("run_*"))
            if runs:
                latest_run = runs[-1]
                metrics_file = latest_run / "metrics.jsonl"

                if metrics_file.exists():
                    # Read last 10 lines
                    with open(metrics_file) as f:
                        lines = f.readlines()[-10:]

                    print(f"=== {experiment} - {latest_run.name} ===")
                    print(f"Metrics file: {metrics_file}")
                    print("-" * 50)
                    for line in lines:
                        data = json.loads(line)
                        step = data.get('step', '?')
                        loss = data.get('loss', '?')
                        reward = data.get('mean_reward', '?')
                        print(f"Step {step}: loss={loss:.4f}, reward={reward:.4f}")
                else:
                    print(f"Waiting for metrics... ({experiment})")
        else:
            print(f"Experiment not started yet: {experiment}")

        # Backup status
        print("-" * 50)
        backup_manager.status()

        time.sleep(refresh_seconds)

# Uncomment to run monitor (blocks the cell):
# monitor_training("grpo_baseline")
```

---

## Cell 15: Cleanup and Final Backup

```python
# Final backup before ending session
backup_manager.backup_now()
backup_manager.stop()

# Show what's backed up
!ls -la {DRIVE_BACKUP_DIR}
```

---

## Memory Optimization for Colab

If you run out of GPU memory, add these flags:

```python
# For T4 GPU (16GB) - use gradient checkpointing
!python scripts/train.py \
    --experiment grpo_baseline \
    --batch-size 1 \
    --num-generations 4

# For A100 (40GB) - can use larger batches
!python scripts/train.py \
    --experiment grpo_baseline \
    --batch-size 4 \
    --num-generations 8
```

---

## Troubleshooting

### Out of Memory
```python
# Clear GPU cache
import torch
torch.cuda.empty_cache()

# Reduce batch size and generations
# --batch-size 1 --num-generations 4
```

### Session Disconnected
1. Re-run Cells 1-4 (Mount Drive, Clone, Install, Start Backup)
2. Run Cell 10 (Restore from Drive and Resume)

### WandB Issues
```python
# Disable WandB if having issues
# Add --no-wandb to training commands
```

### Import Errors
```python
# Make sure you're in the right directory
import os
os.chdir("/content/chartrl/app")
```

---

## Experiment Tracking

| Experiment | Policy | HCPC | Expected ID Acc | Expected OOD Gap |
|------------|--------|------|-----------------|------------------|
| grpo_baseline | GRPO | No | ~70% | -15% |
| grpo_hcpc | GRPO | Yes | ~70% | -12% |
| nsr_baseline | NSR | No | ~68% | -14% |
| nsr_hcpc | NSR | Yes | ~69% | -10% |
| w_reinforce_baseline | W-REINFORCE | No | ~71% | -13% |
| w_reinforce_hcpc | W-REINFORCE | Yes | ~72% | -8% |

---

## Quick Reference

```bash
# Train
python scripts/train.py --experiment <name>

# Resume
python scripts/train.py --experiment <name> --resume

# Quick test
python scripts/train.py --experiment <name> --subset-size 100

# Evaluate
python scripts/evaluate.py --checkpoint outputs/<name>/best

# Run all
python scripts/run_experiments.py
```
