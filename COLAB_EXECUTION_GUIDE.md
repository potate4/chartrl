# Chart-RVR: Colab Execution Guide

> **Pure execution steps - copy and run in sequence**

---

## Part 1: Setup (15-20 minutes)

### Step 1.1: Start Colab A100

1. Go to: https://colab.research.google.com/
2. Runtime → Change runtime type → **A100 GPU** → Save
3. Verify GPU:

```bash
!nvidia-smi
```

Expected output: Should show **A100-SXM4-40GB**

---

### Step 1.2: Clone Repository

```bash
!git clone https://github.com/sanchit97/chartrl.git
%cd chartrl
!pwd
```

Expected output: `/content/chartrl`

---

### Step 1.3: Install Dependencies

```bash
# PyTorch with CUDA 12.4
!pip install -q torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# Flash Attention 2 (critical - takes ~5 min)
!pip install -q flash-attn==2.7.4.post1 --no-build-isolation

# Core libraries
!pip install -q trl==0.12.0 transformers==4.53.1 datasets==3.6.0 accelerate==1.7.0 deepspeed==0.15.3 peft==0.15.2

# Additional dependencies
!pip install -q qwen-vl-utils==0.0.11 wandb==0.20.1 sentence-transformers sacrebleu scipy scikit-image Pillow==10.4.0
```

**Verify installation:**

```bash
!python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
!python -c "import flash_attn; print(f'Flash Attention: {flash_attn.__version__}')"
!python -c "from trl import GRPOTrainer; print('TRL: OK')"
```

Expected: All imports successful, CUDA=True

---

### Step 1.4: HuggingFace Login

```python
from huggingface_hub import login
login()  # Paste your HF token when prompted
```

Get token from: https://huggingface.co/settings/tokens

---

### Step 1.5: Setup Environment (CRITICAL)

```bash
# Create cache directory
!mkdir -p /content/hf_cache

# Verify files were updated (cache paths should show /content/hf_cache)
!grep "cache_dir = '/content/hf_cache'" main.py
!grep "cache_dir = '/content/hf_cache'" dataset_process.py
!grep "num_processes: 1" deepspeed_zero3.yaml
```

**Expected output:**
- All 3 commands should show matching lines
- If not, the code modifications didn't work

---

## Part 2: Download Training Dataset (15-30 minutes)

### Step 2.1: Download and Save Dataset

```python
from datasets import load_dataset
import os

cache_dir = '/content/hf_cache'

print("Downloading Chart-RVR GRPO training dataset...")
dataset = load_dataset("sanchit97/chart-rvr-grpo-train", cache_dir=cache_dir)

print(f"\nDataset info: {dataset}")
print(f"Number of samples: {len(dataset['train'])}")

# Save to disk for faster loading
save_path = f"{cache_dir}/grpo-chartrvr-train"
dataset.save_to_disk(save_path)
print(f"\n✓ Dataset saved to: {save_path}")

# Verify
assert os.path.exists(save_path), "Dataset save failed!"
print("\n✓ Verification passed")

# Inspect sample
sample = dataset['train'][0]
print("\n=== Sample Data ===")
print(f"Query: {sample['query'][:100]}...")
print(f"Label: {sample['label']}")
print(f"Chart Type: {sample['chart_type']}")
print(f"Has table: {sample['table'] is not None}")
print(f"Has reasoning: {len(sample['reasoning'])} chars")
```

**Expected output:**
```
Dataset info: DatasetDict({'train': Dataset(34200 rows)})
Number of samples: 34200
✓ Dataset saved to: /content/hf_cache/grpo-chartrvr-train
✓ Verification passed
```

**If download fails:** Check HF token, try again

---

## Part 3: GRPO Training (12-16 hours)

### Step 3.1: Verify Setup Before Training

```bash
# Check dataset
!ls -lh /content/hf_cache/grpo-chartrvr-train/

# Check GPU memory (should be mostly free)
!nvidia-smi

# Check code modifications
!head -n 50 main.py | grep cache_dir
```

---

### Step 3.2: Start GRPO Training

```bash
# Full training (34K samples, 4 epochs, ~14-16 hours)
!accelerate launch --config_file=deepspeed_zero3.yaml main.py \
  --mode grpo \
  --vlm-name qwen2-5-3b \
  --dataset-name evochart \
  --seed 2025
```

**For TESTING (1K samples, ~30-45 min):**

Edit `main.py` line 554-555:
```python
# Uncomment these lines:
train_dataset = train_dataset.select(range(1000))
logging.info(f"Using subset: {len(train_dataset)} samples")
```

Then run:
```bash
!accelerate launch --config_file=deepspeed_zero3.yaml main.py \
  --mode grpo \
  --vlm-name qwen2-5-3b \
  --dataset-name evochart \
  --seed 2025
```

---

### Step 3.3: Monitor Training

Initial logs should show:
```
Loading dataset from disk: /content/hf_cache/grpo-chartrvr-train
Loaded 34200 training samples  (or 1000 for test)
Training dataset formatted. Sample prompt length: ~2000
Loading evaluation dataset (EvoChart)...
Evaluation dataset loaded: 200 samples
✓ Datasets ready for GRPO training
```

Training logs (every 50 steps):
```
Step 50/17100 | Loss: X.XX | Reward: ~3.5
Step 100/17100 | Loss: X.XX | Reward: ~4.0
...
```

**Monitor GPU:**
```bash
!watch -n 30 nvidia-smi
```

**Expected memory usage:** ~30-35GB / 40GB

**Training progress:**

| Epoch | Steps | Reward | Time |
|-------|-------|--------|------|
| 1 | ~4275 | ~3-4 | ~3-4h |
| 2 | ~8550 | ~4-5 | ~3-4h |
| 3 | ~12825 | ~5-6 | ~3-4h |
| 4 | ~17100 | ~6-7 | ~3-4h |

**Checkpoints saved:**
```
./grpo-start-ckpts/qwen2-5-3b-prm-large-train-v2-2025/
├── checkpoint-500/
├── checkpoint-1000/
├── checkpoint-1500/
...
└── checkpoint-17100/  (final)
```

---

### Step 3.4: Handle Errors

**OOM Error (Out of Memory):**

Edit `main.py` line 604 and 710-711:
```python
per_device_train_batch_size=1  # was 2
num_generations=2  # was 4
```

**Dataset not found:**
```bash
# Re-download
!rm -rf /content/hf_cache/grpo-chartrvr-train
# Then run Part 2 again
```

**Training crashes:**
```bash
# Check logs
!tail -100 grpo-start-ckpts/qwen2-5-3b-prm-large-train-v2-2025/training.log
```

---

## Part 4: Evaluation (4-6 hours for all 6 benchmarks)

### Step 4.1: Verify Checkpoint

```bash
# List checkpoints
!ls -lh ./grpo-start-ckpts/qwen2-5-3b-prm-large-train-v2-2025/

# Should show checkpoint-500, checkpoint-1000, etc.
# The highest number is your final checkpoint
```

---

### Step 4.2: Evaluate on ChartQA (In-Domain)

```bash
!python main.py \
  --mode eval \
  --vlm-name qwen2-5-3b \
  --dataset-name chartqa-src \
  --dataset-split test \
  --cot True \
  --grpo-lora True \
  --seed 2025
```

**Expected output:**
```
Loading local GRPO checkpoint: ./grpo-start-ckpts/.../checkpoint-XXXX
✓ Loaded GRPO LoRA adapters
Starting evaluation on XXXX samples
...
Final Accuracy: ~84-85%
```

**Target:** 84.56%
**Acceptable:** 82-86%

---

### Step 4.3: Evaluate on PlotQA (In-Domain)

```bash
!python main.py \
  --mode eval \
  --vlm-name qwen2-5-3b \
  --dataset-name plotqa \
  --dataset-split test \
  --cot True \
  --grpo-lora True \
  --seed 2025
```

**Target:** 78.68%
**Acceptable:** 76-80%

---

### Step 4.4: Evaluate on ChartFC (In-Domain)

```bash
!python main.py \
  --mode eval \
  --vlm-name qwen2-5-3b \
  --dataset-name chartfc \
  --dataset-split test \
  --cot True \
  --grpo-lora True \
  --seed 2025
```

**Target:** 77.62%
**Acceptable:** 75-79%

---

### Step 4.5: Evaluate on EvoChart (OOD) 🎯

```bash
!python main.py \
  --mode eval \
  --vlm-name qwen2-5-3b \
  --dataset-name evochart \
  --dataset-split test \
  --cot True \
  --grpo-lora True \
  --seed 2025
```

**Target:** 53.36%
**Acceptable:** 51-55%

**This is the KEY result showing OOD generalization!**

---

### Step 4.6: Evaluate on ChartQAPro (OOD)

```bash
!python main.py \
  --mode eval \
  --vlm-name qwen2-5-3b \
  --dataset-name chartqapro \
  --dataset-split test \
  --cot True \
  --grpo-lora True \
  --seed 2025
```

**Target:** 28.38%
**Acceptable:** 26-30%

---

### Step 4.7: Evaluate on ChartBench (OOD)

```bash
!python main.py \
  --mode eval \
  --vlm-name qwen2-5-3b \
  --dataset-name chartbench \
  --dataset-split test \
  --cot True \
  --grpo-lora True \
  --seed 2025
```

**Target:** 68.32%
**Acceptable:** 66-70%

---

## Part 5: Results Summary

Create results table:

| Dataset | Type | Your Result | Paper Target | Diff | Status |
|---------|------|-------------|--------------|------|--------|
| ChartQA | ID | ___ % | 84.56% | ___ | ✓/✗ |
| PlotQA | ID | ___ % | 78.68% | ___ | ✓/✗ |
| ChartFC | ID | ___ % | 77.62% | ___ | ✓/✗ |
| EvoChart | OOD | ___ % | 53.36% | ___ | ✓/✗ |
| ChartQAPro | OOD | ___ % | 28.38% | ___ | ✓/✗ |
| ChartBench | OOD | ___ % | 68.32% | ___ | ✓/✗ |

**Success criteria:** All results within ±2% of target

---

## Alternative: Evaluate Pre-trained Model (Skip Training)

### Download Pre-trained Model

```python
from huggingface_hub import snapshot_download

# Download Chart-RVR-3B
snapshot_download(
    repo_id="sanchit97/chart-rvr-3b",
    local_dir="./models/chart-rvr-3b",
    cache_dir="/content/hf_cache"
)

print("✓ Model downloaded to ./models/chart-rvr-3b")
```

### Run Evaluation

The code automatically detects and loads from HuggingFace if no local checkpoint exists.

Just run evaluation commands from Part 4 with `--grpo-lora True`.

---

## Troubleshooting Quick Reference

### Error: "Dataset not found"
```bash
!rm -rf /content/hf_cache/grpo-chartrvr-train
# Re-run Part 2
```

### Error: "CUDA out of memory"
Edit `main.py`:
- Line 604: `per_device_train_batch_size=1`
- Line 710-711: `num_generations=2`

### Error: "No checkpoints found"
```bash
# Check if training completed
!ls ./grpo-start-ckpts/qwen2-5-3b-prm-large-train-v2-2025/

# If empty, re-run training
```

### Error: "Module 'trl' not found"
```bash
!pip install trl==0.12.0
```

### Error: "Flash attention import error"
```bash
!pip uninstall flash-attn -y
!pip install flash-attn==2.7.4.post1 --no-build-isolation
```

### Training stuck at 0%
- Check GPU: `!nvidia-smi`
- Check logs for errors
- Restart runtime and try again

### Evaluation showing random performance (~10-20%)
- Checkpoint not loaded correctly
- Check logs for "✓ Loaded GRPO LoRA adapters"
- If missing, checkpoint path is wrong

---

## Time Estimates

| Task | Duration | Can Skip? |
|------|----------|-----------|
| Setup | 15-20 min | No |
| Dataset download | 15-30 min | No |
| GRPO training (full) | 14-16 hours | Yes* |
| GRPO training (1K test) | 30-45 min | Yes* |
| Evaluation (6 datasets) | 4-6 hours | No |

*Can skip if using pre-trained models

**Total for full reproduction:** ~20-24 hours
**Total with pre-trained models:** ~5-7 hours

---

## Post-Training: Save Checkpoint to Google Drive (Optional)

```python
from google.colab import drive
drive.mount('/content/drive')

# Copy checkpoint
!cp -r ./grpo-start-ckpts/qwen2-5-3b-prm-large-train-v2-2025/ /content/drive/MyDrive/chart-rvr-checkpoint/

print("✓ Checkpoint saved to Google Drive")
```

**Load from Drive next time:**
```python
from google.colab import drive
drive.mount('/content/drive')

!cp -r /content/drive/MyDrive/chart-rvr-checkpoint/ ./grpo-start-ckpts/qwen2-5-3b-prm-large-train-v2-2025/
```

---

## Quick Sanity Check Commands

Run these before starting training:

```bash
# 1. Check GPU
!nvidia-smi | grep A100

# 2. Check dataset
!ls /content/hf_cache/grpo-chartrvr-train/dataset_dict.json

# 3. Check code modifications
!grep "/content/hf_cache" main.py | head -1

# 4. Check imports
!python -c "from trl import GRPOTrainer; import flash_attn; print('✓ All imports OK')"

# 5. Check disk space (need ~50GB free)
!df -h /content
```

All checks should pass before starting training.

---

## Emergency: Colab Disconnected During Training

**If Colab disconnects and training stops:**

1. Check if checkpoint was saved:
```bash
!ls ./grpo-start-ckpts/qwen2-5-3b-prm-large-train-v2-2025/
```

2. If checkpoints exist, training can't be resumed automatically with GRPO
3. You can either:
   - **Option A:** Use the latest checkpoint for evaluation
   - **Option B:** Restart training from scratch
   - **Option C:** Use pre-trained model from HuggingFace

**To prevent disconnection:**
- Use Colab Pro (longer runtime)
- Keep browser tab open
- Run overnight when you won't need the computer

---

**That's it! Follow these steps in sequence and you should successfully reproduce Chart-RVR results.**
