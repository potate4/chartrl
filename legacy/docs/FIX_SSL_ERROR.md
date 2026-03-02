# Fix: SSL Error When Loading Model

**Error:** `Ssl(Error { code: ErrorCode(5) })` when downloading model weights from Hugging Face CDN

---

## 🔧 Quick Fixes (Choose One)

### Fix 1: Use Cached Model (RECOMMENDED)

Since the model already downloaded successfully in your first run, use the cached version:

**Add this cell BEFORE training:**

```python
# Force use of cached model (skip download)
import os
os.environ['HF_HUB_OFFLINE'] = '1'  # Use cached files only
os.environ['TRANSFORMERS_OFFLINE'] = '1'

print("✓ Configured to use cached model")
```

**Then run training normally:**
```python
!accelerate launch --config_file=deepspeed_zero3.yaml main.py \
  --mode grpo \
  --vlm-name qwen2-5-3b \
  --dataset-name evochart \
  --seed 2025 \
  --subset-size 100 \
  --num-epochs 1 \
  --num-generations 2 \
  --batch-size 2 \
  --disable-gradient-checkpointing \
  2>&1 | tee poc_grpo.log
```

---

### Fix 2: Disable SSL Verification (NOT RECOMMENDED but works)

**Add before training:**

```python
# Disable SSL verification (use with caution)
import os
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['SSL_CERT_FILE'] = ''

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

print("⚠ SSL verification disabled")
```

---

### Fix 3: Pre-download Model Explicitly

**Run this cell first to download model:**

```python
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

print("Downloading model explicitly...")

model_name = "Qwen/Qwen2.5-VL-3B-Instruct"

# Download with retry
try:
    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True,
        resume_download=True,  # Resume if interrupted
    )
    print("✓ Processor downloaded")

    # Download model weights (this will cache them)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="cpu",  # Load to CPU first (saves memory)
        resume_download=True,
    )
    print("✓ Model downloaded")

    # Clear from memory
    del model
    del processor
    import gc
    gc.collect()

    print("✓ Model cached successfully")

except Exception as e:
    print(f"Download failed: {e}")
    print("Retrying with SSL verification disabled...")

    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="cpu",
    )

    del model
    del processor
    gc.collect()

    print("✓ Model downloaded (with SSL verification disabled)")
```

**Then run training:**
```python
# Now training will use cached model
!accelerate launch --config_file=deepspeed_zero3.yaml main.py ...
```

---

### Fix 4: Update SSL Certificates

**Run in a cell:**

```python
# Update SSL certificates
!apt-get update -qq
!apt-get install -y -qq ca-certificates
!update-ca-certificates

print("✓ SSL certificates updated")
```

---

## 🎯 Recommended Solution

**Use Fix 1 (Cached Model)** since you already downloaded the model successfully:

### Complete Cell Sequence:

**Cell A: Use Cached Model**
```python
import os

# Force offline mode (use cached files)
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

print("="*80)
print("CONFIGURED TO USE CACHED MODEL")
print("="*80)
print("Model location:", os.environ.get('HF_HOME', '/root/.cache/huggingface'))
print("Cache will be used instead of downloading")
print("="*80)
```

**Cell B: Train GRPO**
```python
print("="*80)
print("TRAINING 1/2: GRPO BASELINE")
print("="*80)

!accelerate launch --config_file=deepspeed_zero3.yaml main.py \
  --mode grpo \
  --vlm-name qwen2-5-3b \
  --dataset-name evochart \
  --seed 2025 \
  --subset-size 100 \
  --num-epochs 1 \
  --num-generations 2 \
  --batch-size 2 \
  --disable-gradient-checkpointing \
  2>&1 | tee poc_grpo.log

print("\n" + "="*80)
print("✓ GRPO training complete")
print("="*80)
```

**Cell C: Train NSR**
```python
print("="*80)
print("TRAINING 2/2: NSR")
print("="*80)

!accelerate launch --config_file=deepspeed_zero3.yaml main.py \
  --mode nsr \
  --vlm-name qwen2-5-3b \
  --dataset-name evochart \
  --seed 2025 \
  --subset-size 100 \
  --num-epochs 1 \
  --num-generations 4 \
  --batch-size 2 \
  --reward-threshold 0.5 \
  --disable-gradient-checkpointing \
  2>&1 | tee poc_nsr.log

print("\n" + "="*80)
print("✓ NSR training complete")
print("="*80)
```

---

## 🔍 Verify Cache Location

**Check if model is already cached:**

```python
import os
from pathlib import Path

cache_dir = Path(os.environ.get('HF_HOME', '/root/.cache/huggingface'))
model_cache = cache_dir / "hub" / "models--Qwen--Qwen2.5-VL-3B-Instruct"

print(f"Cache directory: {cache_dir}")
print(f"Model cache: {model_cache}")
print(f"Model cached: {model_cache.exists()}")

if model_cache.exists():
    print("\n✓ Model is already cached!")
    print("You can use HF_HUB_OFFLINE=1 to skip download")

    # Show cache size
    import subprocess
    size = subprocess.check_output(['du', '-sh', str(model_cache)]).decode().split()[0]
    print(f"Cache size: {size}")
else:
    print("\n✗ Model not in cache")
    print("Need to download first")
```

---

## 🐛 Debug: Why SSL Error?

The error happens because:

1. **Hugging Face CDN changed:** New CDN endpoint requires updated SSL certificates
2. **Network issues:** Your connection to `us.gcp.cdn.hf.co` has SSL handshake problems
3. **Certificate mismatch:** Local certificates don't match CDN certificates

**The error message:**
```
Ssl(Error { code: ErrorCode(5), cause: None }, X509VerifyResult { code: 0, error: "ok" })
```
- `ErrorCode(5)` = SSL handshake failure
- `X509VerifyResult: ok` = Certificate itself is valid, but connection failed

---

## ✅ Test if Fix Works

**After applying fix, test:**

```python
# Test model loading
from transformers import AutoProcessor

try:
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct",
        trust_remote_code=True,
        local_files_only=True,  # Only use cache
    )
    print("✓ Model loads from cache successfully!")
    del processor
except Exception as e:
    print(f"✗ Still failing: {e}")
    print("Try Fix 2 or Fix 3")
```

---

## Summary

**Best approach:**

1. ✅ **Use cached model** (Fix 1) - safest and fastest
2. ⚠️ **Disable SSL** (Fix 2) - works but not secure
3. 🔄 **Pre-download** (Fix 3) - good if cache empty
4. 🔧 **Update certs** (Fix 4) - should fix root cause

**For POC, use Fix 1 since model is already downloaded!**

```python
# Just add this before training:
import os
os.environ['HF_HUB_OFFLINE'] = '1'
```

Then run both GRPO and NSR training commands normally.
