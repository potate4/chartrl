"""Model loading utilities for HCPC-RLVR."""

from typing import Optional, Tuple, Any
from pathlib import Path
import torch

from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    AutoModelForCausalLM,
)
from peft import PeftModel, get_peft_model

from .lora_config import get_lora_config_from_training_config, get_lora_config


def load_model(
    model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
    device_map: str = "auto",
    torch_dtype=torch.bfloat16,
    use_flash_attention: bool = True,
    cache_dir: Optional[str] = None,
) -> Tuple[Any, Any]:
    """
    Load model and processor for inference.

    Args:
        model_name: HuggingFace model name or path
        device_map: Device mapping strategy
        torch_dtype: Model dtype
        use_flash_attention: Whether to use Flash Attention 2
        cache_dir: Cache directory for downloads

    Returns:
        Tuple of (model, processor)
    """
    # Load processor
    processor = AutoProcessor.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )

    # Determine model class
    if "qwen" in model_name.lower():
        model_class = Qwen2_5_VLForConditionalGeneration
    else:
        model_class = AutoModelForCausalLM

    # Load model
    attn_impl = "flash_attention_2" if use_flash_attention else "eager"

    model = model_class.from_pretrained(
        model_name,
        device_map=device_map,
        torch_dtype=torch_dtype,
        attn_implementation=attn_impl,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )

    return model, processor


def load_model_for_training(
    config,
) -> Tuple[Any, Any]:
    """
    Load model configured for GRPO training.

    Key differences from inference:
    - device_map=None (TRL handles device placement)
    - LoRA adapters applied
    - Gradient checkpointing enabled

    Args:
        config: TrainingConfig instance

    Returns:
        Tuple of (model, processor)
    """
    # Load processor
    processor = AutoProcessor.from_pretrained(
        config.model_name,
        cache_dir=config.cache_dir,
        trust_remote_code=True,
    )

    # For GRPO training, we need device_map=None
    # TRL/DeepSpeed will handle device placement
    model_kwargs = {
        "torch_dtype": torch.bfloat16 if config.bf16 else torch.float32,
        "cache_dir": config.cache_dir,
        "trust_remote_code": True,
    }

    # Flash attention can cause issues with some GRPO setups
    if config.use_flash_attention:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    # Load base model
    if "qwen" in config.model_name.lower():
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            config.model_name,
            device_map=None,  # Important for TRL
            **model_kwargs,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            device_map=None,
            **model_kwargs,
        )

    # Enable gradient checkpointing
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Apply LoRA
    lora_config = get_lora_config_from_training_config(config)
    model = get_peft_model(model, lora_config)

    # Print trainable parameters
    model.print_trainable_parameters()

    return model, processor


def load_model_with_checkpoint(
    checkpoint_path: str,
    base_model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
    device_map: str = "auto",
    cache_dir: Optional[str] = None,
) -> Tuple[Any, Any]:
    """
    Load model with a trained checkpoint.

    Args:
        checkpoint_path: Path to checkpoint directory
        base_model_name: Base model name
        device_map: Device mapping
        cache_dir: Cache directory

    Returns:
        Tuple of (model, processor)
    """
    checkpoint_path = Path(checkpoint_path)

    # Load processor
    processor = AutoProcessor.from_pretrained(
        base_model_name,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )

    # Load base model
    if "qwen" in base_model_name.lower():
        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            base_model_name,
            device_map=device_map,
            torch_dtype=torch.bfloat16,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map=device_map,
            torch_dtype=torch.bfloat16,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )

    # Load LoRA adapter
    model = PeftModel.from_pretrained(
        base_model,
        checkpoint_path,
        is_trainable=False,
    )

    # Merge weights for faster inference (optional)
    # model = model.merge_and_unload()

    model.eval()

    return model, processor


def prepare_model_for_generation(model, processor):
    """
    Prepare model for generation.

    Sets appropriate generation config.

    Args:
        model: Model instance
        processor: Processor instance

    Returns:
        Model with generation config set
    """
    # Set pad token if not set
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # Update model config
    model.config.pad_token_id = processor.tokenizer.pad_token_id

    return model


def count_parameters(model) -> dict:
    """
    Count model parameters.

    Args:
        model: Model instance

    Returns:
        Dict with parameter counts
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
        "trainable_pct": trainable / total * 100 if total > 0 else 0,
    }
