"""LoRA and quantization configurations."""

from typing import List, Optional
from peft import LoraConfig, TaskType


def get_lora_config(
    r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
    task_type: TaskType = TaskType.CAUSAL_LM,
) -> LoraConfig:
    """
    Get LoRA configuration for fine-tuning.

    Args:
        r: LoRA rank (default 8 for efficiency)
        lora_alpha: LoRA alpha scaling (typically 2 * r)
        lora_dropout: Dropout probability
        target_modules: Modules to apply LoRA to
        task_type: Task type for PEFT

    Returns:
        LoraConfig instance
    """
    if target_modules is None:
        # Default modules for Qwen2.5-VL
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "up_proj",
            "down_proj",
            "gate_proj",
        ]

    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        task_type=task_type,
        bias="none",
        inference_mode=False,
    )


def get_lora_config_from_training_config(config) -> LoraConfig:
    """
    Create LoRA config from training config.

    Args:
        config: TrainingConfig instance

    Returns:
        LoraConfig
    """
    return get_lora_config(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
    )


def get_quantization_config(load_in_4bit: bool = True):
    """
    Get quantization configuration for 4-bit training.

    Only use this if memory constrained.

    Args:
        load_in_4bit: Whether to use 4-bit quantization

    Returns:
        BitsAndBytesConfig or None
    """
    if not load_in_4bit:
        return None

    try:
        from transformers import BitsAndBytesConfig
        import torch

        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    except ImportError:
        print("Warning: bitsandbytes not available, skipping quantization")
        return None


# Predefined configs for different scenarios
LORA_CONFIGS = {
    # Standard config for GRPO (memory efficient)
    "grpo": {
        "r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "up_proj", "down_proj", "gate_proj",
        ],
    },

    # Higher rank for SFT (more capacity)
    "sft": {
        "r": 64,
        "lora_alpha": 128,
        "lora_dropout": 0.05,
        "target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "up_proj", "down_proj", "gate_proj",
        ],
    },

    # Minimal config for testing
    "minimal": {
        "r": 4,
        "lora_alpha": 8,
        "lora_dropout": 0.1,
        "target_modules": ["q_proj", "v_proj"],
    },
}


def get_preset_lora_config(preset: str) -> LoraConfig:
    """
    Get a preset LoRA configuration.

    Args:
        preset: One of 'grpo', 'sft', 'minimal'

    Returns:
        LoraConfig
    """
    if preset not in LORA_CONFIGS:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(LORA_CONFIGS.keys())}")

    return get_lora_config(**LORA_CONFIGS[preset])
