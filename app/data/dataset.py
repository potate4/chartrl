"""Dataset loading and handling for HCPC-RLVR."""

from typing import Optional, Dict, List, Any, Callable
from pathlib import Path
import json

from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from PIL import Image

from .prompts import SYSTEM_PROMPT, format_conversation
from .preprocessing import process_image_for_model


class ChartDataset:
    """
    Dataset class for chart reasoning tasks.

    Handles loading from HuggingFace and local sources.
    """

    def __init__(
        self,
        dataset_name: str,
        split: str = "train",
        cache_dir: Optional[str] = None,
        subset_size: Optional[int] = None,
    ):
        """
        Initialize dataset.

        Args:
            dataset_name: HuggingFace dataset name or local path
            split: Dataset split (train, validation, test)
            cache_dir: Cache directory for downloads
            subset_size: Optional limit on dataset size
        """
        self.dataset_name = dataset_name
        self.split = split
        self.cache_dir = cache_dir
        self.subset_size = subset_size
        self.dataset = None

    def load(self) -> Dataset:
        """Load the dataset."""
        # Try loading from HuggingFace
        try:
            self.dataset = load_dataset(
                self.dataset_name,
                split=self.split,
                cache_dir=self.cache_dir,
            )
        except Exception:
            # Try loading from local path
            if Path(self.dataset_name).exists():
                self.dataset = self._load_local(self.dataset_name)
            else:
                raise ValueError(f"Could not load dataset: {self.dataset_name}")

        # Apply subset if specified
        if self.subset_size and len(self.dataset) > self.subset_size:
            self.dataset = self.dataset.select(range(self.subset_size))

        return self.dataset

    def _load_local(self, path: str) -> Dataset:
        """Load dataset from local JSON/JSONL file."""
        path = Path(path)

        if path.suffix == ".json":
            with open(path) as f:
                data = json.load(f)
        elif path.suffix == ".jsonl":
            data = []
            with open(path) as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

        return Dataset.from_list(data)

    def __len__(self) -> int:
        if self.dataset is None:
            self.load()
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict:
        if self.dataset is None:
            self.load()
        return self.dataset[idx]


def load_training_dataset(
    config,
    processor=None,
) -> Dataset:
    """
    Load training dataset formatted for GRPO training.

    Args:
        config: Training configuration
        processor: Optional model processor for formatting

    Returns:
        HuggingFace Dataset ready for training
    """
    # Load from HuggingFace
    dataset = load_dataset(
        config.dataset_name,
        split="train",
        cache_dir=config.cache_dir,
    )

    # Apply subset if specified
    if config.subset_size:
        dataset = dataset.select(range(min(config.subset_size, len(dataset))))

    # Format for GRPO training
    def format_example(example):
        """Format a single example for GRPO."""
        # Build conversation format
        conversation = format_conversation(example["query"])

        # Process image
        image = example.get("image")
        if image is not None:
            image = process_image_for_model(image)

        return {
            "prompt": conversation,
            "images": [image] if image is not None else [],
            "label": example.get("label", ""),
            "table": example.get("table", {}),
            "chart_type": example.get("chart_type", ""),
            "reasoning": example.get("reasoning", ""),
        }

    # Apply formatting
    dataset = dataset.map(
        format_example,
        remove_columns=dataset.column_names,
        desc="Formatting training data",
    )

    return dataset


def load_eval_dataset(
    dataset_name: str,
    split: str = "test",
    cache_dir: Optional[str] = None,
    subset_size: Optional[int] = None,
) -> Dataset:
    """
    Load evaluation dataset.

    Supports multiple evaluation datasets:
    - ChartQA (in-domain)
    - EvoChart (OOD)
    - ChartQAPro (OOD)
    - ChartBench (OOD)

    Args:
        dataset_name: Dataset name or path
        split: Split to load
        cache_dir: Cache directory
        subset_size: Optional size limit

    Returns:
        Evaluation dataset
    """
    # Map common names to HuggingFace paths
    dataset_map = {
        "chartqa": "HuggingFaceM4/ChartQA",
        "evochart": "lmms-lab/EvoChart",
        "chartqapro": "charxiv/ChartQAPro",
    }

    hf_name = dataset_map.get(dataset_name.lower(), dataset_name)

    # Load dataset
    dataset = load_dataset(
        hf_name,
        split=split,
        cache_dir=cache_dir,
    )

    # Apply subset
    if subset_size:
        dataset = dataset.select(range(min(subset_size, len(dataset))))

    return dataset


def create_grpo_collator(processor):
    """
    Create a collate function for GRPO training.

    Args:
        processor: Model processor

    Returns:
        Collate function
    """
    def collate_fn(batch: List[Dict]) -> Dict:
        """Collate batch of examples."""
        prompts = [ex["prompt"] for ex in batch]
        images = [ex["images"] for ex in batch]
        labels = [ex["label"] for ex in batch]

        # Additional fields for reward computation
        tables = [ex.get("table", {}) for ex in batch]
        chart_types = [ex.get("chart_type", "") for ex in batch]
        reasonings = [ex.get("reasoning", "") for ex in batch]

        return {
            "prompts": prompts,
            "images": images,
            "labels": labels,
            "tables": tables,
            "chart_types": chart_types,
            "reasonings": reasonings,
        }

    return collate_fn


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 1,
    shuffle: bool = True,
    collate_fn: Optional[Callable] = None,
    num_workers: int = 0,
) -> DataLoader:
    """
    Create a DataLoader for the dataset.

    Args:
        dataset: Dataset to wrap
        batch_size: Batch size
        shuffle: Whether to shuffle
        collate_fn: Optional collate function
        num_workers: Number of worker processes

    Returns:
        DataLoader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )


def get_dataset_info(dataset: Dataset) -> Dict[str, Any]:
    """
    Get information about a dataset.

    Args:
        dataset: Dataset to analyze

    Returns:
        Dict with dataset statistics
    """
    info = {
        "num_examples": len(dataset),
        "columns": dataset.column_names,
    }

    # Sample a few examples for inspection
    if len(dataset) > 0:
        sample = dataset[0]
        info["sample_keys"] = list(sample.keys())

        # Check for specific columns
        if "chart_type" in dataset.column_names:
            types = dataset["chart_type"]
            info["chart_type_distribution"] = dict(
                zip(*[(list(set(types)), [types.count(t) for t in set(types)])])
            )

    return info
