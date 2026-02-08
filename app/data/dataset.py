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


def _normalize_label(value) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        for item in value:
            if isinstance(item, str) and item.strip():
                return item
        return str(value[0]) if value else ""
    return str(value)


def _normalize_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(str(v) for v in value if v is not None)
    return str(value)


def _normalize_table(value) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    if isinstance(value, list):
        if len(value) == 2 and isinstance(value[0], list) and isinstance(value[1], list):
            return {"columns": value[0], "rows": value[1]}
        for item in value:
            if isinstance(item, dict):
                return item
        return {}
    return {}


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
    dataset = load_dataset(
        config.dataset_name,
        split="train",
        cache_dir=config.cache_dir,
    )

    if config.subset_size:
        dataset = dataset.select(range(min(config.subset_size, len(dataset))))

    def format_example(example):
        """Format a single example for GRPO."""
        question_text = _normalize_text(example.get("query"))
        prompt = format_conversation(question_text)

        image = example.get("image")
        if image is not None:
            image = process_image_for_model(image)

        return {
            "prompt": prompt,
            "images": [image] if image is not None else [],
            "label": _normalize_label(example.get("label")),
            "table": _normalize_table(example.get("table")),
            "chart_type": _normalize_text(example.get("chart_type")),
            "reasoning": _normalize_text(example.get("reasoning")),
        }

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
    dataset_map = {
        "chartqa": "HuggingFaceM4/ChartQA",
        "evochart": "lmms-lab/EvoChart",
        "chartqapro": "charxiv/ChartQAPro",
    }

    hf_name = dataset_map.get(dataset_name.lower(), dataset_name)

    dataset = load_dataset(
        hf_name,
        split=split,
        cache_dir=cache_dir,
    )

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
        dataset: Dataset to load
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
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
