"""
Utility functions for SAGE evaluation.
"""

import json
import os
from pathlib import Path
from typing import Iterable, Any, Union
import numpy as np


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    import random
    import torch
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def load_jsonl(file: Union[str, Path]) -> Iterable[Any]:
    """Load JSONL file line by line."""
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                yield json.loads(line)
            except Exception as e:
                print(f"Error loading line: {line[:100]}... Error: {e}")
                continue


def save_jsonl(samples: list, save_path: Union[str, Path]):
    """Save samples to JSONL file."""
    folder = os.path.dirname(save_path)
    os.makedirs(folder, exist_ok=True)
    
    with open(save_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"Saved {len(samples)} samples to {save_path}")


def load_json(file: Union[str, Path]) -> dict:
    """Load JSON file."""
    with open(file, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: dict, save_path: Union[str, Path]):
    """Save data to JSON file."""
    folder = os.path.dirname(save_path)
    os.makedirs(folder, exist_ok=True)
    
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"Saved metrics to {save_path}")
