"""
Utility functions for the DisGenerator.
File I/O and general helpers.
"""

import os
import glob
import json
from typing import List, Dict


def get_next_batch_number(output_dir: str) -> int:
    """
    Find the next available batch number by checking existing files.
    
    Args:
        output_dir: Directory containing batch_*.jsonl files
        
    Returns:
        Next available batch number (1-indexed)
    """
    pattern = os.path.join(output_dir, "batch_*.jsonl")
    existing = glob.glob(pattern)
    if not existing:
        return 1
    
    # Extract numbers and find max
    numbers = []
    for f in existing:
        try:
            num = int(os.path.basename(f).replace("batch_", "").replace(".jsonl", ""))
            numbers.append(num)
        except ValueError:
            continue
    
    return max(numbers) + 1 if numbers else 1


def write_batch_file(filepath: str, records: List[Dict]) -> None:
    """
    Write multiple trajectories to a batch file (one per line).
    
    Args:
        filepath: Path to the output JSONL file
        records: List of record dictionaries to write
    """
    with open(filepath, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def ensure_output_dir(output_dir: str) -> str:
    """
    Ensure the output directory exists, creating it if necessary.
    
    Args:
        output_dir: Path to the output directory
        
    Returns:
        Absolute path to the output directory
    """
    abs_path = os.path.abspath(output_dir)
    os.makedirs(abs_path, exist_ok=True)
    return abs_path
