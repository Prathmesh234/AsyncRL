"""
DataLoader for consuming GRPO generation batches.

Two modes:
  Queue mode  – reads from a shared BufferQueue (in-memory deque).
  File mode   – monitors a directory for new JSONL batches (legacy).

Queue mode is preferred: it removes disk I/O and gives tighter backpressure
control via the queue's maxlen.
"""

import json
import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

# Allow importing BufferQueue from the serving root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from buffer_queue import BufferQueue


class DataLoader:
    """Loads and manages GRPO generation batches for the trainer."""

    def __init__(
        self,
        generations_dir: str,
        buffer_queue: Optional[BufferQueue] = None,
        batch_size: int = 10,
    ):
        """
        Initialize DataLoader.

        Args:
            generations_dir: Directory containing JSONL generation files
                             (used only in file mode).
            buffer_queue:    Shared BufferQueue instance.  When provided the
                             loader operates in queue mode and ignores the
                             filesystem.
            batch_size:      Number of groups to pull per training step in
                             queue mode.  In file mode one file = one batch.
        """
        self.buffer_queue = buffer_queue
        self.batch_size = batch_size

        # File-mode state
        self.generations_dir = Path(generations_dir)
        self.generations_dir.mkdir(parents=True, exist_ok=True)
        self.processed_files: set = set()

    # ------------------------------------------------------------------
    # Primary consumer API
    # ------------------------------------------------------------------

    def get_next_batch(self) -> Optional[List[Dict[str, Any]]]:
        """
        Return the next batch of grouped records, or None if no data is ready.

        Queue mode:  pulls up to `batch_size` groups from BufferQueue; returns
                     None when the queue is empty so the caller can back-off.
        File mode:   reads the next unprocessed JSONL file as before.
        """
        if self.buffer_queue is not None:
            return self._get_from_queue()
        return self._get_from_file()

    # ------------------------------------------------------------------
    # Queue mode
    # ------------------------------------------------------------------

    def _get_from_queue(self) -> Optional[List[Dict[str, Any]]]:
        """Pull up to batch_size groups from the BufferQueue."""
        batch = self.buffer_queue.get_batch(self.batch_size)
        return batch if batch else None

    # ------------------------------------------------------------------
    # File mode (legacy)
    # ------------------------------------------------------------------

    def _get_from_file(self) -> Optional[List[Dict[str, Any]]]:
        """Get the next unprocessed JSONL batch from disk."""
        available_files = sorted(self.generations_dir.glob("batch_*.jsonl"))

        for file in available_files:
            if file not in self.processed_files:
                groups = self._load_jsonl(file)
                self.processed_files.add(file)
                return groups

        return None

    def _load_jsonl(self, filepath: Path) -> List[Dict[str, Any]]:
        """Load a JSONL file and group completions by group_id."""
        from collections import defaultdict

        records = []
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

        groups = defaultdict(lambda: {"prompt": None, "prompt_ids": None, "completions": []})

        for r in records:
            gid = r.get("group_id") or r.get("gen_id")

            if groups[gid]["prompt"] is None:
                groups[gid]["prompt"] = r.get("prompt")
                groups[gid]["prompt_ids"] = r.get("prompt_ids")

            if "completion" in r:
                groups[gid]["completions"].append(r["completion"])
            elif "completions" in r:
                groups[gid]["completions"].extend(r["completions"])

        return [{"gen_id": gid, **data} for gid, data in groups.items()]

    # ------------------------------------------------------------------
    # Inspection helpers (mirrors the old DataLoader API)
    # ------------------------------------------------------------------

    def count_available(self) -> int:
        """Number of groups / files currently available for training."""
        if self.buffer_queue is not None:
            return self.buffer_queue.count_available()
        available_files = sorted(self.generations_dir.glob("batch_*.jsonl"))
        return len([f for f in available_files if f not in self.processed_files])

    def count_processed(self) -> int:
        """Number of JSONL files processed (file mode only)."""
        return len(self.processed_files)

    def reset(self):
        """Reset processed-files tracker (file mode only)."""
        self.processed_files.clear()

    def peek_next_batch(self) -> Optional[List[Dict[str, Any]]]:
        """
        Peek at the next batch without consuming it.

        Queue mode:  not meaningful (consuming is atomic); returns None.
        File mode:   returns the next file's contents without marking it done.
        """
        if self.buffer_queue is not None:
            return None  # Can't peek a deque non-destructively in queue mode
        available_files = sorted(self.generations_dir.glob("batch_*.jsonl"))
        for file in available_files:
            if file not in self.processed_files:
                return self._load_jsonl(file)
        return None

    def peek_next_batch_file(self) -> Optional[Path]:
        """
        Return the path to the next unprocessed batch file (file mode only).
        Returns None in queue mode.
        """
        if self.buffer_queue is not None:
            return None
        available_files = sorted(self.generations_dir.glob("batch_*.jsonl"))
        for file in available_files:
            if file not in self.processed_files:
                return file
        return None
