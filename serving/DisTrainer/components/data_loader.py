"""
DataLoader for consuming JSONL generation files.
Monitors a directory for new JSONL batches from the Generator.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional


class DataLoader:
    """Loads and manages JSONL generation files."""
    
    def __init__(self, generations_dir: str):
        """
        Initialize DataLoader.
        
        Args:
            generations_dir: Directory containing JSONL generation files
        """
        self.generations_dir = Path(generations_dir)
        self.generations_dir.mkdir(parents=True, exist_ok=True)
        self.processed_files: set = set()
    
    def get_next_batch(self) -> Optional[List[Dict[str, Any]]]:
        """
        Get the next unprocessed JSONL batch.
        
        Returns:
            List of generation groups, or None if no new data
        """
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
        
        # Read all records
        records = []
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        
        # Hashmap to collect completions by group_id
        groups = defaultdict(lambda: {"prompt": None, "prompt_ids": None, "completions": []})
        
        for r in records:
            gid = r.get("group_id") or r.get("gen_id")
            
            # Store prompt info (only once per group)
            if groups[gid]["prompt"] is None:
                groups[gid]["prompt"] = r.get("prompt")
                groups[gid]["prompt_ids"] = r.get("prompt_ids")
            
            # Collect completions (handle both formats)
            if "completion" in r:
                groups[gid]["completions"].append(r["completion"])
            elif "completions" in r:
                groups[gid]["completions"].extend(r["completions"])
        
        # Convert to list
        return [{"gen_id": gid, **data} for gid, data in groups.items()]
    
    def count_available(self) -> int:
        """Count number of unprocessed JSONL files."""
        available_files = sorted(self.generations_dir.glob("batch_*.jsonl"))
        unprocessed = [f for f in available_files if f not in self.processed_files]
        return len(unprocessed)
    
    def count_processed(self) -> int:
        """Count number of processed JSONL files."""
        return len(self.processed_files)
    
    def reset(self):
        """Reset processed files tracker (to reprocess all data)."""
        self.processed_files.clear()
    
    def peek_next_batch(self) -> Optional[List[Dict[str, Any]]]:
        """
        Peek at the next batch without marking it as processed.
        
        Returns:
            List of generation groups, or None if no new data
        """
        available_files = sorted(self.generations_dir.glob("batch_*.jsonl"))
        
        for file in available_files:
            if file not in self.processed_files:
                return self._load_jsonl(file)
        
        return None
    
    def peek_next_batch_file(self) -> Optional[Path]:
        """
        Get the path to the next unprocessed batch file.
        
        Returns:
            Path to next batch file, or None if no new data
        """
        available_files = sorted(self.generations_dir.glob("batch_*.jsonl"))
        
        for file in available_files:
            if file not in self.processed_files:
                return file
        
        return None

