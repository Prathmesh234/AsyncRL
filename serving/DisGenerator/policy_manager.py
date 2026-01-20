"""
PolicyManager for hot-swapping LoRA policies in vLLM.

Enables zero-downtime policy updates by loading new LoRA adapters via
LoRARequest with unique lora_int_id values. vLLM maintains multiple
adapters in GPU memory with automatic LRU eviction.

Architecture:
- DisTrainer saves new policies to models/policy-N-timestamp/
- DisTrainer updates latest_adapter symlink and writes .policy_ready signal
- PolicyManager watches for new policies and creates new LoRARequest objects
- Batch orchestrator uses current LoRARequest for all new vLLM requests
- Old in-flight requests complete with previous policy automatically
"""

import os
import logging
import threading
import time
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger("PolicyManager")


@dataclass
class LoRARequest:
    """
    LoRA request configuration for vLLM.

    vLLM uses this to identify and load specific LoRA adapters.
    Each adapter needs a unique lora_int_id.
    """
    lora_name: str  # Identifier for this adapter (e.g., "grpo-adapter")
    lora_int_id: int  # Unique integer ID for this adapter version
    lora_path: str  # Path to the adapter directory

    def to_dict(self) -> dict:
        """Convert to dictionary for vLLM API."""
        return {
            "lora_name": self.lora_name,
            "lora_int_id": self.lora_int_id,
            "lora_path": self.lora_path
        }


class PolicyManager:
    """
    Manages hot-swapping of LoRA policies for DisGenerator.

    Responsibilities:
    - Track current policy version and corresponding LoRARequest
    - Watch for new policies from DisTrainer
    - Provide thread-safe access to current LoRARequest
    - Automatically increment lora_int_id for new policies
    """

    def __init__(
        self,
        models_dir: str,
        lora_name: str = "grpo-adapter",
        poll_interval: float = 5.0,
        enable_hotswap: bool = True
    ):
        """
        Initialize PolicyManager.

        Args:
            models_dir: Path to DisTrainer/models directory
            lora_name: Name identifier for the LoRA adapter
            poll_interval: Seconds between policy checks
            enable_hotswap: If False, policy is loaded once at startup
        """
        self.models_dir = Path(models_dir)
        self.lora_name = lora_name
        self.poll_interval = poll_interval
        self.enable_hotswap = enable_hotswap

        # Thread safety
        self._lock = threading.Lock()

        # Current policy state
        self._current_policy_path: Optional[str] = None
        self._current_lora_request: Optional[LoRARequest] = None
        self._lora_int_id_counter = 1  # Start from 1 (0 reserved for no adapter)

        # Watcher thread
        self._watcher_thread: Optional[threading.Thread] = None
        self._stop_watching = threading.Event()

        # Initialize with current policy
        self._detect_and_load_policy(initial=True)

    def start_watching(self):
        """Start the policy watcher thread."""
        if not self.enable_hotswap:
            logger.info("Hot-swap disabled. Policy will not be updated automatically.")
            return

        if self._watcher_thread is not None and self._watcher_thread.is_alive():
            logger.warning("Policy watcher already running")
            return

        self._stop_watching.clear()
        self._watcher_thread = threading.Thread(
            target=self._policy_watcher,
            daemon=True,
            name="PolicyWatcher"
        )
        self._watcher_thread.start()
        logger.info(f"Started policy watcher (poll interval: {self.poll_interval}s)")

    def stop_watching(self):
        """Stop the policy watcher thread."""
        if self._watcher_thread is None:
            return

        logger.info("Stopping policy watcher...")
        self._stop_watching.set()
        self._watcher_thread.join(timeout=10)
        logger.info("Policy watcher stopped")

    def get_current_lora_request(self) -> Optional[LoRARequest]:
        """
        Get the current LoRARequest for use in vLLM requests.

        Thread-safe. Returns None if no policy is loaded.
        """
        with self._lock:
            return self._current_lora_request

    def get_current_policy_info(self) -> dict:
        """Get current policy information (for logging/debugging)."""
        with self._lock:
            if self._current_lora_request is None:
                return {"status": "no_policy", "lora_int_id": None, "path": None}
            return {
                "status": "active",
                "lora_name": self._current_lora_request.lora_name,
                "lora_int_id": self._current_lora_request.lora_int_id,
                "path": self._current_lora_request.lora_path
            }

    def _policy_watcher(self):
        """
        Background thread that watches for new policies.

        Checks for:
        1. Changes to latest_adapter symlink
        2. .policy_ready signal file from DisTrainer
        """
        logger.info("Policy watcher started")

        while not self._stop_watching.is_set():
            try:
                # Check for policy updates
                self._detect_and_load_policy(initial=False)

                # Sleep with interrupt support
                self._stop_watching.wait(timeout=self.poll_interval)

            except Exception as e:
                logger.error(f"Error in policy watcher: {e}", exc_info=True)
                # Continue watching despite errors
                time.sleep(self.poll_interval)

    def _detect_and_load_policy(self, initial: bool = False):
        """
        Detect if a new policy is available and load it.

        Args:
            initial: True if this is the first load at startup
        """
        # Get latest policy path
        latest_path = self._get_latest_policy_path()

        if latest_path is None:
            if initial:
                logger.warning("No policy found at startup. Will continue without LoRA adapter.")
            return

        # Check if policy has changed
        with self._lock:
            if latest_path == self._current_policy_path:
                # No change
                return

            # New policy detected!
            old_path = self._current_policy_path
            old_lora_id = self._current_lora_request.lora_int_id if self._current_lora_request else None

            # Create new LoRARequest with incremented ID
            new_lora_request = LoRARequest(
                lora_name=self.lora_name,
                lora_int_id=self._lora_int_id_counter,
                lora_path=latest_path
            )

            # Update state
            self._current_policy_path = latest_path
            self._current_lora_request = new_lora_request
            self._lora_int_id_counter += 1

            # Log the hot-swap
            if initial:
                logger.info(
                    f"📥 Initial policy loaded: lora_int_id={new_lora_request.lora_int_id}, "
                    f"path={Path(latest_path).name}"
                )
            else:
                logger.info(
                    f"🔄 HOT-SWAP: Policy updated! "
                    f"Old: lora_int_id={old_lora_id}, New: lora_int_id={new_lora_request.lora_int_id}, "
                    f"path={Path(latest_path).name}"
                )

        # Check and consume .policy_ready signal if it exists
        self._consume_policy_ready_signal()

    def _get_latest_policy_path(self) -> Optional[str]:
        """
        Get the path to the latest policy.

        Checks for 'latest_adapter' symlink (preferred) or scans for highest policy-N.
        """
        if not self.models_dir.exists():
            return None

        # Check for latest_adapter symlink (preferred)
        latest_symlink = self.models_dir / "latest_adapter"
        if latest_symlink.exists():
            # Resolve symlink to actual path
            resolved = latest_symlink.resolve()
            if resolved.exists():
                return str(resolved)

        # Fallback: Scan for highest policy-N
        import re
        max_version = -1
        best_path = None
        pattern = re.compile(r"policy-(\d+)")

        for entry in self.models_dir.iterdir():
            if entry.is_dir():
                match = pattern.match(entry.name)
                if match:
                    version = int(match.group(1))
                    if version > max_version:
                        max_version = version
                        best_path = str(entry)

        return best_path

    def _consume_policy_ready_signal(self):
        """
        Check for and remove .policy_ready signal file.

        DisTrainer creates this file after saving a new checkpoint.
        """
        signal_file = self.models_dir / ".policy_ready"
        if signal_file.exists():
            try:
                signal_file.unlink()
                logger.debug("Consumed .policy_ready signal")
            except Exception as e:
                logger.warning(f"Failed to remove .policy_ready signal: {e}")
