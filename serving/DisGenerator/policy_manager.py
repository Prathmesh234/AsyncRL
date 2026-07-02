"""
PolicyManager for hot-swapping LoRA policies in vLLM.

Enables zero-downtime policy updates using vLLM's dynamic LoRA endpoints
(requires VLLM_ALLOW_RUNTIME_LORA_UPDATING=True on the servers):

  POST /v1/load_lora_adapter    {"lora_name": ..., "lora_path": ...}
  POST /v1/unload_lora_adapter  {"lora_name": ...}

Per-request adapter selection in vLLM's OpenAI-compatible server is done via
the `model` field of the request — there is NO per-request lora_request
parameter. So the flow is:

- The vLLM servers register the initial adapter as `<lora_name>` at startup
  (--lora-modules "<lora_name>=<path to latest_adapter>").
- DisTrainer saves new policies to models/policy-N-timestamp/, retargets the
  latest_adapter symlink, and writes a .policy_ready signal.
- PolicyManager polls for changes. On a new policy it loads the adapter on
  EVERY vLLM server (prefill + decode — both phases of the disaggregated
  proxy must know the name) under a fresh versioned name `<lora_name>-vN`.
- Only after all servers accept the adapter does the current model name
  switch; the orchestrator stamps it into each request's `model` field.
- In-flight requests keep using the previous name (still loaded); the
  version before that is unloaded to bound GPU/CPU adapter memory.
"""

import json
import logging
import threading
import urllib.error
import urllib.request
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger("PolicyManager")


class PolicyManager:
    """
    Manages hot-swapping of LoRA policies for DisGenerator.

    Responsibilities:
    - Track the current policy version and its vLLM adapter name
    - Watch for new policies from DisTrainer
    - Push new adapters to every vLLM server via /v1/load_lora_adapter
    - Retire old adapters via /v1/unload_lora_adapter (keeps current + previous)
    - Provide thread-safe access to the current adapter name for requests
    """

    # How many adapter versions stay loaded on the servers. 2 = current +
    # previous, so in-flight requests on the old name finish cleanly.
    # Keep this <= the servers' --max-loras setting.
    KEEP_LOADED = 2

    def __init__(
        self,
        models_dir: str,
        lora_name: str = "grpo-adapter",
        poll_interval: float = 5.0,
        enable_hotswap: bool = True,
        server_urls: Optional[List[str]] = None,
        request_timeout: float = 60.0,
    ):
        """
        Initialize PolicyManager.

        Args:
            models_dir: Path to DisTrainer/models directory
            lora_name: Base adapter name; must match the name registered via
                       --lora-modules on the vLLM servers
            poll_interval: Seconds between policy checks
            enable_hotswap: If False, the startup adapter is used forever
            server_urls: Base URLs of ALL vLLM servers (prefill + decode),
                         e.g. ["http://localhost:20001", "http://localhost:20002"]
            request_timeout: Timeout for load/unload HTTP calls (adapter
                             loading from disk can take a few seconds)
        """
        self.models_dir = Path(models_dir)
        self.lora_name = lora_name
        self.poll_interval = poll_interval
        self.enable_hotswap = enable_hotswap
        self.server_urls = [u.rstrip("/") for u in (server_urls or [])]
        self.request_timeout = request_timeout

        if self.enable_hotswap and not self.server_urls:
            logger.warning(
                "Hot-swap enabled but no vLLM server URLs configured — "
                "new policies CANNOT be pushed to the servers. Set "
                "VLLM_SERVER_URLS or let the client discover them from the proxy."
            )

        # Thread safety
        self._lock = threading.Lock()

        # Current policy state. Version 0 is the adapter the servers
        # registered at startup under the base lora_name.
        self._version = 0
        self._current_model_name: str = lora_name
        self._current_policy_path: Optional[str] = None

        # Watcher thread
        self._watcher_thread: Optional[threading.Thread] = None
        self._stop_watching = threading.Event()

        # Record the policy the servers started with (no HTTP needed:
        # start scripts resolve the same latest_adapter symlink).
        initial_path = self._get_latest_policy_path()
        if initial_path is None:
            logger.warning("No policy found at startup. Requests will use the base model name.")
        else:
            self._current_policy_path = initial_path
            logger.info(
                f"📥 Initial policy: model_name={self._current_model_name}, "
                f"path={Path(initial_path).name} (registered at server startup)"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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

    def get_current_model_name(self) -> str:
        """
        Adapter name to put in the `model` field of vLLM requests.

        Thread-safe. Returns the base lora_name until the first hot-swap.
        """
        with self._lock:
            return self._current_model_name

    def get_current_policy_info(self) -> dict:
        """Get current policy information (for logging/debugging)."""
        with self._lock:
            return {
                "status": "active" if self._current_policy_path else "no_policy",
                "model_name": self._current_model_name,
                "version": self._version,
                "path": self._current_policy_path,
                "servers": list(self.server_urls),
            }

    # ------------------------------------------------------------------
    # Watcher internals
    # ------------------------------------------------------------------

    def _policy_watcher(self):
        """Background thread that polls for new policies and swaps them in."""
        logger.info("Policy watcher started")

        while not self._stop_watching.is_set():
            try:
                self._detect_and_swap_policy()
            except Exception as e:
                logger.error(f"Error in policy watcher: {e}", exc_info=True)
            # Sleep with interrupt support
            self._stop_watching.wait(timeout=self.poll_interval)

    def _detect_and_swap_policy(self):
        """
        Check for a new policy; if found, push it to all vLLM servers and
        switch the current model name only after every server accepted it.
        A failed push leaves state unchanged so the next poll retries.
        """
        latest_path = self._get_latest_policy_path()
        if latest_path is None:
            return

        with self._lock:
            if latest_path == self._current_policy_path:
                return  # No change
            old_name = self._current_model_name
            new_version = self._version + 1
        new_name = f"{self.lora_name}-v{new_version}"

        if not self.server_urls:
            logger.error(
                f"New policy detected ({Path(latest_path).name}) but no server "
                f"URLs configured — cannot hot-swap. Still serving {old_name}."
            )
            return

        # Load the new adapter on every server (outside the lock: this is
        # slow I/O, and readers must keep getting the old name meanwhile).
        if not self._load_adapter_everywhere(new_name, latest_path):
            logger.error(
                f"Hot-swap aborted: {new_name} failed to load on at least one "
                f"server. Still serving {old_name}; will retry in {self.poll_interval}s."
            )
            return

        with self._lock:
            self._version = new_version
            self._current_model_name = new_name
            self._current_policy_path = latest_path

        logger.info(
            f"🔄 HOT-SWAP: {old_name} → {new_name} "
            f"(path={Path(latest_path).name}) on {len(self.server_urls)} servers"
        )

        self._consume_policy_ready_signal()

        # Retire the version before the previous one (keep current + previous
        # loaded for in-flight requests).
        retired = new_version - self.KEEP_LOADED
        if retired >= 0:
            retired_name = self.lora_name if retired == 0 else f"{self.lora_name}-v{retired}"
            self._unload_adapter_everywhere(retired_name)

    # ------------------------------------------------------------------
    # vLLM dynamic LoRA HTTP calls
    # ------------------------------------------------------------------

    def _load_adapter_everywhere(self, name: str, path: str) -> bool:
        """POST /v1/load_lora_adapter on every server. True if all succeeded."""
        ok = True
        for base in self.server_urls:
            if not self._post_json(
                f"{base}/v1/load_lora_adapter",
                {"lora_name": name, "lora_path": path},
            ):
                ok = False
            else:
                logger.info(f"Loaded {name} on {base}")
        return ok

    def _unload_adapter_everywhere(self, name: str):
        """POST /v1/unload_lora_adapter on every server (best effort)."""
        for base in self.server_urls:
            if self._post_json(
                f"{base}/v1/unload_lora_adapter",
                {"lora_name": name},
            ):
                logger.info(f"Unloaded {name} on {base}")
            else:
                logger.warning(f"Failed to unload {name} on {base} (non-fatal)")

    def _post_json(self, url: str, payload: dict) -> bool:
        """POST a JSON payload; True on 2xx."""
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"}, method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=self.request_timeout) as resp:
                return 200 <= resp.status < 300
        except urllib.error.HTTPError as e:
            body = e.read()[:300]
            logger.error(f"{url} -> HTTP {e.code}: {body!r}")
            return False
        except Exception as e:
            logger.error(f"{url} -> {e}")
            return False

    # ------------------------------------------------------------------
    # Policy detection (unchanged from the original design)
    # ------------------------------------------------------------------

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
