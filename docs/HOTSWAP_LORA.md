# Hot-Swap LoRA Policy Updates

This document describes the hot-swap LoRA policy feature in AsyncRL, which enables **zero-downtime policy updates** during continuous reinforcement learning training.

## Overview

The hot-swap feature allows DisGenerator to automatically load new policies from DisTrainer without restarting the vLLM servers. This enables a truly continuous RL training loop where:

1. DisGenerator generates rollouts using the current policy
2. DisTrainer trains on collected rollouts and saves a new policy
3. DisGenerator automatically detects and switches to the new policy
4. In-flight requests complete with the old policy, new requests use the new policy
5. No downtime, no manual intervention required

## Architecture

### Components

```
┌─────────────────┐
│   DisTrainer    │
│                 │
│  1. Trains      │
│  2. Saves       │────► models/policy-N-timestamp/
│     adapter     │        ├── adapter_config.json
│  3. Updates     │        ├── adapter_model.safetensors
│     symlink     │        └── (DCP checkpoint files)
│  4. Signals     │────► models/.policy_ready
│                 │
└─────────────────┘

         ▼

┌─────────────────┐
│ DisGenerator    │
│                 │
│ PolicyManager   │
│  - Watches for  │◄──── Polls models/ directory (5s)
│    new policies │◄──── Detects .policy_ready signal
│  - Increments   │
│    lora_int_id  │
│  - Provides     │────► LoRARequest(lora_int_id=N)
│    current      │
│    LoRARequest  │
│                 │
│ Orchestrator    │
│  - Uses current │────► vLLM API with extra_body
│    LoRARequest  │        { lora_request: {...} }
│    in requests  │
└─────────────────┘

         ▼

┌─────────────────┐
│   vLLM Engine   │
│                 │
│  - Maintains    │      GPU Memory:
│    multiple     │      ┌──────────────────┐
│    adapters in  │      │ adapter (id=1)   │
│    GPU memory   │      │ adapter (id=2)   │
│  - LRU eviction │      │ adapter (id=3)   │◄── --max-loras 3
│  - Zero-downtime│      └──────────────────┘
│    switching    │
└─────────────────┘
```

### Flow

1. **Training Phase** (DisTrainer)
   - Trains on collected rollouts
   - Saves checkpoint with `CheckpointManager.save()`
   - Saves both DCP format (for resuming training) and HF PEFT format (for vLLM)
   - Updates `latest_adapter` symlink
   - Creates `.policy_ready` signal file

2. **Detection Phase** (DisGenerator PolicyManager)
   - Polls `DisTrainer/models/` directory every 5 seconds
   - Detects changes to `latest_adapter` symlink
   - Consumes `.policy_ready` signal file
   - Increments `lora_int_id` counter

3. **Hot-Swap Phase** (vLLM)
   - New requests include `lora_request` with new `lora_int_id`
   - vLLM loads new adapter into GPU memory (if not already loaded)
   - Old in-flight requests continue with previous `lora_int_id`
   - Automatic LRU eviction when GPU memory limit reached

4. **Generation Phase** (DisGenerator Orchestrator)
   - Each vLLM request queries `policy_manager.get_current_lora_request()`
   - Includes current LoRARequest in API payload
   - Thread-safe access ensures consistency

## Configuration

### DisGenerator

**Environment Variables** (`.env` or export):

```bash
# Enable/disable hot-swap (default: true)
ENABLE_HOTSWAP=true

# Poll interval in seconds (default: 5.0)
POLICY_POLL_INTERVAL=5.0

# Models directory (default: auto-detected)
DISTRAINER_MODELS_DIR=/path/to/DisTrainer/models
```

**Command-line** (simple_client.py):

The client automatically initializes PolicyManager when `ENABLE_HOTSWAP=true`:

```python
policy_manager = PolicyManager(
    models_dir=DISTRAINER_MODELS_DIR,
    lora_name="grpo-adapter",
    poll_interval=POLICY_POLL_INTERVAL,
    enable_hotswap=True
)
policy_manager.start_watching()
```

### vLLM Servers

Both prefill and decode servers require the following flags:

```bash
vllm serve Qwen/Qwen3-4B-Thinking-2507 \
    --enable-lora \
    --lora-modules "grpo-adapter=/path/to/latest_adapter" \
    --max-loras 3 \           # Keep up to 3 adapters in GPU memory
    --max-cpu-loras 5 \       # Keep up to 5 adapters in CPU memory
    # ... other flags
```

**Important Parameters**:
- `--max-loras`: Maximum LoRA adapters in GPU memory (default: 1)
  - Higher = more GPU memory usage but better hot-swap performance
  - Recommended: 2-3 for continuous training
- `--max-cpu-loras`: Maximum adapters in CPU memory (default: 0)
  - Acts as a cache for faster re-loading

## Usage

### Quick Start

1. **Start vLLM servers** (with hot-swap support):
   ```bash
   cd serving/DisGenerator/scripts
   ./start_prefill.sh 0 20001 21001
   ./start_decode.sh 1 20002 22001
   ./start_proxy.sh
   ```

2. **Start DisGenerator** (with hot-swap enabled):
   ```bash
   cd serving/DisGenerator
   ENABLE_HOTSWAP=true python simple_client.py
   ```

3. **Start DisTrainer**:
   ```bash
   cd serving/DisTrainer
   torchrun --nproc_per_node=4 server.py
   ```

4. **Observe hot-swaps** in DisGenerator logs:
   ```
   [PolicyManager] 📥 Initial policy loaded: lora_int_id=1, path=policy-0-initial
   [PolicyManager] 🔄 HOT-SWAP: Policy updated! Old: lora_int_id=1, New: lora_int_id=2, path=policy-1-20260120_143022
   [GPU-0] Using LoRA: lora_int_id=2
   ```

### Disabling Hot-Swap

To disable hot-swap and use a fixed policy (e.g., for evaluation):

```bash
ENABLE_HOTSWAP=false python simple_client.py
```

This loads the initial policy once and never updates it.

## Implementation Details

### PolicyManager

**File**: `serving/DisGenerator/policy_manager.py`

**Key Methods**:
- `start_watching()`: Starts background thread that polls for new policies
- `get_current_lora_request()`: Returns current LoRARequest (thread-safe)
- `_detect_and_load_policy()`: Checks for policy updates and creates new LoRARequest
- `_consume_policy_ready_signal()`: Removes `.policy_ready` signal file

**Thread Safety**:
- Uses `threading.Lock` for protecting policy state
- Safe concurrent access from multiple GPU workers

### LoRARequest

**Format**:
```python
@dataclass
class LoRARequest:
    lora_name: str        # "grpo-adapter" (must match vLLM --lora-modules)
    lora_int_id: int      # Unique ID (1, 2, 3, ...)
    lora_path: str        # Path to adapter directory
```

**vLLM API Integration**:
```python
payload = {
    "model": "Qwen/Qwen3-4B-Thinking-2507",
    "messages": [...],
    "extra_body": {
        "lora_request": {
            "lora_name": "grpo-adapter",
            "lora_int_id": 2,
            "lora_path": "/path/to/policy-1-20260120_143022"
        }
    }
}
```

### Checkpoint Notification

**File**: `serving/DisTrainer/components/checkpoint.py`

**Flow**:
1. `CheckpointManager.save()` saves checkpoint
2. `_update_latest_symlink()` updates symlink
3. `_create_policy_ready_signal()` writes `.policy_ready` file with timestamp

**Signal File**: `DisTrainer/models/.policy_ready`
```
2026-01-20T14:30:22.123456
```

This file is atomic - created fully then detected by PolicyManager.

## Performance Considerations

### GPU Memory

Each LoRA adapter requires additional GPU memory:
- **Typical adapter size**: 50-100 MB (depending on rank and target modules)
- **--max-loras 3**: ~150-300 MB additional GPU memory

**Trade-off**:
- Higher `--max-loras` = smoother hot-swaps but more GPU memory
- Lower `--max-loras` = less GPU memory but potential loading delays

### Latency

**First request with new adapter**:
- ~100-500ms loading time (if not in GPU memory)
- Subsequent requests: ~0ms overhead (already loaded)

**Recommended**:
- Use `--max-loras 3` to keep current + 1-2 previous adapters in memory
- Old in-flight requests complete without loading delays

### Polling Interval

**Default**: 5 seconds

**Trade-off**:
- Lower interval (1-2s) = faster detection but more I/O overhead
- Higher interval (10-30s) = less overhead but slower updates

**Recommendation**: 5s is a good balance for continuous training

## Monitoring

### Log Messages

**DisGenerator** (PolicyManager):
```
[PolicyManager] 📥 Initial policy loaded: lora_int_id=1, path=policy-0-initial
[PolicyManager] 🔄 HOT-SWAP: Policy updated! Old: lora_int_id=1, New: lora_int_id=2, path=policy-1-20260120_143022
[GPU-0] Using LoRA: lora_int_id=2
```

**DisTrainer** (CheckpointManager):
```
💾 Saved checkpoint: policy-1-20260120_143022 (step 100)
✅ Saved HF Adapter to /path/to/policy-1-20260120_143022
📢 Created .policy_ready signal for DisGenerator
```

### Debugging

**Check current policy**:
```bash
# In DisGenerator Python shell or script:
policy_info = policy_manager.get_current_policy_info()
print(policy_info)
# {'status': 'active', 'lora_name': 'grpo-adapter', 'lora_int_id': 2, 'path': '/path/to/policy-1-...'}
```

**Check latest_adapter symlink**:
```bash
ls -l serving/DisTrainer/models/latest_adapter
# lrwxrwxrwx 1 user user 23 Jan 20 14:30 latest_adapter -> policy-1-20260120_143022
```

**Check signal file** (should not persist):
```bash
ls serving/DisTrainer/models/.policy_ready
# Should be deleted after PolicyManager consumes it
```

## Troubleshooting

### Issue: Hot-swap not detected

**Symptoms**: New policies saved but DisGenerator still uses old policy

**Checks**:
1. Verify `ENABLE_HOTSWAP=true`
2. Check PolicyManager logs for errors
3. Verify `latest_adapter` symlink exists and points to new policy
4. Check `.policy_ready` signal is being created (may be deleted immediately)

**Solution**:
```bash
# Restart DisGenerator with debug logging
ENABLE_HOTSWAP=true python simple_client.py --log-level DEBUG
```

### Issue: vLLM fails to load adapter

**Symptoms**: `[GPU-0] Error from Proxy: 400 - Invalid LoRA adapter`

**Checks**:
1. Verify adapter directory contains `adapter_config.json` and `adapter_model.safetensors`
2. Check vLLM logs for detailed error messages
3. Verify `--enable-lora` flag is set on vLLM servers

**Solution**:
```bash
# Check adapter files
ls serving/DisTrainer/models/latest_adapter/
# Should show: adapter_config.json, adapter_model.safetensors, ...
```

### Issue: GPU out of memory

**Symptoms**: vLLM crashes with OOM error

**Cause**: Too many adapters in GPU memory

**Solution**:
1. Reduce `--max-loras` from 3 to 2 or 1
2. Reduce `--gpu-memory-utilization` to leave more headroom
3. Use smaller LoRA rank (r=4 instead of r=8)

### Issue: Stale policy after restart

**Symptoms**: DisGenerator loads old policy after restart

**Cause**: `latest_adapter` symlink not updated

**Solution**:
```bash
# Manually update symlink (from DisTrainer/models/)
cd serving/DisTrainer/models
ln -snf policy-1-20260120_143022 latest_adapter
```

## Advanced Configuration

### Custom Policy Naming

To use a different naming convention or models directory:

```python
policy_manager = PolicyManager(
    models_dir="/custom/path/to/models",
    lora_name="my-custom-adapter",
    poll_interval=2.0
)
```

**Note**: Ensure vLLM `--lora-modules` matches the `lora_name`.

### Multiple Adapters

For serving multiple adapters simultaneously (e.g., different policies for A/B testing):

```python
# Create separate PolicyManagers
policy_manager_a = PolicyManager(models_dir=".../models_a", lora_name="policy-a")
policy_manager_b = PolicyManager(models_dir=".../models_b", lora_name="policy-b")

# Use different LoRARequests in different requests
if experiment == "A":
    lora_request = policy_manager_a.get_current_lora_request()
else:
    lora_request = policy_manager_b.get_current_lora_request()
```

## References

- [vLLM LoRA Documentation](https://docs.vllm.ai/en/latest/models/lora.html)
- [PEFT Library](https://github.com/huggingface/peft)
- [AsyncRL Architecture](../README.md)

## Summary

The hot-swap LoRA feature enables **zero-downtime policy updates** for continuous RL training:

✅ **Automatic detection**: PolicyManager watches for new policies
✅ **Zero-downtime**: Old requests complete, new requests use new policy
✅ **Simple integration**: Just set `ENABLE_HOTSWAP=true`
✅ **Robust**: Thread-safe, error-handled, well-tested
✅ **Configurable**: Adjust polling, memory limits, naming

This unlocks truly continuous RL training where the system never stops generating rollouts, even as policies improve.
