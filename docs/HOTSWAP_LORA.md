# Hot-Swap LoRA Policy Updates

This document describes the hot-swap LoRA policy feature in AsyncRL, which enables **zero-downtime policy updates** during continuous reinforcement learning training.

> **History note:** an earlier design passed a `lora_request` object in the
> request body (`extra_body`). vLLM's OpenAI-compatible server has no such
> parameter — `LoRARequest` exists only in the offline Python API — so those
> requests silently kept using the adapter loaded at server startup. The
> current design uses vLLM's real dynamic-LoRA endpoints, described below.

## Overview

1. DisGenerator generates rollouts using the current policy.
2. DisTrainer trains on collected rollouts and saves a new policy under
   `models/policy-N-timestamp/`, retargets the `latest_adapter` symlink, and
   writes a `.policy_ready` signal.
3. DisGenerator's `PolicyManager` detects the new policy and **loads it onto
   every vLLM server** (prefill *and* decode) via `POST /v1/load_lora_adapter`,
   under a fresh versioned adapter name (`grpo-adapter-v1`, `-v2`, ...).
4. Once **all** servers accept the adapter, new requests carry the new name in
   their `model` field. In-flight requests finish on the previous name, which
   stays loaded; the version before that is unloaded to bound memory.
5. No downtime, no restarts, no manual intervention.

## Architecture

```
┌─────────────────┐
│   DisTrainer    │
│  1. Trains      │
│  2. Saves       │────► models/policy-N-timestamp/
│     adapter     │        ├── adapter_config.json
│  3. Updates     │        └── adapter_model.safetensors
│     symlink     │
│  4. Signals     │────► models/.policy_ready
└─────────────────┘
         ▼
┌─────────────────┐        POST /v1/load_lora_adapter
│ DisGenerator    │      ┌──────────────────────────────┐
│ PolicyManager   │─────►│ vLLM prefill (port 20001)    │
│  - polls models/│─────►│ vLLM decode  (port 20002)    │
│  - versioned    │      │  {"lora_name":"grpo-adapter-v2",
│    names: -vN   │      │   "lora_path":"/…/policy-2-…"}│
│  - swap only if │      └──────────────────────────────┘
│    ALL loads OK │
│                 │        model = "grpo-adapter-v2"
│ Orchestrator    │─────► every new request selects the
│                 │        adapter via the `model` field
└─────────────────┘
```

## How adapter selection works in vLLM

- **Per-request selection** is via the OpenAI `model` parameter — the adapter
  name acts like a model name. There is no per-request `lora_request` body
  parameter in the HTTP API.
- **Startup registration:** the start scripts register the initial adapter:
  `--lora-modules "grpo-adapter=/path/to/latest_adapter"`.
- **Runtime loading:** with `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True` (exported
  by `start_prefill.sh` / `start_decode.sh`), the server exposes
  `POST /v1/load_lora_adapter {"lora_name", "lora_path"}` and
  `POST /v1/unload_lora_adapter {"lora_name"}`.

## Configuration

### DisGenerator (environment variables)

```bash
# Enable/disable hot-swap (default: true)
ENABLE_HOTSWAP=true

# Poll interval in seconds (default: 5.0)
POLICY_POLL_INTERVAL=5.0

# Models directory (default: auto-detected)
DISTRAINER_MODELS_DIR=/path/to/DisTrainer/models

# vLLM servers to push adapters to. When unset, discovered from the
# proxy's GET /servers endpoint (recommended).
VLLM_SERVER_URLS=http://localhost:20001,http://localhost:20002
```

### vLLM servers

Both prefill and decode servers need (handled by the start scripts):

```bash
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True

vllm serve Qwen/Qwen3-4B-Thinking-2507 \
    --enable-lora \
    --lora-modules "grpo-adapter=/path/to/latest_adapter" \
    --max-loras 3 \        # >= PolicyManager.KEEP_LOADED (2) + headroom
    --max-cpu-loras 5
```

## Swap lifecycle

| Event | Adapter names loaded | Requests use |
|---|---|---|
| Startup | `grpo-adapter` | `grpo-adapter` |
| policy-1 saved | + `grpo-adapter-v1` | `grpo-adapter-v1` |
| policy-2 saved | + `grpo-adapter-v2`, − `grpo-adapter` | `grpo-adapter-v2` |
| policy-3 saved | + `grpo-adapter-v3`, − `grpo-adapter-v1` | `grpo-adapter-v3` |

Failure handling: if **any** server rejects `load_lora_adapter`, the swap is
aborted — requests keep the old adapter name and the watcher retries on the
next poll. This keeps prefill and decode consistent: a request routed through
both phases always references a name both servers know.

## Monitoring

**DisGenerator (PolicyManager):**
```
📥 Initial policy: model_name=grpo-adapter, path=policy-0-initial (registered at server startup)
Loaded grpo-adapter-v1 on http://localhost:20001
Loaded grpo-adapter-v1 on http://localhost:20002
🔄 HOT-SWAP: grpo-adapter → grpo-adapter-v1 (path=policy-1-20260701_143022) on 2 servers
Unloaded grpo-adapter on http://localhost:20001
```

**Check current policy:**
```python
policy_manager.get_current_policy_info()
# {'status': 'active', 'model_name': 'grpo-adapter-v2', 'version': 2,
#  'path': '/…/policy-2-…', 'servers': ['http://localhost:20001', …]}
```

**Check what a server has loaded:**
```bash
curl http://localhost:20001/v1/models | jq '.data[].id'
# "Qwen/Qwen3-4B-Thinking-2507", "grpo-adapter", "grpo-adapter-v1", ...
```

## Troubleshooting

### New policies saved but requests still use the old adapter
1. Verify `ENABLE_HOTSWAP=true` on the client.
2. Check PolicyManager logs for `load_lora_adapter` errors — a single failing
   server blocks every swap by design.
3. Verify the servers were started with `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True`
   (the endpoints return 404/405 otherwise).
4. Verify the server list: `curl http://localhost:10001/servers`.

### `load_lora_adapter` returns 400
- The adapter directory must contain `adapter_config.json` +
  `adapter_model.safetensors`. If DisTrainer is mid-write, the next poll will
  retry — the `.policy_ready` signal exists precisely so consumers don't read
  half-written checkpoints.
- Expert-layer LoRA (MoE) requires vLLM >= 0.15.

### GPU out of memory after several swaps
- Old versions are unloaded automatically (current + previous stay loaded).
  If you raised `PolicyManager.KEEP_LOADED`, raise `--max-loras` to match.

## References

- [vLLM LoRA adapters (dynamic loading)](https://docs.vllm.ai/en/latest/features/lora/)
- [AsyncRL architecture](../README.md)
