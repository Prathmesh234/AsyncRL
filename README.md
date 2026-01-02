# AsyncRL: Reinforcement Learning with Tools

AsyncRL contains multiple reinforcement learning projects that teach large language models how to use external tools safely and effectively. The repository combines supervised bootstrapping, reinforcement learning with live tooling feedback, and modular serving infrastructure built for asynchronous execution.

Deepwiki link for in depth documentation - https://deepwiki.com/Prathmesh234/AsyncRL/1-overview


Medium Blog - https://medium.com/@ppbhatt500/building-asyncrl-a-multi-tool-reinforcement-learning-pipeline-for-software-engineering-tasks-0fde815ed2b4

<img width="636" height="819" alt="image" src="https://github.com/user-attachments/assets/ff5d88fa-bd3f-4546-b4e3-e43722291b70" />


Distributed Generator and Distributed Trainer 

AsyncRL consists of two primary components operating in a continuous feedback loop:

### Distributed Generator (`DisGenerator`)

A scalable, multi-worker trajectory generator that combines **disaggregated vLLM serving** with **asynchronous tool execution** to maximize GPU throughput while supporting tool-using agents.

Core ideas

* **Disaggregated vLLM Serving**

  * Splits **prefill** and **decode** across GPUs to improve end-to-end throughput.

* **`AsyncBatchOrchestrator` (Dual-Queue Architecture)**

  * **Task Queue:** trajectories waiting for model generation
  * **Tool Queue:** asynchronous tool jobs (web search, code execution, Azure CLI)

Streaming tool interception

* Watches the token stream in real time for tool markers like:

  * `<web> ... </web>`, `<code> ... </code>`, `<azure> ... </azure>`
* When a tool call is detected:

  1. **Pause** generation
  2. **Dispatch** tool execution asynchronously
  3. **Resume** generation with tool results injected back into the context

Concurrency model

* **GPU Workers (4–64):** high-throughput inference via a vLLM proxy (compute-bound). Multiple workers per GPU. 
* **Tool Workers (32+):** I/O-bound tool execution without blocking GPU workers

On-policy data generation

* For each batch, automatically **reloads the latest policy checkpoint** so generated trajectories reflect the current policy state.


### Distributed Trainer (`DisTrainer`)

A production-grade, TorchTitan-inspired trainer for **distributed RL fine-tuning**, built to continuously consume new rollouts and publish versioned policy checkpoints.

Architecture

* **TorchTitan-Inspired Distributed Training**

  * Uses **FSDP2 (Fully Sharded Data Parallelism 2)** for memory-efficient multi-GPU training.

Continuous training loop

* **Automatic Batch Detection**

  * Watches `data/generations/` and automatically starts training when new batches arrive.

GRPO loss pipeline

* **Group-wise Completion Processing**

  * Groups multiple completions by the same prompt
  * Computes **group-relative advantages**
* **Stability & Regularization**

  * Applies **KL regularization** against a **reference policy**
* **Action masking**

  * Uses **action masks** to exclude tool-generated tokens (e.g., tool outputs) from contributing to the loss

Checkpointing

* **Versioned Policy Snapshots**

  * Saves policies as `policy-N-{timestamp}`
  * Uses **Distributed Checkpoint (DCP)** for scalable, fault-tolerant checkpoint writes

Control plane

* **HTTP Control Interface (FastAPI)**

  * `/train` — manually trigger a training run
  * `/status` — monitor health / progress
  * `/checkpoint` — force-save a checkpoint





## Table of Contents

1. [Key Capabilities](#key-capabilities)
2. [Project Structure](#project-structure)
3. [Core Components](#core-components)
   1. [LLM Policy](#llm-policy)
   2. [Tool Containers](#tool-containers)
   3. [Grading Infrastructure](#grading-infrastructure)
4. [Distributed Generator + Trainer (On-Policy)](#distributed-generator--trainer-on-policy)
   1. [How It Works](#how-it-works)
   2. [Data Contract (JSONL)](#data-contract-jsonl)
   3. [Policy Versioning](#policy-versioning)
   4. [Run the Distributed System](#run-the-distributed-system)
   5. [Monitoring](#monitoring)
5. [Running the Containers Locally](#running-the-containers-locally)
6. [Dataset Generation and Imitation Learning](#dataset-generation-and-imitation-learning)
7. [Web RL Environment](#web-rl-environment)
8. [Deployment Notes](#deployment-notes)
9. [Serving Strategy](#serving-strategy)
10. [ToolGRPOTrainer (Legacy Off-Policy Pipeline)](#toolgrpotrainer-legacy-off-policy-pipeline)
11. [Future Work](#future-work)

## Key Capabilities

- Custom integration with Hugging Face TRL's `GRPOTrainer` to unlock multi-turn tool calling during trajectory generation via the
  `ToolGRPOTrainer` overrides in `serving/`.
- LoRA adapters that teach the policy to reason with `<web>`, `<code>`, and `<azure>` execution tags.
- Asynchronous container orchestration using Azure Service Bus Topics and Subscriptions for command dispatch and reward collection.

## Project Structure

### 1. Main RL-Tools Project (Original)

This is the core project demonstrating how to train a reasoning-focused LLM (Qwen-4B Thinking with LoRA adapters) via Group Relative Preference Optimization (GRPO) to operate external tools inside containerized environments.

**Location**: Root directory and subdirectories

- `finetuning/` — Model fine-tuning notebooks.
- `inference/` — Model inference notebooks.
- `training/` — Training scripts.
- `serving/` — Model serving infrastructure, including the custom `ToolGRPOTrainer` implementation.
- `GeneratorFS/` — Generated model files.
- `qwen3-4b-thinking-openthoughts-lora/` — LoRA model checkpoints.

### 2. Web RL Environment (New Addition)

A containerized web-based RL environment API that integrates with Azure Service Bus for command processing (topics plus subscriptions).

**Location**: `web-rl-env/` directory

## Core Components

### LLM Policy

- Base model: `Qwen/Qwen3-4B-Thinking-2507`.
- Fine-tuned with LoRA for structured tool use (`<web>`, `<code>`, `<azure>` tags).
- Hosted on an A100 GPU for online training and evaluation.
- Training-time inference accelerated by **vLLM** for fast candidate sampling.
- Final serving deployed with **SGLang** for low-latency, multi-session inference and asynchronous tool handling.

### Tool Containers

Each tool is isolated into its own Azure Container Instance (ACI) for safety and modularity. Communication is asynchronous using a shared command topic with per-tool subscriptions, plus a reward topic that aggregates outcomes.

- **Web Container** — Headless Chromium (Playwright) session that opens documentation portals, navigates API references, and captures authoritative URLs that can be surfaced back to the policy.
- **Code Container** — Lightweight Python and Linux environment rooted at `/workspace` for file manipulation, shell commands, and running evaluation scripts required by the tasks.
- **Azure Container** — Hardened `az` CLI image with whitelisted subscription, VM, and resource group subcommands for cloud-administration workflows.

### Grading Infrastructure

- **Trajectory Grading (Grader-1)** — Uses the `openai/gpt-oss-20b` model to evaluate reasoning trajectories end-to-end. The grader runs on dual H100 GPUs to take advantage of the model's large KV cache, enabling reliable scoring for trajectories that reach beyond 130,000 tokens. The move to this model brought noticeably faster grading while preserving context fidelity for very long tool interactions. The team is actively porting the grading script to the `xprefillydecode` execution path to simplify future deployments.
- **Reward Signals** — Structural rewards encourage correct usage of `<think>`, `<web>`, `<code>`, and `<azure>` tags. Outcome rewards verify that requested actions (file edits, Azure commands, documentation retrieval) completed successfully. Trajectory rewards from Grader-1 are combined with these signals to produce a scalar reward for GRPO updates.

## Distributed Generator + Trainer (On-Policy)

AsyncRL’s primary training path is now the **distributed DisGenerator + DisTrainer loop** under `serving/`, designed for **on-policy GRPO**. The generator continuously reloads the **latest** policy checkpoint while the trainer consumes new trajectories and produces the next policy version. This keeps data fresh and aligned with the current policy rather than relying on a fixed, static model snapshot.

### How It Works

**DisGenerator (trajectory generation, vLLM disaggregated serving)**

- **Disaggregated prefill/decode vLLM** runs on dedicated GPUs to maximize throughput.
- The **AsyncBatchOrchestrator** streams tokens, **intercepts tool calls**, executes tools, and resumes generation.
- Each completion includes:
  - `old_logprobs` captured during streaming (for KL penalty).
  - `action_mask` to exclude tool outputs from the loss.
  - a scalar `reward` from structural + outcome + grader signals.
- Trajectories are written to `serving/DisTrainer/data/generations/batch_*.jsonl`.
- The generator **auto-selects the latest policy** via `get_latest_policy_path()` in `serving/DisGenerator/config.py`.

**DisTrainer (FSDP2 GRPO training)**

- Watches `data/generations/` for new batch files and trains automatically.
- Computes GRPO loss by grouping completions, forming advantages, and applying KL regularization.
- Saves new policy checkpoints as `policy-N-YYYYMMDD_HHMMSS` under `serving/DisTrainer/models/`.
- Deletes processed batches to avoid replay and keep disk clean.

**Why this is on-policy**

- **On-policy** means trajectories are generated by the **current** model being trained.
- DisGenerator reloads the latest checkpoint every time it starts a batch, so data reflects the most recent policy.
- In contrast, the legacy ToolGRPOTrainer path (see end of README) uses a **fixed policy snapshot** to generate batches, which is **off-policy** with respect to subsequent updates.

### Data Contract (JSONL)

Each trajectory is stored as a JSONL record:

```json
{
  "gen_id": "prompt1-0",
  "group_id": "prompt1",
  "prompt": "<|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...",
  "prompt_ids": [1234, 5678],
  "completion": {
    "text": "<think>...</think>\n<web>{...}</web>\n<tool_result>...</tool_result>\n<solution>...</solution>",
    "completion_ids": [9012, 3456],
    "old_logprobs": [-0.1, -0.2, 0.0],
    "action_mask": [1, 1, 0, 1],
    "reward": 2.35
  },
  "metadata": {
    "timestamp": 1735234567.123,
    "status": "COMPLETED",
    "num_turns": 3
  }
}
```

**Key fields**

- `action_mask`: `1` = model token (include in loss), `0` = tool output (exclude).
- `old_logprobs`: streaming log-probs used for KL penalty.
- `reward`: combined scalar reward for GRPO.

### Policy Versioning

Policies are saved as:

```
DisTrainer/models/
├── policy-0-initial
├── policy-1-20251226_131234
├── policy-2-20251226_143456
└── policy-N-{timestamp}
```

DisGenerator always loads the newest policy, so generation follows training updates.

### Run the Distributed System

> The minimum setup is **4 GPUs**: 2 for generation (1 prefill + 1 decode) and 2 for training.

#### Option 1: 4-GPU setup (minimum)

**Terminal 1 — Start DisGenerator (GPUs 0-1):**

```bash
cd serving/DisGenerator
export CUDA_VISIBLE_DEVICES=0,1
./scripts/start_all.sh 1p1d
```

**Terminal 2 — Run the Orchestrator:**

```bash
cd serving/DisGenerator
uv run python simple_client.py
```

**Terminal 3 — Start DisTrainer (GPUs 2-3):**

```bash
cd serving/DisTrainer
export CUDA_VISIBLE_DEVICES=2,3
export PYTHONPATH=..:$PYTHONPATH
torchrun --nproc_per_node=2 -m DisTrainer.train --config config/train_config.toml
```

#### Option 2: 8-GPU setup (recommended)

**Terminal 1 — Start DisGenerator (GPUs 0-3):**

```bash
cd serving/DisGenerator
export CUDA_VISIBLE_DEVICES=0,1,2,3
./scripts/start_all.sh 2p2d
```

**Terminal 2 — Run the Orchestrator:**

```bash
cd serving/DisGenerator
uv run python simple_client.py
```

**Terminal 3 — Start DisTrainer (GPUs 4-7):**

```bash
cd serving/DisTrainer
export CUDA_VISIBLE_DEVICES=4,5,6,7
export PYTHONPATH=..:$PYTHONPATH
torchrun --nproc_per_node=4 -m DisTrainer.train --config config/train_config.toml
```

### Monitoring

```bash
# Watch for new generation batches
watch -n 5 'ls -la serving/DisTrainer/data/generations/'

# Check trainer status
curl http://localhost:8000/status
```

## Running the Containers Locally

All containers expect an `.env` file that provides the Azure Service Bus connection information used for command and reward fan-out. Containers can be launched independently for debugging or together for end-to-end evaluation.

### Shared Prerequisites

```bash
# From repository root
cp serving/.env.example serving/.env  # Fill in Service Bus credentials
az servicebus topic create ...        # Ensure topics and subscriptions exist
```

### Web Container

```bash
cd serving/web-container
docker build -t asyncrl-web .
docker run --env-file ../.env -p 3000:3000 asyncrl-web
```

This exposes a FastAPI service that proxies navigation commands to Playwright and reports back rendered content and URLs.

### Code Container

```bash
cd serving/code-container
docker build -t asyncrl-code .
docker run --env-file ../.env -v $(pwd)/../../:/workspace asyncrl-code
```

This mounts the repository into `/workspace` so the agent can edit files and execute scripts inside the sandbox.

### Azure Container

```bash
cd serving/azure-container
docker build -t asyncrl-azure .
docker run --env-file ../.env asyncrl-azure
```

This container provides authenticated `az` CLI access with the minimal permission set required by the tasks.

When deployed in Azure Container Instances, the same images subscribe to the shared command topic and stream results back on the reward topic.

## Dataset Generation and Imitation Learning

Before launching GRPO, we bootstrap the policy via supervised fine-tuning (SFT) on synthetic tool-use demonstrations. Tool proficiency is the primary bottleneck for the base policy, so we curated trajectories that explicitly show:

- How to call each tool container with the correct XML tags.
- What configuration blocks and payload formats look like for command dispatch.
- How to stitch tool outputs back into structured `<solution>` responses.

### Synthetic Trajectories via OpenAI Batch API

Trajectories were generated using the OpenAI Batch API to process multiple prompts asynchronously. Roughly 40% of jobs were cut short because of quota limits. Among the completed jobs we filtered out demonstrations that were too short or had formatting glitches, keeping only long-form, clean dialogues suitable for imitation learning. The curated dataset now contains clear tool invocations with full request/response structure.

### Curriculum Design

To avoid overwhelming the small policy, we adopted a curriculum with three rungs:

- **Easy** — Single-tool problems (for example, basic file I/O or a single documentation lookup).
- **Easy-Medium** — Multi-step flows that combine two tools but with generous guidance.
- **Medium** — Realistic support-style tickets requiring sequencing across all three containers.

We intentionally skipped hard trajectories because the current model capacity is insufficient; they lead to exploration collapse. Medium-hard scenarios are being designed as the next stage once stability improves.

### Imitation Learning plus GRPO

1. **SFT Warm Start** — Fine-tune on the curated trajectories so the policy learns the syntax of tool tags, expected observation formats, and general operating procedures.
2. **GRPO Fine-Tuning** — Switch to reinforcement learning where the policy interacts with the live containers. GRPO encourages adherence to the response schema while optimizing for successful task completion.

This combination teaches basic tool competence via imitation and then refines performance with RL-driven rewards.

## Web RL Environment

The `web-rl-env/` directory contains a containerized web API for RL environment management backed by Azure Service Bus (Topics plus Subscriptions).

### Features

- `GET /health` endpoint for liveness checks.
- Command processing via a shared command topic.
- Reward publication via a reward topic.
- Docker Compose deployment for local testing.
- `.env` driven configuration.

### Quick Start

```bash
cd web-rl-env
docker-compose up --build
# API served at http://localhost:8000
```

### Configuration

Create a `.env` file with your Azure Service Bus connection string and topic settings:

```bash
AZURE_SERVICE_BUS_CONNECTION_STRING=your_connection_string_here
COMMAND_TOPIC_NAME=commandtopic
COMMAND_SUBSCRIPTION_NAME=rlcommandbustopic
REWARD_TOPIC_NAME=rewardtopic
REWARD_SUBSCRIPTION_NAME=rlcommandbustopic
```

## Deployment Notes

- Containers are pre-warmed for faster cold starts.
- Topic-based fan-out allows multiple executors to observe commands if desired.
- Subscriptions enable isolated replay and filtering without touching publishers.

## Serving Strategy

- Training: vLLM for parallel candidate generation.
- Deployment: SGLang for low-latency structured generation.

## ToolGRPOTrainer (Legacy Off-Policy Pipeline)

We still keep the custom `ToolGRPOTrainer` implementation for single-node experimentation and debugging, but it is **off-policy**
relative to the distributed system. In this flow, the generator samples batches from a fixed model snapshot, and training updates
the policy *after* those batches were produced. That means the data no longer reflects the most recent policy at the time of
optimization.

### Why the custom trainer still matters

The upstream TRL `GRPOTrainer` assumes single-turn responses while sampling completions, which breaks multi-turn tool calling.
`serving/ToolGRPOTrainer` replaces the decoding loop and reward wiring so trajectories can interleave tool calls with model tokens
without dropping context.

### ToolGRPOTrainer workflow (off-policy)

1. Sample *m* completions per prompt from a fixed policy checkpoint.
2. Emit tool commands onto the shared command topic; executors listen on their subscriptions.
3. Executors publish results to the reward topic; the trainer consumes them via its reward subscription.
4. Compute group-relative advantages (\(A_i = r_i - \bar{r}\)).
5. Update the policy with a KL penalty against the frozen reference model.

### ToolGRPOTrainer Run Summary

The latest multi-turn GRPO run used `serving/ToolGRPOTrainer/run_custom_grpo.py` to fine-tune the Qwen-3B policy with live tool
interactions enabled by the custom trainer overrides. The experiment bootstrapped the policy from the GRPO-pretrained checkpoint
in `grpo-qwen-training/checkpoint-100` while reusing the production LoRA adapter from `GeneratorFS/qwen3-4b-thinking-openthoughts-lora/checkpoint-2280`.

**Prompt curriculum.** Eighteen prompts were grouped into three difficulty bands covering single-tool warmups, two-tool workflows,
and end-to-end production scenarios. The curriculum reuses the same structured prompt wrapper employed by the standard GRPO
scripts so that completions remain compatible with the reward functions in `serving/ToolGRPOTrainer`.

**Reward shaping.** Training combined four reward sources:

- `tool_reward_fn`: encourages successful `<web>`, `<code>`, and `<azure>` calls.
- `char_reward_fn`: stabilizes completion length and discourages runaway tool loops.
- `format_reward_fn`: verifies the structural tags required by the graders.
- `grader_reward_fn`: streams numeric scores from the external grader container via Azure Service Bus.

**Trainer configuration.** The key hyperparameters for the run are listed below; the values align with the overrides in
`run_custom_grpo.py` and the recorded state in `serving/ToolGRPOTrainer/grpo-streamed/checkpoint-10/trainer_state.json`.

| Setting | Value |
| --- | --- |
| Output directory | `serving/ToolGRPOTrainer/grpo-streamed` |
| Max steps | 10 |
| Generations per prompt | 4 |
| Per-device train batch size | 4 |
| Learning rate | 5e-6 with linear decay |
| Max completion length | 20,000 tokens |
| Precision | bfloat16 with gradient checkpointing |
| Logging interval | Every step (1) |
| LoRA config | `r=8`, `alpha=16`, `dropout=0.1`, targets `q_proj`, `k_proj`, `v_proj`, `o_proj` |

**Checkpoint and telemetry.** The streamed outputs and model weights are saved under `serving/ToolGRPOTrainer/grpo-streamed`.
The latest adapter lives in `checkpoint-10/`, which also contains `trainer_state.json` with step-by-step reward traces. W&B
telemetry for the same run is available under the project `AsyncRL Trainer` with the run name `latest3:tool-use-grpo-trainer-run`.

## Future Work

- Additional tools (database, Git, REST API agent).
- Multi-objective reward fusion.
- Curriculum schedules.
- Richer reward shaping for deep web browsing.
