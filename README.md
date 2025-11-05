# AsyncRL: Reinforcement Learning with Tools

AsyncRL contains multiple reinforcement learning projects that teach large language models how to use external tools safely and effectively. The repository combines supervised bootstrapping, reinforcement learning with live tooling feedback, and modular serving infrastructure built for asynchronous execution.

Deepwiki link for in depth documentation - https://deepwiki.com/Prathmesh234/AsyncRL/1-overview


Medium Blog - https://medium.com/@ppbhatt500/building-asyncrl-a-multi-tool-reinforcement-learning-pipeline-for-software-engineering-tasks-0fde815ed2b4

## Pretty cool Feature

By default, GRPOTrainer from huggingface does not enable multi turn tool calling while generating tokens for completitions in GRPO. Had to write custom functions to overwrite it (shoutout to codex for helping me out here - You Can find the implementation under ToolGRPOTrainer under serving folder).

## Table of Contents

1. [Key Capabilities](#key-capabilities)
2. [Project Structure](#project-structure)
3. [Core Components](#core-components)
   1. [LLM Policy](#llm-policy)
   2. [Tool Containers](#tool-containers)
   3. [Grading Infrastructure](#grading-infrastructure)
4. [Running the Containers Locally](#running-the-containers-locally)
5. [Training with GRPO](#training-with-grpo)
   1. [ToolGRPOTrainer Run Summary](#toolgrpotrainer-run-summary)
6. [Dataset Generation and Imitation Learning](#dataset-generation-and-imitation-learning)
7. [Web RL Environment](#web-rl-environment)
8. [Deployment Notes](#deployment-notes)
9. [Serving Strategy](#serving-strategy)
10. [Future Work](#future-work)

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

## Training with GRPO

We use Hugging Face TRL's `GRPOTrainer` with the following workflow:

> **Why a custom trainer?** The upstream `GRPOTrainer` assumes single-turn responses while sampling completions, so it short-circuits
> any follow-up tool calls that arrive mid-generation. Our `ToolGRPOTrainer` implementation inside `serving/` replaces the decoding
> loop and reward wiring so trajectories can interleave tool calls with model tokens without dropping context. Keeping this override
> in place preserves the multi-turn tool calling behavior highlighted above.

1. Sample *m* completions per input task via the vLLM backend.
2. Emit tool commands onto the shared command topic; executors listen on their subscriptions.
3. Executors publish results to the reward topic; the trainer consumes them via its reward subscription.
4. Compute group-relative advantages (\(A_i = r_i - \bar{r}\)).
5. Update the policy with a KL penalty against the frozen reference model.

### Example Task

> Find the official Microsoft doc that shows how to print the current Azure subscription name using the CLI. Create `/workspace/hello.txt` with "hello world", read it back, then run the Azure command. Finally, report the file's contents, subscription name, and the doc URL.

**Execution sequence:**

1. `<web>` → locate official CLI documentation.
2. `<code>` → create and read `hello.txt`.
3. `<azure>` → run `az account show --query name`.
4. Return `<solution>`.
5. Reward = structural + outcome + trajectory components.

### ToolGRPOTrainer Run Summary

The latest multi-turn GRPO run used `serving/ToolGRPOTrainer/run_custom_grpo.py` to fine-tune the Qwen-3B policy with live tool
interactions enabled by the custom trainer overrides. The experiment bootstrapped the policy from the GRPO-pretrained checkpoint
in `grpo-qwen-training/checkpoint-100` while reusing the production LoRA adapter from `GeneratorFS/qwen3-4b-thinking-openthoughts-lora/checkpoint-2280`.

**Prompt curriculum.** Eighteen prompts were grouped into three difficulty bands covering single-tool warmups, two-tool workflo
ws, and end-to-end production scenarios. The curriculum reuses the same structured prompt wrapper employed by the standard GRPO
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

## Future Work

- Additional tools (database, Git, REST API agent).
- Multi-objective reward fusion.
- Curriculum schedules.
- Richer reward shaping for deep web browsing.

