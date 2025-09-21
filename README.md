# 🚀 AsyncRL: Reinforcement Learning with Tools

This repository contains multiple RL projects focused on training language models to use external tools effectively.

## Pretty cool Feature 

By default, GRPOTrainer from huggingface does not enable interleaved tool calling while generating tokens for completitions in GRPO. Had to write custom functions to overwrite it (shoutout to codex for helping me out here - You Can find the implementation under ToolGRPOTrainer under serving folder). 

## 📁 Project Structure

### 1. **Main RL-Tools Project** (Original)
The core project demonstrating how to train a reasoning-focused LLM (Qwen-4B Thinking with LoRA adapters) via Group Relative Preference Optimization (GRPO) to use external tools inside containerized environments.

**Location**: Root directory and subdirectories
- `finetuning/` - Model fine-tuning notebooks
- `inference/` - Model inference notebooks  
- `training/` - Training scripts
- `serving/` - Model serving infrastructure
- `GeneratorFS/` - Generated model files
- `qwen3-4b-thinking-openthoughts-lora/` - LoRA model checkpoints

### 2. **Web RL Environment** (New Addition)
A containerized web-based RL environment API that integrates with Azure Service Bus for command processing (now via Topics + Subscriptions).

**Location**: `web-rl-env/` directory

---

## 🔑 Core Components (Main Project)

1. **LLM Policy**
   - Base model: `Qwen/Qwen3-4B-Thinking-2507`.
   - Fine-tuned with LoRA for structured tool use (`<web>`, `<code>`, `<azure>`).
   - Hosted on an A100 GPU.
   - Training-time inference accelerated by **vLLM** for fast candidate sampling.
   - Final serving deployed with **SGLang** for low-latency, multi-session inference with async tool handling.

2. **Tool Containers**
   Each tool is isolated into its own Azure Container Instance (ACI) for safety and modularity. Communication is asynchronous using Azure Service Bus **Topics** (one shared command topic with per-tool subscriptions + one reward topic). This replaced the previous storage queue design.

   - **Web Container** – Headless browser (Chromium/Playwright) for docs/APIs.
   - **Code Container** – Python + Linux utilities inside `/workspace`.
   - **Azure Container** – `az` CLI with whitelisted subcommands.

3. **Reward Mechanisms**
   - Structural rewards: correct usage of `<think>`, `<web>`, `<code>`, `<azure>` tags.
   - Outcome rewards:
     - File created/read correctly in Code container.
     - VM spun up or subscription queried successfully in Azure container.
     - Web container retrieved official docs (validated by URL).
   - Trajectory rewards: a lightweight reward model checks reasoning chains and tool sequencing.
   - Rewards aggregated to a scalar per candidate for GRPO.

---

## ⚙️ Training with GRPO

We use Hugging Face TRL's `GRPOTrainer`:

1. Sample *m* completions per input task (via vLLM backend).
2. Emit tool commands onto the command topic (publisher) tagged by tool type; executors read via their subscription.
3. Executors publish results to the reward topic; trainer consumes via its reward subscription.
4. Compute group-relative advantages (A_i = r_i - r_bar).
5. Update the policy with KL penalty versus frozen reference.

### 📝 Example Task

Find the official Microsoft doc that shows how to print the current Azure subscription name using the CLI. Create /workspace/hello.txt with hello world, read it back, then run the Azure command. Finally, report the file's contents, subscription name, and the doc URL.

**Execution sequence:**
1. `<web>` → locate official CLI doc.
2. `<code>` → create and read hello.txt.
3. `<azure>` → run az account show --query name.
4. Return `<solution>`.
5. Reward = structural + outcome + trajectory components.

---

## 🌐 Web RL Environment

A containerized web API for RL environment management with Azure Service Bus (Topics + Subscriptions).

### Features
- Health `/health`
- Command processing via a shared command topic
- Reward publication via a reward topic
- Docker Compose deployment
- `.env` driven configuration

### Quick Start

```bash
# Navigate to web RL environment
cd web-rl-env

# Build and run with Docker Compose
docker-compose up --build

# API at http://localhost:8000
```

### API Endpoints
- `GET /` - Root metadata
- `GET /health` - Health check
- `POST /receive-command` - Inject a command (publishes to command topic)
- `POST /send-reward` - Publish a reward message
- `GET /read-command` - Debug read (subscription pull)

### Configuration
Create a `.env` file with your Azure Service Bus connection string and topic settings:
```bash
AZURE_SERVICE_BUS_CONNECTION_STRING=your_connection_string_here
COMMAND_TOPIC_NAME=commandtopic
COMMAND_SUBSCRIPTION_NAME=rlcommandbustopic
REWARD_TOPIC_NAME=rewardtopic
REWARD_SUBSCRIPTION_NAME=rlcommandbustopic
```

---

## 📦 Deployment Notes

- Containers are pre-warmed.
- Topic-based fan-out allows multiple executors to observe commands if desired.
- Subscriptions enable isolated replay / filtering without touching publishers.

## ✅ Serving Strategy

- Training: vLLM for parallel candidate generation.
- Deployment: SGLang for low-latency structured generation.

## 🔮 Future Work

- Additional tools (DB, Git, REST API agent).
- Multi-objective reward fusion.
- Curriculum schedules.
- Richer reward shaping for web browsing depth.

---

## About

Multiple RL projects for tool-augmented LLMs, now unified on Azure Service Bus Topics + Subscriptions for asynchronous, decoupled execution.
