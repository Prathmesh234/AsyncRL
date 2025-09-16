# 🚀 AsyncRL: Reinforcement Learning with Tools

This repository contains multiple RL projects focused on training language models to use external tools effectively.

## Pretty cool Feature 

By default, GRPOTrainer from huggingface does not enable interleaved tool calling while generating tokens for completitions in GRPO. Had to write custom functions to overwrite it (shoutout to codex for helping me out here - You can find the implementation under ToolGRPOTrainer under serving folder). 

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
A containerized web-based RL environment API that integrates with Azure Service Bus for command processing.

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
   Each tool is isolated into its own Azure Container Instance (ACI) for safety and modularity. Communication is asynchronous using Azure Storage Queues (one send queue + one receive queue per container).

   - **Web Container** – Runs a headless browser (Chromium/Playwright). Used for searching documentation, extracting snippets, and finding APIs.
   - **Code Container** – Runs Python + Linux utilities inside `/workspace`. Used for competitive coding tasks, script execution, project scaffolding, and test verification.
   - **Azure Container** – Runs `az` CLI with whitelisted subcommands. Used for infrastructure tasks like spinning up VMs, checking subscription info, or deploying resources.

3. **Reward Mechanisms**
   - Structural rewards: correct usage of `<think>`, `<web>`, `<code>`, `<azure>` tags.
   - Outcome rewards:
     - File created/read correctly in Code container.
     - VM spun up or subscription queried successfully in Azure container.
     - Web container retrieved official docs (validated by URL).
   - Trajectory rewards: a small closed-source reward model checks reasoning chains (`<think>` sections) and validates correctness of tool sequences.
   - Rewards are aggregated into a single scalar per candidate for GRPO.

---

## ⚙️ Training with GRPO

We use Hugging Face TRL's `GRPOTrainer`:

1. Sample *m* completions per input task (via vLLM backend).
2. Send candidate tool calls into their respective containers via queues.
3. Collect results and compute rewards.
4. Compute group-relative advantages \\(A_i = r_i - \bar r\\).
5. Update the policy model using REINFORCE with a KL penalty versus a frozen reference (the base Qwen model).

### 📝 Example Task

Find the official Microsoft doc that shows how to print the current Azure subscription name using the CLI. Create /workspace/hello.txt with hello world, read it back, then run the Azure command. Finally, report the file's contents, subscription name, and the doc URL.

**Execution sequence:**
1. `<web>` → locate official CLI doc.
2. `<code>` → create and read hello.txt.
3. `<azure>` → run az account show --query name.
4. Return `<solution>` with results.
5. Reward = structural correctness + verified outcomes.

---

## 🌐 Web RL Environment

A containerized web API for RL environment management with Azure Service Bus integration.

### Features
- **Health Check**: `/health` endpoint to check API status
- **Command Processing**: Receive and process commands via Azure Service Bus
- **Reward System**: Send rewards back through Service Bus queues
- **Docker Support**: Fully containerized with Docker Compose
- **Environment Variables**: Secure configuration management

### Quick Start

```bash
# Navigate to web RL environment
cd web-rl-env

# Build and run with Docker Compose
docker-compose up --build

# API available at http://localhost:8000
```

### API Endpoints
- `GET /` - Root endpoint with API information
- `GET /health` - Health check endpoint
- `POST /receive-command` - Receive a command from client
- `POST /send-reward` - Send a reward to Service Bus
- `GET /read-command` - Read commands from Service Bus queue

### Configuration
Create a `.env` file with your Azure Service Bus connection string:
```bash
AZURE_SERVICE_BUS_CONNECTION_STRING=your_connection_string_here
COMMAND_QUEUE_NAME=commandqueue
REWARD_QUEUE_NAME=rewardqueue
```

---

## 📦 Deployment Notes

- Each container is pre-warmed to avoid long startup delays.
- Communication is fully decoupled (no mixing tools).
- Reward models can run on the same GPU as the policy (fastest) or on a separate GPU cluster if open-source reward models are used.

## ✅ Serving Strategy

- **Training phase**: Use vLLM for candidate generation with parallel decoding (group size > 1) and efficient KV caching.
- **Deployment phase**: Use SGLang to serve the RL-tuned model with async sessions, low-latency scheduling, and native support for structured outputs.

## 🔮 Future Work

- Expand tool set (Databases, Git, APIs).
- Use multi-objective reward aggregation.
- Introduce curriculum training (start with Hello World → infra orchestration).
- Enhanced web RL environment with more sophisticated reward mechanisms.

---

## About

This repository contains multiple RL projects focused on training language models to use external tools effectively, with both the main GRPO-based tool use training and a containerized web RL environment for real-time command processing.
