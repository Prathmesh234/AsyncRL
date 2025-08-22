# 🚀 Reinforcement Learning with Tools (RL-Tools Project)

This project demonstrates how to train a reasoning-focused LLM (Qwen-4B Thinking with LoRA adapters) via Group Relative Preference Optimization (GRPO) to use external tools inside containerized environments. The agent learns to accomplish real-world tasks by orchestrating multiple tool calls in sequence.

---

## 🔑 Core Components

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
4. Compute group-relative advantages \(A_i = r_i - \bar r\).
5. Update the policy model using REINFORCE with a KL penalty versus a frozen reference (the base Qwen model).

---

## 📡 System Flow

```mermaid
flowchart TD
    subgraph GPU[LLM Host (GPU Node)]
        P[Qwen 4B Thinking + LoRA\nPolicy Model]
        R[Reference Model (frozen)]
        VLLM[vLLM Engine\n(parallel generation)]
        GRPO[GRPO Trainer]
    end

    subgraph Queues[Azure Storage Queues]
        Wsend[taskqueue-web]
        Wrecv[rewardqueueweb]
        Csend[taskqueue-code]
        Crecv[rewardqueuecode]
        Asend[taskqueue-azure]
        Arecv[rewardqueueazure]
    end

    subgraph Containers[Tool Containers]
        WebC[Web Container\n(Headless Browser)]
        CodeC[Code Container\n(/workspace executor)]
        AzureC[Azure CLI Container]
    end

    P --|generations|--> VLLM
    VLLM --> GRPO
    GRPO --> P
    R -. |KL baseline| .- GRPO

    P --|tool call|--> Wsend
    P --|tool call|--> Csend
    P --|tool call|--> Asend

    Wsend --> WebC --> Wrecv
    Csend --> CodeC --> Crecv
    Asend --> AzureC --> Arecv

    Wrecv --> GRPO
    Crecv --> GRPO
    Arecv --> GRPO

    subgraph Serving[Final Deployment]
        SGL[SGLang Serving Engine\n(async tool sessions)]
    end

    P --> SGL
```

---

## 📝 Example Task

> Find the official Microsoft doc that shows how to print the current Azure subscription name using the CLI. Create `/workspace/hello.txt` with `hello world`, read it back, then run the Azure command. Finally, report the file’s contents, subscription name, and the doc URL.

Execution sequence:
1. `<web>` → locate official CLI doc.
2. `<code>` → create and read `hello.txt`.
3. `<azure>` → run `az account show --query name`.
4. Return `<solution>` with results.
5. Reward = structural correctness + verified outcomes.

---

## 📦 Deployment Notes

- Each container is pre-warmed to avoid long startup delays.
- Communication is fully decoupled (no mixing tools).
- Reward models can run on the same GPU as the policy (fastest) or on a separate GPU cluster if open-source reward models are used.

---

## ✅ Serving Strategy

- **Training phase:**
  - Use vLLM for candidate generation.
  - Supports parallel decoding (group size > 1) with efficient KV caching.
- **Deployment phase:**
  - Use SGLang to serve the RL-tuned model.
  - Features: async sessions, low-latency scheduling, native support for structured outputs.
  - Integrates directly with the tool queue architecture so the production agent can orchestrate tasks across Web, Code, and Azure containers.

---

## 🔮 Future Work

- Expand tool set (Databases, Git, APIs).
- Use multi-objective reward aggregation.
- Introduce curriculum training (start with Hello World → infra orchestration).

---

## 🚀 Quickstart

1. **Build tool containers** (example for the Web container):
   ```bash
   docker build -t rl-tools-web ./path/to/web-container
   docker push <registry>/rl-tools-web
   ```
2. **Launch training**:
   ```bash
   python training/generation.py
   ```
3. **Serve the model with SGLang**:
   ```bash
   cd serving
   python run_model.py --engine sglang
   ```

Feel free to adapt these commands to match your environment and container registry.

