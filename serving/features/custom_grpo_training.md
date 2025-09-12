## Okay as you can see right now we have a nornal implementation of our GRPOTrainer 

However we have to update it so it takes custom generate function which parses token by token to look for the <web>, <code> and <azure> tags and once it comes across we call the functions defined. While we call it, we wait for the response from these tools and once the response is receievd we append it to the output we are producing and continue generation. 
import torch
from trl import GRPOTrainer
from parser import parse_and_validate_tools

# -------------------------------
# Dummy tool functions (simple stubs)
# -------------------------------

def run_web_tool(payload: dict) -> str:
    """Simulated web search."""
    query = payload.get("q", "")
    return f"[web-result] docs found for query: {query}"

def run_code_tool(payload: dict) -> str:
    """Simulated code execution."""
    cmd = payload.get("cmd", "")
    return f"[code-result] executed: {cmd}"

def run_azure_tool(payload: dict) -> str:
    """Simulated Azure CLI execution."""
    args = payload.get("args", [])
    return f"[azure-result] ran: {' '.join(args)}"

# -------------------------------
# Custom Trainer
# -------------------------------

class ToolCallingGRPOTrainer(GRPOTrainer):
    """
    Extension of GRPOTrainer with an interactive rollout loop.
    Pauses on <web>, <code>, <azure> calls and appends <tool_result>.
    """

    def _generate_completions(self, prompts, **gen_kwargs):
        completions = []

        for prompt in prompts:
            conversation = prompt
            full_trace = prompt
            done = False
            turns = 0

            while not done and turns < 5:  # safety cutoff
                # Run model forward
                inputs = self.tokenizer(conversation, return_tensors="pt").to(self.model.device)
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False,
                    temperature=0.2,
                )
                text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

                # Extract only the new part
                new_output = text[len(conversation):].strip()
                full_trace += new_output

                # Parse for tool calls
                tool_calls = parse_and_validate_tools(new_output)

                if tool_calls:
                    for tool in tool_calls:
                        if not tool["is_valid"]:
                            tool_result = "[error] invalid tool schema"
                        elif tool["type"] == "web":
                            tool_result = run_web_tool(tool["parsed_data"])
                        elif tool["type"] == "code":
                            tool_result = run_code_tool(tool["parsed_data"])
                        elif tool["type"] == "azure":
                            tool_result = run_azure_tool(tool["parsed_data"])
                        else:
                            tool_result = "[error] unknown tool"

                        # Feed tool result back
                        conversation += f"\n<tool_result>{tool_result}</tool_result>\n"

                elif "<solution>" in new_output:
                    completions.append(full_trace)
                    done = True
                else:
                    # No tools, no solution — let it continue
                    conversation = text

                turns += 1

            if not done:  # fallback if no solution
                completions.append(full_trace)

        return completions

## Usage 

from datasets import Dataset
from trl import GRPOConfig
from custom_grpo_trainer import ToolCallingGRPOTrainer

def reward_fn(completions, **kwargs):
    # very simple: reward if "<solution>" appears
    return [1.0 if "<solution>" in c else 0.0 for c in completions]

USER_TASKS = [
    "Find the official Azure CLI command to show current subscription name.",
    "Write hello world into a file and read it back."
]

dataset = Dataset.from_list([{"prompt": t} for t in USER_TASKS])

training_args = GRPOConfig(
    output_dir="./grpo-toy-run",
    max_steps=10,
    num_generations=1,
    per_device_train_batch_size=1,
    logging_steps=1,
    learning_rate=1e-5,
)

trainer = ToolCallingGRPOTrainer(
    model="Qwen/Qwen3-4B-Thinking-2507",
    train_dataset=dataset,
    args=training_args,
    reward_fn=reward_fn,
)

trainer.train()
