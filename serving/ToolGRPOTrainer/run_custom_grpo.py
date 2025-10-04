import os
import shutil
from datasets import Dataset
from trl import GRPOConfig
from peft import LoraConfig
from custom_grpo import ToolCallingGRPOTrainer
from reward_fn.tool_reward import tool_reward_fn

print("[PRINT] run_custom_grpo.py importing & preparing dataset")

WANDB_DISABLED_VALUES = {"1", "true", "yes"}

# System prompt (match style of run_grpo.py) with explicit tool instructions

'''
WE HAVE OPTIMIZE THE PROMPT AND MAKE SURE IT USES THE PROMPT IN THE .env fle 


'''
SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    (
        "You are an ORCHESTRATOR/CODING AGENT. Complete real tasks by calling tools and returning a concise final solution.\n"
        "ALLOWED TAGS ONLY\n"
        '- <think>...</think> — plan the next step when needed.\n'
        '- <web>{\"type\": \"web\", \"q\": \"search terms\", \"k\": INTEGER}</web> — q must be non-empty; k must be an integer between 1 and 10.\n'
        '- <code>{\"type\": \"code\", \"code_command\": \"shell command\"}</code> — command must be a non-empty string.\n'
        '- <azure>{\"type\": \"azure\", \"azure_command\": \"az subcommand\"}</azure> — command must be a non-empty string.\n'
        '- <solution>...</solution> — final answer for the user.\n'
        "TOOL PAYLOAD RULES\n"
        "- Wrap only the JSON object inside the tool tag and emit nothing else on that turn.\n"
        "- The JSON must match exactly one of the schemas above — no extra fields, comments, or metadata (never add request_id).\n"
        "- Emit exactly one tool tag per turn and wait for <tool_result> before continuing.\n"
        "- Finish in <solution>...</solution> once requirements are satisfied."
    )
)


USER_TASKS = [
    # Force explicit web tool call (restored explicit tag)
    "Find the latest stable Python version via the web tool then search for any vulerabilities associated with it. Summarize the findings.",
    # Force explicit code tool call
    "Write and execute Python code to print the sum of the first 5 integers: <code>{\"type\": \"code\", \"code_command\": \"python -c 'print(sum(range(1,6)))'\"}</code> then confirm the result.",
    # Force explicit azure tool call
    "Simulate an Azure CLI docs lookup using the azure tool: <azure>{\"type\": \"azure\", \"azure_command\": \"az account show --query name -o tsv\"}</azure> then summarize what it does.",
]

# Wrap prompts similarly to run_grpo.py so formatting is consistent
dataset = Dataset.from_list([
    {"prompt": f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{t}<|im_end|>\n<|im_start|>assistant\n"} for t in USER_TASKS
])

# Use ToolGRPOTrainer-specific wandb configuration
tool_grpo_api_key = os.getenv("TOOL_GRPO_WANDB_API_KEY")
tool_grpo_project = os.getenv("TOOL_GRPO_WANDB_PROJECT", "ToolGRPOTrainer Telemetry")
tool_grpo_run_name = os.getenv("TOOL_GRPO_WANDB_RUN_NAME", "multi-turn-tool-calling-grpo")

wandb_enabled = os.getenv("WANDB_DISABLED", "").strip().lower() not in WANDB_DISABLED_VALUES and tool_grpo_api_key is not None
report_to = "wandb" if wandb_enabled else None

# Set ToolGRPOTrainer-specific wandb environment variables for the training
if wandb_enabled:
    os.environ["WANDB_API_KEY"] = tool_grpo_api_key
    os.environ["WANDB_PROJECT"] = tool_grpo_project
    print(f"[PRINT] ToolGRPOTrainer WANDB logging enabled -> project={tool_grpo_project}")
else:
    print("[PRINT] ToolGRPOTrainer WANDB logging disabled (no API key or WANDB_DISABLED flag set)")

print(f"[PRINT] Dataset size={len(dataset)} example_prompt=\n{dataset[0]['prompt'][:300]!r}")

training_args = GRPOConfig(
    output_dir="./grpo-streamed",
    max_steps=3,
    num_generations=2,
    per_device_train_batch_size=2,
    logging_steps=1,
    learning_rate=5e-6,
    gradient_checkpointing=True,
    bf16=True,
    report_to=report_to,
    log_completions=wandb_enabled,
    wandb_log_unique_prompts=wandb_enabled,
    num_completions_to_print=4 if wandb_enabled else 0,
    run_name=tool_grpo_run_name if wandb_enabled else None,
)
print("[PRINT] Training args ready")

# LoRA config (match run_grpo.py values)
peft_config = LoraConfig(
    task_type="CAUSAL_LM",
    inference_mode=False,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["o_proj", "q_proj", "k_proj", "v_proj"],
    bias="none"
)
print("[PRINT] LoRA config ready")

print("[PRINT] Instantiating trainer (this will load model)...")

# Use the same setup as GRPO/run_grpo.py
# Load GRPO-trained base model + LoRA adapter (similar to run_grpo.py pattern)
from transformers import AutoModelForCausalLM
from peft import PeftModel

base_model_path = "/home/ubuntu/GeneratorFS/grpo-qwen-training/checkpoint-100"  # Latest GRPO checkpoint
adapter_path = "/home/ubuntu/GeneratorFS/training/training/training_script/qwen3-4b-thinking-openthoughts-lora/checkpoint-240"

print(f"[PRINT] Loading GRPO-trained base model: {base_model_path}")
model = AutoModelForCausalLM.from_pretrained(base_model_path)

print(f"[PRINT] Loading LoRA adapter: {adapter_path}")
model = PeftModel.from_pretrained(model, adapter_path)

trainer = ToolCallingGRPOTrainer(
    model=model,  # Pre-loaded model with GRPO + LoRA
    peft_config=peft_config,  # Additional LoRA for tool calling
    train_dataset=dataset,
    args=training_args,
    reward_funcs=tool_reward_fn,
)
print("[PRINT] Trainer instantiated")

# Introspect the bound method to confirm override
import types
print("[PRINT] trainer.generate_completions ->", trainer.generate_completions.__qualname__)
print("[PRINT] trainer class ->", trainer.__class__)
print("[PRINT] MRO ->", trainer.__class__.mro())

# Sanity test: call generate_completions manually on a single prompt BEFORE training
print("[PRINT] Calling trainer.generate_completions manually (sanity test)...")
_test_prompt = [dataset[0]["prompt"]]
try:
    _manual_comps = trainer.generate_completions(_test_prompt, max_new_tokens=8)
    print("\n[PRINT] Manual completions returned (len=", len(_manual_comps), ")")
    for i, c in enumerate(_manual_comps):
        print(f"[PRINT] Manual completion {i} (first 200 chars): {repr(c[:200])}")
except Exception as e:
    print("[PRINT][ERROR] Manual generate_completions failed:", e)

# If we STILL did not see our custom prints, force monkey patch
if not isinstance(trainer.generate_completions, types.MethodType) or "ToolCallingGRPOTrainer" not in trainer.generate_completions.__qualname__:
    print("[PRINT][WARN] Custom generate_completions not bound. Monkey patching...")
    def _patched(prompts, **kw):
        print("[PRINT] PATCHED generate_completions executing")
        return trainer._generate_completions(prompts, **kw)
    trainer.generate_completions = types.MethodType(_patched, trainer)
    print("[PRINT] Patch applied. New qualname:", trainer.generate_completions.__qualname__)

print("[PRINT] Starting GRPO training (SIMPLE PRINT DEBUG RUN)...")
trainer.train()

print("[PRINT] GRPO Training complete - Deleting YOLO Model run")
# Clean up the output directory
if os.path.exists(training_args.output_dir):
    shutil.rmtree(training_args.output_dir)
    print(f"[PRINT] Deleted training output directory: {training_args.output_dir}")
else:
    print(f"[PRINT] Output directory not found: {training_args.output_dir}")

print("[PRINT] Cleanup complete!")
