import os
import shutil
from datasets import Dataset
from trl import GRPOConfig
from peft import LoraConfig
from custom_grpo import ToolCallingGRPOTrainer
from reward_fn.tool_reward import tool_reward_fn

print("[PRINT] run_custom_grpo.py importing & preparing dataset")

# System prompt (match style of run_grpo.py) with explicit tool instructions

'''
WE HAVE OPTIMIZE THE PROMPT AND MAKE SURE IT USES THE PROMPT IN THE .env fle 


'''
SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "You are a helpful AI assistant with access to three tools.\n"
    "Tools: <web>{query}</web> for web search. <code>{code}</code> to run code (JSON with cmd,cwd,timeout_s). <azure>{azure}</azure> for Azure CLI (JSON with args list).\n"
    "Decide if a tool is needed BEFORE answering. If you use a tool, emit only the tool tag, wait for <tool_result> then continue reasoning. Finish final answer inside <solution>...</solution>."
)

USER_TASKS = [
    # Force explicit web tool call (restored explicit tag)
    "Find the latest stable Python version via the web tool then give final answer",
    # Force explicit code tool call
    "Write and execute Python code to print the sum of the first 5 integers: <code>{\"cmd\": \"python -c 'print(sum(range(1,6)))'\", \"cwd\": \".\", \"timeout_s\": 5}</code> then confirm the result.",
    # Force explicit azure tool call
    "Simulate an Azure CLI docs lookup using the azure tool: <azure>{\"args\": [\"az account show --query name -o tsv\"]}</azure> then summarize what it does.",
]

# Wrap prompts similarly to run_grpo.py so formatting is consistent
dataset = Dataset.from_list([
    {"prompt": f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{t}<|im_end|>\n<|im_start|>assistant\n"} for t in USER_TASKS
])

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
trainer = ToolCallingGRPOTrainer(
    model="Qwen/Qwen3-4B-Thinking-2507",  # Base model
    peft_config=peft_config,
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
