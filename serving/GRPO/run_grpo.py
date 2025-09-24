import os
import logging
import sys
# Add parent directory to path so sibling package 'reward_fn' is importable when running from GRPO folder
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)
from dotenv import load_dotenv
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig
from reward_fn.tool_reward import tool_reward_fn
from reward_fn.char_reward import char_reward_fn
from reward_fn.format_reward import format_reward_fn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from parent directory
load_dotenv("../.env")

# Get system prompt from environment
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "You are a helpful AI assistant.")


WANDB_DISABLED_VALUES = {"1", "true", "yes"}


def main():
    USER_TASKS = [
    # Level 1: Easy (Single Tool)
    "Search for FastAPI docs and create a simple POST endpoint for user registration.",
    "Look up PostgreSQL connection examples and write Python code with error handling.",
    "Find matplotlib tutorials and create a Python script for generating sales charts.",
    "Search for JWT authentication tutorials and implement token validation in JavaScript.",
    "Look up pytest documentation and write tests for a REST API.",
    "Search for Express.js security best practices and add Helmet middleware to an API.",
    "Find React hooks documentation and build a login form component with validation.",

    # Level 2: Medium (Web + Code / Web + Azure)
    "Search for Azure App Service deployment guides and deploy a simple web application.",
    "Find Azure CLI documentation and create a resource group in the West Europe region.",
    "Search for Redis tutorials, set up Azure Cache, and implement caching in Express.js API.",
    "Look up Docker best practices and create a Dockerfile for a Node.js Express app.",
    "Search for Azure Functions bindings docs and implement a queue-triggered function in Python.",
    "Find Azure Key Vault docs and write code to retrieve secrets in a Flask app.",
    "Search for TypeScript interfaces guide and create data models for e-commerce system.",
    "Look up database indexing strategies and optimize PostgreSQL queries for user table.",

    # Level 3: Advanced (Triple Tools / Multi-step)
    "Find MongoDB Atlas docs, create Azure VM, and write Python script for database migration.",
    "Look up SignalR Azure docs and build a real-time notification system in Node.js.",
    "Search for SendGrid API docs, configure Azure Function, and create email service endpoint.",
    "Research microservices patterns, create Azure container instances, and implement service discovery.",
    "Look up Azure CDN tutorials and deploy static assets for a React frontend.",
    "Search for Kubernetes docs, create Azure AKS cluster, and write deployment YAML files.",
    "Research monitoring best practices, set up Azure Monitor, and create custom dashboards with alerts.",
    "Find security best practices, configure Azure AD, and implement role-based access control.",

    # Level 4: Complex Scenarios (Real-life Mini Projects)
    "Look up CI/CD pipeline examples and write GitHub Actions workflow for automated testing.",
    "Find GitHub Actions deployment examples and set up CI/CD for Python Flask API.",
    "Search for backup solutions, create Azure recovery vault, and automate database backups.",
    "Look up caching strategies, configure Azure Redis, and implement distributed caching in API.",
    "Search for cron job examples and create a scheduled Azure Function for cleanup tasks.",
    "Find Bicep examples and write a Bicep template for Azure Storage + Function App.",
    "Look up Azure Cognitive Services usage and create a text sentiment API."
]
    
    dataset = Dataset.from_list([
        {
            "prompt": f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{task}<|im_end|>\n<|im_start|>assistant\n"
        }
        for task in USER_TASKS
    ])
    logger.info(f"Loaded {len(dataset)} prompts into Dataset")

    wandb_enabled = os.getenv("WANDB_DISABLED", "").strip().lower() not in WANDB_DISABLED_VALUES
    report_to = "wandb" if wandb_enabled else None

    if wandb_enabled and not os.getenv("WANDB_PROJECT"):
        default_project = "grpo-training"
        os.environ.setdefault("WANDB_PROJECT", default_project)
        logger.info("WANDB_PROJECT not set; defaulting to '%s'", default_project)

    if wandb_enabled:
        logger.info("Weights & Biases logging enabled; reporting to '%s'", os.getenv("WANDB_PROJECT"))
    else:
        logger.info("Weights & Biases logging disabled via WANDB_DISABLED env flag.")

    training_args = GRPOConfig(
        output_dir="/home/ubuntu/GeneratorFS/grpo-qwen-training",
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.3,
        max_steps=100,
        num_generations=2,                 # Generate 1 rollout for simple testing
        per_device_train_batch_size=2,
        logging_steps=5,
        learning_rate=5e-6,
        save_steps=50,
        report_to=report_to,
        log_completions=wandb_enabled,
        wandb_log_unique_prompts=wandb_enabled,
        num_completions_to_print=4 if wandb_enabled else 0,
        run_name=os.getenv("WANDB_RUN_NAME") if wandb_enabled else None,
    )

    logger.info("Initializing GRPO Trainer with multiple reward functions...")
    logger.info("Using reward functions: tool_reward, char_reward, format_reward")

    # For LoRA models, we need to specify the base model and LoRA config
    base_model = "Qwen/Qwen3-4B-Thinking-2507"  # Base model from adapter config
    
    # Configure LoRA settings from your existing adapter
    from peft import LoraConfig
    
    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        inference_mode=False,  # Set to False for training
        r=8,  # from your adapter_config.json
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["o_proj", "q_proj", "k_proj", "v_proj"],  # from your config
        bias="none"
    )
    
    trainer = GRPOTrainer(
        model=base_model,             # Use base model
        peft_config=peft_config,      # Apply LoRA configuration
        train_dataset=dataset,
        args=training_args,
        reward_funcs=[tool_reward_fn, char_reward_fn, format_reward_fn]  # Multiple reward functions
    )

    logger.info("Starting GRPO training...")
    trainer.train()
    logger.info("Training completed!")


if __name__ == "__main__":
    main()
