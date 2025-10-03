import os
import logging
import sys
from dotenv import load_dotenv
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM

# Set CUDA memory optimization
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ---------------------------
# Setup logging
# ---------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------
# Paths & imports
# ---------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

# Import reward functions after path setup
from reward_fn.tool_reward import tool_reward_fn
from reward_fn.char_reward import char_reward_fn
from reward_fn.format_reward import format_reward_fn

# Load environment variables
load_dotenv(os.path.join(PARENT_DIR, ".env"))

# ---------------------------
# Env vars
# ---------------------------
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "You are a helpful AI assistant.")
WANDB_DISABLED_VALUES = {"1", "true", "yes"}
wandb_enabled = os.getenv("WANDB_DISABLED", "").strip().lower() not in WANDB_DISABLED_VALUES

# ---------------------------
# Weights & Biases setup
# ---------------------------
if wandb_enabled:
    import wandb
    if not os.getenv("WANDB_API_KEY"):
        logger.warning("WANDB_API_KEY not set — logging may fail.")

    project = os.getenv("WANDB_PROJECT", "grpo-training")
    run_name = os.getenv("WANDB_RUN_NAME", None)
    entity = os.getenv("WANDB_ENTITY", None)  # optional: your W&B username/org

    wandb.init(
        project=project,
        name=run_name,
        entity=entity,
        sync_tensorboard=True,
        save_code=True
    )
    logger.info(f"Weights & Biases logging enabled → project='{project}', run='{wandb.run.name}'")
else:
    logger.info("Weights & Biases logging disabled via WANDB_DISABLED env flag.")


def main():
    # ---------------------------
    # Dataset
    # ---------------------------
    USER_TASKS = [
        # Level 1: Easy (Single Tool)
    "Your team is prototyping a small signup feature for a hackathon project. A backend service is needed to accept user details. Search FastAPI documentation and design a POST endpoint that safely stores a new user's data.",
    "A student app you built keeps crashing whenever it loses its connection to the database. Investigate PostgreSQL connection examples and write Python code that includes error handling and retries to make it more resilient.",
    "The sales department wants to understand trends from last quarter. They’ve handed you CSV data but need visual insights. Look up matplotlib tutorials and create a Python script that produces clear bar and line charts for sales.",
    "Your authentication flow is incomplete. The product owner requires tokens to be verified before granting access to a dashboard. Search JWT authentication tutorials and implement validation logic in a small JavaScript app.",
    "A REST API you delivered has no tests, and QA flagged it as a risk. Search pytest documentation and create basic unit tests to verify endpoints return expected responses.",
    "An external audit flagged missing HTTP headers in your Node.js API. Look up Express.js security best practices and add middleware that strengthens protection against common attacks.",
    "A React-based login form allows empty submissions and confuses users. Find React hooks documentation and create a component that validates input before allowing a login attempt.",

    # Level 2: Medium (Web + Code / Web + Azure)
    "A client wants to show a demo of their site running live in the cloud. Search Azure App Service deployment guides and deploy a small sample web application so they can preview it.",
    "The operations team asked for a new sandbox environment. Search Azure CLI documentation and create a resource group in a specific region to prepare for other resources.",
    "Users are complaining that their search results load slowly after multiple requests. Research Redis tutorials, configure Azure Cache for Redis, and add caching to an Express.js backend to speed things up.",
    "Your manager wants the current API containerized so it can run in staging. Look up Docker best practices and write a Dockerfile that works well for a Node.js Express app.",
    "Background job processing is required to handle image uploads asynchronously. Search Azure Functions bindings documentation and implement a Python queue-triggered function to process uploaded files.",
    "Secrets are currently hardcoded in your Flask app, which is insecure. Find Azure Key Vault documentation and update the code so secrets are pulled dynamically at runtime.",
    "An e-commerce project needs strong type safety for data models. Look up TypeScript interfaces and design interfaces for products, orders, and user accounts to ensure consistency across the system.",
    "Customer complaints suggest the user table is slowing queries in production. Research indexing strategies and improve PostgreSQL query performance with indexing on key fields.",

    # Level 3: Advanced (Triple Tools / Multi-step)
    "The company is moving away from a local database. Look up MongoDB Atlas documentation, provision an Azure VM, and write a Python script to migrate existing data to the new hosted database.",
    "A new chat feature is needed for the learning platform. Search Azure SignalR documentation and implement a Node.js service that supports real-time notifications between users.",
    "Marketing requested automated email confirmations for new signups. Find SendGrid API documentation, configure an Azure Function, and implement a service endpoint that sends emails.",
    "Leadership is asking for a microservices demo to evaluate scalability. Research microservices architecture, deploy multiple services on Azure Container Instances, and configure basic service discovery.",
    "Static images and JS bundles are loading too slowly for international users. Look up Azure CDN tutorials and configure a CDN to serve React frontend assets globally.",
    "The team is standardizing deployments with Kubernetes. Search Azure AKS documentation, provision a cluster, and prepare YAML manifests to deploy your app’s components.",
    "Operations wants visibility into system health before launch. Research monitoring practices, enable Azure Monitor, and create dashboards with alerts for API error rates.",
    "A new customer requires stricter compliance. Look up Azure AD integration guides and configure role-based access so that only admins can access sensitive endpoints.",

    # Level 4: Complex Scenarios (Real-life Mini Projects)
    "Developers are wasting time manually testing every pull request. Look up CI/CD pipeline examples and create a GitHub Actions workflow that runs automated test suites on every PR.",
    "Your team’s Flask API is deployed manually and often fails. Search GitHub Actions deployment workflows and set up a CI/CD pipeline that automatically deploys updates to Azure after tests pass.",
    "Management requires a recovery plan for critical databases. Search for backup and disaster recovery solutions, configure an Azure Recovery Vault, and automate nightly backups of database resources.",
    "Users notice inconsistent API performance at scale. Research distributed caching strategies, configure Azure Redis, and integrate it into the backend to handle heavy traffic.",
    "Temporary files are filling up storage on a service. Search cron job examples and create a scheduled Azure Function that cleans up old files at regular intervals.",
    "The organization is adopting Infrastructure as Code. Look up Bicep examples and write a Bicep template that provisions Azure Storage and a Function App in one deployment.",
    "Product managers want to measure customer sentiment in real time. Find Azure Cognitive Services documentation and build an API endpoint that takes text input and returns a sentiment score."
]


    dataset = Dataset.from_list([
        {
            "prompt": f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
                      f"<|im_start|>user\n{task}<|im_end|>\n<|im_start|>assistant\n"
        }
        for task in USER_TASKS
    ])
    logger.info(f"Loaded {len(dataset)} prompts into Dataset")

    # ---------------------------
    # Training config
    # ---------------------------
    report_to = "wandb" if wandb_enabled else "none"

    training_args = GRPOConfig(
        output_dir="/home/ubuntu/GeneratorFS/grpo-qwen-post-traj-sft",
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.3,  
        max_steps=100,
        num_generations=2,
        per_device_train_batch_size=2,    
        logging_steps=5,
        learning_rate=5e-6,
        save_steps=50,
        max_prompt_length=1536,           # Reduced from 2048 to 1536
        max_completion_length=2500,       # Reduced from 6000 to 2500
        report_to=report_to,
        log_completions=wandb_enabled,
        wandb_log_unique_prompts=wandb_enabled,
        num_completions_to_print=4 if wandb_enabled else 0,
        run_name=os.getenv("WANDB_RUN_NAME") if wandb_enabled else None,
    )

    logger.info("Initializing GRPO Trainer with existing trained models...")

    # ---------------------------
    # Load base model + trained adapter
    # ---------------------------
    # Use existing GRPO-trained model as base
    base_model_path = "/home/ubuntu/GeneratorFS/grpo-qwen-training/checkpoint-100"  # Latest GRPO checkpoint
    # Use checkpoint-240 adapter from training directory
    adapter_path = "/home/ubuntu/GeneratorFS/training/training/training_script/qwen3-4b-thinking-openthoughts-lora/checkpoint-240"

    # Load GRPO-trained base model + LoRA adapter
    logger.info(f"Loading GRPO-trained base model: {base_model_path}")
    model = AutoModelForCausalLM.from_pretrained(base_model_path)
    
    logger.info(f"Loading LoRA adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)

    trainer = GRPOTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        reward_funcs=[tool_reward_fn, char_reward_fn, format_reward_fn]
    )

    logger.info("Starting GRPO training...")
    trainer.train()
    logger.info("Training completed!")


if __name__ == "__main__":
    main()
