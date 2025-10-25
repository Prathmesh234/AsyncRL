import os
import shutil
from datasets import Dataset
from trl import GRPOConfig
from peft import LoraConfig
from custom_grpo import ToolCallingGRPOTrainer
from reward_fn.tool_reward import tool_reward_fn
from reward_fn.char_reward import char_reward_fn
from reward_fn.format_reward import format_reward_fn
from reward_fn.grader_reward import grader_reward_fn
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

print("[PRINT] run_custom_grpo.py importing & preparing dataset")

WANDB_DISABLED_VALUES = {"1", "true", "yes"}
DEFAULT_TOOL_WANDB_PROJECT = "ToolGRPOTrainer Telemetry"
DEFAULT_TOOL_WANDB_RUN_NAME = "multi-turn-tool-calling-grpo"

# Multiple reward functions will be combined automatically by the TRL library

# System prompt - use the same approach as run_grpo.py (relies on .env file)
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "You are a helpful AI assistant.")


USER_TASKS = [
    # Level 1: Single Tool Usage - Focused tasks with clear objectives
    "Research the current best practices for Python web application security, focusing on FastAPI and Flask frameworks. Look up the most common vulnerabilities like SQL injection, XSS, and authentication issues. Summarize the top 5 security recommendations and explain how to implement input validation and secure authentication in a Python web API.",
    
    "Create a Python script that processes a CSV file containing sales data with columns: date, product_name, quantity, price, customer_region. The script should calculate total revenue per region, find the best-selling product, and generate a summary report showing monthly trends. Include proper error handling for missing data and save the results to a JSON file.",
    
    "Use Azure CLI to set up a basic web application infrastructure. Create a resource group in East US, provision a Linux virtual machine (Standard_B1s size), configure a network security group allowing HTTP and SSH traffic, and set up a simple storage account for file uploads. Document each command and explain the purpose of each resource created.",
    
    "Research Docker containerization fundamentals for web applications. Compare the benefits of using containers versus traditional deployments, explain the difference between images and containers, and investigate best practices for writing Dockerfiles. Focus on security considerations, image size optimization, and multi-stage builds for production applications.",
    
    "Build a Python web scraping tool that extracts product information from an e-commerce website. Use libraries like requests and BeautifulSoup to gather product names, prices, and descriptions. Implement proper rate limiting, handle HTTP errors gracefully, store the data in a SQLite database, and create a simple data visualization showing price distributions using matplotlib.",
    
    "Research Azure pricing and service options for small to medium business deployments. Compare the costs and features of Azure App Service versus Virtual Machines for hosting web applications. Analyze storage options (Blob storage vs File storage) and calculate estimated monthly costs for a web application serving 10,000 users with moderate traffic patterns.",
    
    # Level 2: Multi-Tool Usage - Coordinated workflows with practical applications  
    "Research Redis caching strategies for web applications, focusing on cache-aside patterns and TTL (time-to-live) settings. Then implement a Python Flask application that uses Redis for caching database query results. Include functions to set, get, and invalidate cache entries, and demonstrate the performance improvement by comparing response times with and without caching enabled.",
    
    "Study JWT token authentication for REST APIs, including token structure, expiration handling, and security best practices. Then create a Node.js Express server that implements JWT-based login functionality with user registration, login endpoints, protected routes that require valid tokens, and proper error handling for expired or invalid tokens. Include middleware for token validation.",
    
    "Research PostgreSQL database optimization techniques, focusing on indexing strategies and query performance. Then write a Python application using psycopg2 or SQLAlchemy that creates a sample database with user and order tables, implements proper indexing, includes connection pooling for better performance, and provides functions for common database operations with transaction handling.",
    
    "Study React component optimization and state management patterns, including useState, useEffect hooks, and component re-rendering. Then build a React application with a product catalog that fetches data from an API, implements search and filter functionality, includes pagination for large datasets, and uses proper error boundaries to handle API failures gracefully.",
    
    "Research Azure Key Vault for secure secret management and learn about access policies and managed identities. Then use Azure CLI to create a Key Vault instance, store database connection strings and API keys as secrets, configure access policies for a web application, and demonstrate how to retrieve secrets programmatically using Azure SDK.",
    
    "Study Azure Functions for serverless computing, focusing on HTTP triggers and blob storage integration. Then create a serverless image processing function using Azure CLI that automatically resizes uploaded images, stores them in blob storage, and returns the processed image URL. Include proper error handling and monitoring configuration.",
    
    # Level 3: Multi-Tool Usage + Integrated Solutions - End-to-end implementations
    "Research application monitoring and health check best practices for web services, including uptime monitoring and performance metrics. Then create a Python Flask web service with health check endpoints that verify database connectivity and external API availability, write code to send custom metrics to a monitoring service, and use Azure CLI to set up Application Insights with basic alerting rules for service health monitoring.",
    
    "Study continuous integration and deployment patterns for web applications, focusing on automated testing and containerized deployments. Then research CI/CD best practices and security scanning, create a simple build script that runs tests and builds a Docker image, and use Azure CLI to configure a basic deployment pipeline that automatically deploys containerized applications to Azure Container Instances when code changes.",
    
    "Research database migration strategies and data backup procedures for production applications. Then investigate best practices for migrating application data safely, write a Python script that can export data from one database format and import it to another with data validation checks, and use Azure CLI to set up Azure Database for PostgreSQL with automated backup configuration and connection testing.",
    
    "Study serverless architecture patterns and event-driven processing for file uploads and processing workflows. Then research Azure Functions integration with storage services, implement a Python-based image processing function that automatically processes uploaded files (resize, compress, add watermarks), and deploy it using Azure CLI with blob storage triggers and basic monitoring to track processing success rates.",
    
    "Research API management and rate limiting techniques for public web services, focusing on request throttling and usage analytics. Then investigate Azure API Management features and security policies, create a Python Flask API with built-in rate limiting using Redis for request counting, and configure Azure API Management using CLI commands to implement additional security policies and developer access controls.",
    
    "Study centralized logging practices for distributed applications, including structured logging and log aggregation techniques. Then research log management best practices for microservices, write a Python application that generates structured JSON logs with correlation IDs and custom fields, and use Azure CLI to configure Azure Monitor Log Analytics for log collection with basic queries and dashboards for application monitoring."
]

# Wrap prompts similarly to run_grpo.py so formatting is consistent
dataset = Dataset.from_list([
    {"prompt": f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{t}<|im_end|>\n<|im_start|>assistant\n"} for t in USER_TASKS
])

# Weights & Biases configuration (match baseline GRPO setup with optional overrides)
tool_grpo_api_key = os.getenv("TOOL_GRPO_WANDB_API_KEY")
if tool_grpo_api_key:
    os.environ["WANDB_API_KEY"] = tool_grpo_api_key

tool_grpo_project = os.getenv("TOOL_GRPO_WANDB_PROJECT")
tool_grpo_run_name = os.getenv("TOOL_GRPO_WANDB_RUN_NAME")
tool_grpo_entity = os.getenv("TOOL_GRPO_WANDB_ENTITY")

wandb_disabled_flag = os.getenv("WANDB_DISABLED", "").strip().lower()
wandb_enabled = wandb_disabled_flag not in WANDB_DISABLED_VALUES

wandb_project = os.getenv("WANDB_PROJECT")
if not wandb_project:
    wandb_project = tool_grpo_project or DEFAULT_TOOL_WANDB_PROJECT
    os.environ["WANDB_PROJECT"] = wandb_project

wandb_run_name = os.getenv("WANDB_RUN_NAME")
if not wandb_run_name:
    wandb_run_name = tool_grpo_run_name or DEFAULT_TOOL_WANDB_RUN_NAME
    os.environ["WANDB_RUN_NAME"] = wandb_run_name

wandb_entity = os.getenv("WANDB_ENTITY")
if not wandb_entity and tool_grpo_entity:
    wandb_entity = tool_grpo_entity
    os.environ["WANDB_ENTITY"] = wandb_entity

if wandb_enabled:
    try:
        import wandb
    except ImportError:
        wandb = None
    if wandb is None:
        print("[PRINT][WARN] wandb package not installed; disabling logging.")
        wandb_enabled = False
    else:
        if not os.getenv("WANDB_API_KEY"):
            print("[PRINT][WARN] WANDB_API_KEY not set — logging may fail.")
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            entity=wandb_entity,
            sync_tensorboard=True,
            save_code=True,
        )
        print(
            f"[PRINT] ToolGRPOTrainer WANDB logging enabled -> project={wandb_project} run={wandb_run_name}"
        )
else:
    print("[PRINT] ToolGRPOTrainer WANDB logging disabled via WANDB_DISABLED or missing config")

report_to = "wandb" if wandb_enabled else "none"

print(f"[PRINT] Dataset size={len(dataset)} example_prompt=\n{dataset[0]['prompt'][:300]!r}")

training_args = GRPOConfig(
    output_dir="./grpo-streamed",
    max_steps=10,  # Increased from 3 to 10 for more training steps
    num_generations=4,  # Increased from 2 to 4 generations per prompt
    per_device_train_batch_size=4,  # Changed from 2 to 4 to be divisible by num_generations
    logging_steps=1,
    learning_rate=5e-6,
    gradient_checkpointing=True,
    bf16=True,
    max_completion_length=10000,
    report_to=report_to,
    log_completions=wandb_enabled,
    wandb_log_unique_prompts=wandb_enabled,
    num_completions_to_print=4 if wandb_enabled else 0,
    run_name=wandb_run_name if wandb_enabled else None,
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

base_model_path = "/home/ubuntu/UTAH-GRPO/AsyncRL/grpo-qwen-training/checkpoint-100"  # Latest GRPO checkpoint
adapter_path = "/home/ubuntu/UTAH-GRPO/AsyncRL/GeneratorFS/qwen3-4b-thinking-openthoughts-lora"  # LoRA adapter

print(f"[PRINT] Loading GRPO-trained base model: {base_model_path}")
model = AutoModelForCausalLM.from_pretrained(base_model_path)

print(f"[PRINT] Loading LoRA adapter: {adapter_path}")
model = PeftModel.from_pretrained(model, adapter_path)

trainer = ToolCallingGRPOTrainer(
    model=model,  # Pre-loaded model with GRPO + LoRA
    peft_config=peft_config,  # Additional LoRA for tool calling
    train_dataset=dataset,
    args=training_args,
    reward_funcs=[tool_reward_fn, char_reward_fn, format_reward_fn, grader_reward_fn],
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
