import os
import logging
from dotenv import load_dotenv
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig
from reward_fn.tool_reward import tool_reward_fn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Get system prompt from environment
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "You are a helpful AI assistant.")


def main():
    USER_TASKS = [
        # Mixed web + code tasks
        "Find the latest FastAPI documentation and create a simple POST endpoint for user registration.",
        "Search for PostgreSQL connection examples and write Python code to connect with error handling.",
        "Look up Docker best practices and create a Dockerfile for a Node.js Express app.",
        "Find React hooks documentation and build a login form component with validation.",
        "Search for JWT authentication tutorials and implement token validation in JavaScript.",
        
        # Mixed azure + code tasks  
        "Create an Azure storage account and write Python code to upload files to blob storage.",
        "Set up Azure Function App and create a simple HTTP trigger function in Python.",
        "Create Azure SQL database and write a connection script with proper error handling.",
        "Set up Azure Key Vault and write code to retrieve secrets in a Flask application.",
        "Create Azure Service Bus queue and implement a message sender in Node.js.",
        
        # Mixed web + azure tasks
        "Find Azure CLI documentation and create a resource group in the West Europe region.",
        "Search for Azure App Service deployment guides and deploy a simple web application.",
        "Look up Azure Cosmos DB tutorials and create a database with a users container.",
        "Find Azure Container Registry docs and push a custom Docker image to ACR.",
        "Search for Azure Logic Apps examples and create a workflow for email notifications.",
        
        # Triple tool tasks (web + azure + code)
        "Find MongoDB Atlas documentation, create Azure VM, and write Python script for database migration.",
        "Search for Redis tutorials, set up Azure Cache, and implement caching in Express.js API.",
        "Look up AWS S3 integration guides, create Azure storage, and build file upload service.",
        "Find Kubernetes documentation, create Azure AKS cluster, and write deployment YAML files.",
        "Search for SendGrid API docs, configure Azure Function, and create email service endpoint.",
        
        # Code-focused with web research
        "Research Socket.io documentation and build a real-time chat application with rooms.",
        "Find matplotlib tutorials and create a Python script for generating sales charts.",
        "Look up pytest documentation and write comprehensive tests for a REST API.",
        "Search for TypeScript interfaces guide and create data models for e-commerce system.",
        "Find Express.js middleware examples and build authentication middleware with rate limiting.",
        
        # Azure infrastructure with code implementation
        "Create Azure Application Insights and implement custom telemetry tracking in Python app.",
        "Set up Azure Blob storage container and build a file management API with Node.js.",
        "Create Azure Virtual Network with subnets and write Terraform configuration scripts.",
        "Set up Azure Load Balancer and create health check endpoints for multiple services.",
        "Create Azure API Management instance and configure API policies with custom headers.",
        
        # Web research with practical implementation
        "Research GraphQL best practices and implement a simple schema with resolvers.",
        "Find Nginx configuration guides and create reverse proxy setup for microservices.",
        "Look up database indexing strategies and optimize PostgreSQL queries for user table.",
        "Search for OAuth 2.0 implementation and create secure API authentication flow.",
        "Find CI/CD pipeline examples and write GitHub Actions workflow for automated testing.",
        
        # Complex mixed scenarios
        "Research microservices patterns, create Azure container instances, and implement service discovery.",
        "Find monitoring best practices, set up Azure Monitor, and create custom dashboards with alerts.",
        "Look up caching strategies, configure Azure Redis, and implement distributed caching in API.",
        "Search for backup solutions, create Azure recovery vault, and automate database backups.",
        "Find security best practices, configure Azure AD, and implement role-based access control."
    ]
    
    dataset = Dataset.from_list([
        {
            "prompt": f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{task}<|im_end|>\n<|im_start|>assistant\n"
        }
        for task in USER_TASKS
    ])
    logger.info(f"Loaded {len(dataset)} prompts into Dataset")

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
        save_steps=50
    )

    logger.info("Initializing GRPO Trainer...")

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
        reward_funcs=tool_reward_fn
    )

    logger.info("Starting GRPO training...")
    trainer.train()
    logger.info("Training completed!")


if __name__ == "__main__":
    main()
