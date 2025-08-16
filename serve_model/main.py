import subprocess
import sys
import os

def main():
    print("Starting vLLM server for Qwen model...")
    
    # Path to your LoRA model
    lora_path = "/home/ubuntu/GeneratorFS/GeneratorFS/qwen3-4b-thinking-openthoughts-lora"
    
    # vLLM serve command
    cmd = [
        "vllm", "serve", "Qwen/Qwen3-4B-Thinking-2507",
        "--dtype", "auto",
        "--max-model-len", "128000",
        "--api-key", "token-abc123",
        "--enable-lora",
        "--lora-modules", f"qwen-lora={lora_path}"
    ]
    
    try:
        # Run the vLLM server
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running vLLM server: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nShutting down vLLM server...")
        sys.exit(0)

if __name__ == "__main__":
    main()