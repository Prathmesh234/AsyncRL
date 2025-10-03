import subprocess
import sys
import os

def main():
    print("Starting vLLM server with GRPO post-trajectory SFT adapter...")
    
    # Use original base model (GRPO checkpoint only contains adapter weights)
    base_model_path = "Qwen/Qwen3-4B-Thinking-2507"
    # Path to GRPO-trained adapter (already includes thinking LoRA behavior)
    grpo_adapter_path = "/home/ubuntu/GeneratorFS/grpo-qwen-post-traj-sft/checkpoint-100"
    
    # vLLM serve command with GRPO adapter only
    # Note: GRPO adapter was trained on base+thinking_lora, so it already incorporates both behaviors
    cmd = [
        "vllm", "serve", base_model_path,
        "--dtype", "auto",
        "--max-model-len", "128000",
        "--api-key", "token-abc123",
        "--enable-lora",
        "--lora-modules", f"grpo-adapter={grpo_adapter_path}"
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