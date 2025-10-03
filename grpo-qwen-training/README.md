---
base_model: Qwen/Qwen3-4B-Thinking-2507
library_name: transformers
model_name: grpo-qwen-training
tags:
- generated_from_trainer
- trl
- grpo
licence: license
---

# Model Card for grpo-qwen-training

This model is a fine-tuned version of [Qwen/Qwen3-4B-Thinking-2507](https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507).
It has been trained using [TRL](https://github.com/huggingface/trl) with GRPO for tool-use and multi-step reasoning capabilities.

## Training Details

- **Latest Training Run**: October 3, 2025 - **COMPLETED** ✅
- **Training Steps**: 100/100 (completed successfully)
- **Duration**: 9 minutes 48 seconds
- **Method**: GRPO (Group Relative Policy Optimization)
- **Focus**: Tool-use agent training with realistic dummy trajectories
- **Tools**: Web search, code execution, Azure CLI integration

## Quick start

```python
from transformers import pipeline

question = "If you had a time machine, but could only go to the past or the future once and never return, which would you choose and why?"
generator = pipeline("text-generation", model="None", device="cuda")
output = generator([{"role": "user", "content": question}], max_new_tokens=128, return_full_text=False)[0]
print(output["generated_text"])
```

## Training procedure

[<img src="https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-28.svg" alt="Visualize in Weights & Biases" width="150" height="24"/>](https://wandb.ai/ppbhatt500-verizon/AsyncRL%20Trainer/runs/kp60wg3f) 


This model was trained with GRPO, a method introduced in [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://huggingface.co/papers/2402.03300).

### Training Configuration

The model was trained on multi-step tool-use tasks with:
- **System Prompt**: Strict tool call/result format enforcement
- **Reward Functions**: Tool usage, format compliance, and character length rewards  
- **Training Data**: 30 realistic scenarios from simple API creation to complex CI/CD pipelines
- **Memory Optimization**: Configured for A100 GPU with VLLM colocated setup
- **Base Model**: Pre-trained GRPO checkpoint-100 + LoRA adapter from checkpoint-240

### Framework versions

- TRL: 0.22.2
- Transformers: 4.55.2
- Pytorch: 2.7.1
- Datasets: 4.0.0
- Tokenizers: 0.21.4

## Citations

Cite GRPO as:

```bibtex
@article{shao2024deepseekmath,
    title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
    author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
    year         = 2024,
    eprint       = {arXiv:2402.03300},
}

```

Cite TRL as:
    
```bibtex
@misc{vonwerra2022trl,
	title        = {{TRL: Transformer Reinforcement Learning}},
	author       = {Leandro von Werra and Younes Belkada and Lewis Tunstall and Edward Beeching and Tristan Thrush and Nathan Lambert and Shengyi Huang and Kashif Rasul and Quentin Gallou{\'e}dec},
	year         = 2020,
	journal      = {GitHub repository},
	publisher    = {GitHub},
	howpublished = {\url{https://github.com/huggingface/trl}}
}
```