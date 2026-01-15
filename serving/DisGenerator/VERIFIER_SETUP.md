# Prime Intellect Verifier Integration Guide

This guide shows how to integrate **Prime Intellect's verifiers library** for mathematical solution verification in DisGenerator.

## Overview

- **Library**: Prime Intellect's `verifiers` (official library)
- **Verifier**: `MathRubric` for symbolic math verification
- **Dataset**: GSM8K (Grade School Math 8K)
- **API Keys**: **NOT REQUIRED** for local verification

---

## Installation

### Step 1: Install Prime Intellect Verifiers

```bash
cd /home/ubuntu/rl-storage/AsyncRL/serving/DisGenerator

# Install verifiers library
uv add verifiers

# Or with pip
pip install verifiers
```

### Step 2: Install Dataset Library

```bash
# For downloading GSM8K dataset
uv add datasets

# Or with pip
pip install datasets
```

---

## Download GSM8K Dataset

```bash
# Download 100 samples from GSM8K (for testing)
uv run python download_gsm8k.py --num-samples 100 --output prompts_gsm8k.jsonl

# Download full training set
uv run python download_gsm8k.py --num-samples -1 --split train --output prompts_gsm8k_full.jsonl

# Download test set
uv run python download_gsm8k.py --num-samples -1 --split test --output prompts_gsm8k_test.jsonl
```

This creates a JSONL file with format:
```json
{
  "prompt_id": "gsm8k_train_0",
  "messages": [{"role": "user", "content": "Natalia sold clips..."}],
  "expected_answer": "48",
  "metadata": {"dataset": "gsm8k", "split": "train"}
}
```

---

## How Prime Intellect's Verifiers Work

### MathRubric

The `MathRubric` from Prime Intellect provides:

1. **Symbolic Verification**: Checks mathematical equivalence (e.g., "1/2" == "0.5")
2. **LaTeX Parsing**: Extracts answers from `\boxed{}` format
3. **Local Execution**: No API keys or external services needed

### Example Usage

```python
import verifiers as vf

# Create MathRubric
rubric = vf.MathRubric()

# Verify a solution
completion = [{"role": "assistant", "content": "The answer is \\boxed{48}"}]
expected_answer = "48"

reward = rubric.correct_answer(completion, expected_answer)
# Returns: 1.0 (correct) or 0.0 (incorrect)
```

---

## Integration with DisGenerator

The `solution_verifier.py` module provides:

### Main Function

```python
from solution_verifier import compute_verification_reward

# Extract <solution> and verify
reward = compute_verification_reward(
    completion_text="<solution>The answer is \\boxed{48}</solution>",
    expected_answer="48"
)
# Returns: 1.0 if correct, 0.0 if incorrect
```

### Fallback Chain

1. **Prime Intellect MathRubric** (preferred)
2. **math-verify library** (if verifiers not installed)
3. **Simple string matching** (if neither installed)

---

## Testing the Verifier

### Quick Test

```bash
cd /home/ubuntu/rl-storage/AsyncRL/serving/DisGenerator

# Test the verifier
uv run python -c "
from solution_verifier import compute_verification_reward

# Test 1: Correct answer
reward1 = compute_verification_reward(
    '<solution>The answer is \\\\boxed{48}</solution>',
    '48'
)
print(f'Test 1 (correct): {reward1}')  # Should be 1.0

# Test 2: Incorrect answer
reward2 = compute_verification_reward(
    '<solution>The answer is \\\\boxed{50}</solution>',
    '48'
)
print(f'Test 2 (incorrect): {reward2}')  # Should be 0.0
"
```

---

## Full Workflow

### 1. Download Dataset

```bash
uv run python download_gsm8k.py --num-samples 10 --output prompts_gsm8k.jsonl
cp prompts_gsm8k.jsonl prompts.jsonl
```

### 2. Start DisGenerator Servers

```bash
./scripts/start_all.sh 1p1d --use-base-model
```

### 3. Run Client (WITHOUT tools, WITH verification)

```bash
# Set model name and disable tools
LORA_ADAPTER_NAME=openthoughts-adapter uv run python simple_client.py --tool ""
```

### 4. Check Results

Completions are saved to:
```
/home/ubuntu/rl-storage/AsyncRL/serving/DisTrainer/data/generations/batch_00001.jsonl
```

Each record includes verification rewards in the metadata.

---

## Expected Answer Format

Prime Intellect's MathRubric expects answers in **LaTeX `\boxed{}` format**:

### Good Format (Recommended)

```
<solution>
Let me solve this step by step:
1. First calculation: 24 × 2 = 48
2. Therefore, the answer is \boxed{48}
</solution>
```

### Also Works

```
<solution>48</solution>
```

The verifier will extract the numerical answer and verify it.

---

## Troubleshooting

### Issue: "verifiers library not installed"

```bash
uv add verifiers
# or
pip install verifiers
```

### Issue: "math-verify not installed"

```bash
pip install math-verify
```

### Issue: Verification always returns 0.0

Check that:
1. `<solution>` tags are present in completion
2. `expected_answer` is provided in prompts.jsonl
3. Answer format matches (use `\boxed{}` for best results)

### Issue: Import errors

```bash
# Ensure you're in the right environment
cd /home/ubuntu/rl-storage/AsyncRL/serving/DisGenerator
uv sync
```

---

## API Keys (Not Needed for Verification)

**You do NOT need API keys** for local verification with MathRubric.

API keys are only needed if you want to:
- Use Prime Intellect's **hosted inference** (optional)
- Use Prime Intellect's **training platform** (optional)
- Publish environments to their **hub** (optional)

For local verification, everything runs on your machine.

---

## Recommended Datasets

| Dataset | Size | Description | Best For |
|---------|------|-------------|----------|
| **GSM8K** | 8.5K | Grade school math | Multi-step reasoning |
| **MATH** | 12.5K | Competition math | Advanced problems |
| **AIME** | 1.4K | Math olympiad | Very hard problems |

All available on HuggingFace and work with Prime Intellect's MathRubric.

---

## Next Steps

1. **Test with small dataset** (10-100 samples)
2. **Verify rewards are computed** correctly
3. **Scale up** to full GSM8K training set
4. **Integrate into training loop** (DisTrainer)

For questions, see:
- [Prime Intellect Docs](https://docs.primeintellect.ai/verifiers)
- [Verifiers GitHub](https://github.com/PrimeIntellect-ai/verifiers)
