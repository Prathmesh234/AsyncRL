"""
GRPO (Group Relative Policy Optimization) loss computation.
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Any


def compute_grpo_loss(
    model,
    group_batch: List[Dict[str, Any]],
    beta: float = 0.01
) -> torch.Tensor:
    """
    Compute GRPO loss for a batch of generation groups.
    
    GRPO uses group-relative advantages instead of a baseline:
    - For each group (1 prompt + N completions), compute mean reward
    - Advantage = reward - group_mean
    - Normalize advantages within group
    - Compute policy gradient loss with KL penalty
    
    Args:
        model: Policy model (FSDP2-wrapped)
        group_batch: List of groups, each containing:
            - prompt_ids: List[int] - tokenized prompt
            - completions: List of dicts with:
                - completion_ids: List[int] - tokenized completion
                - reward: float - reward score
                - old_logprobs: List[float] - log probs from generator
        beta: KL penalty coefficient
    
    Returns:
        Scalar loss tensor
    """
    total_loss = torch.tensor(0.0, device="cuda", requires_grad=True)
    num_completions = 0
    
    for group in group_batch:
        prompt_ids = group["prompt_ids"]
        completions = group["completions"]
        
        if len(completions) == 0:
            continue
        
        # 1. Extract rewards and compute group-relative advantages
        rewards = torch.tensor(
            [c["reward"] for c in completions],
            device="cuda",
            dtype=torch.float32
        )
        
        # Compute normalized advantages
        mean_reward = rewards.mean()
        std_reward = rewards.std() + 1e-8
        advantages = (rewards - mean_reward) / std_reward
        
        # 2. Forward pass for each completion
        for i, completion in enumerate(completions):
            completion_ids = completion["completion_ids"]
            old_logprobs = completion.get("old_logprobs", None)
            action_mask = completion.get("action_mask", None)
            
            # Concatenate prompt + completion
            input_ids = torch.tensor(
                prompt_ids + completion_ids,
                device="cuda",
                dtype=torch.long
            ).unsqueeze(0)  # Add batch dimension
            
            # Get new logprobs from current policy
            # Note: mixed precision context handled by FSDP, but explicit cast ensures safety
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                outputs = model(input_ids)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            
            # Compute log probabilities
            log_probs = F.log_softmax(logits, dim=-1)
            
            # Get log probs for the completion tokens only
            # Shift: logits[t] predicts token[t+1]
            prompt_len = len(prompt_ids)
            completion_len = len(completion_ids)
            
            # Extract completion log probs (only for tokens where action_mask=1)
            completion_logprobs = []
            masked_old_logprobs = []
            
            # The logit at index i predicts the token at index i+1
            # To predict the first completion token (at index prompt_len), 
            # we need the logit at index (prompt_len - 1)
            start_logit_idx = prompt_len - 1
            
            for t in range(completion_len):
                # Skip if action_mask is 0 (tool result token)
                if action_mask is not None and t < len(action_mask) and action_mask[t] == 0:
                    continue
                    
                logit_idx = start_logit_idx + t
                
                # Ensure we don't go out of bounds
                if logit_idx < input_ids.size(1) - 1:
                    # The target token is the NEXT token after this logit
                    target_token = input_ids[0, logit_idx + 1]
                    token_logprob = log_probs[0, logit_idx, target_token]
                    completion_logprobs.append(token_logprob)
                    
                    # Also collect corresponding old logprob for KL
                    if old_logprobs is not None and t < len(old_logprobs):
                        masked_old_logprobs.append(old_logprobs[t])
            
            if len(completion_logprobs) == 0:
                continue
            
            # Sum log probs for the completion (only model-generated tokens)
            new_logprob_sum = torch.stack(completion_logprobs).sum()
            
            # 3. Compute policy gradient loss
            # Loss = -advantage * log_prob
            policy_loss = -advantages[i] * new_logprob_sum
            
            # 4. Add KL penalty if old_logprobs available
            if len(masked_old_logprobs) > 0:
                old_logprob_tensor = torch.tensor(
                    masked_old_logprobs,
                    device="cuda",
                    dtype=torch.float32
                )
                new_logprob_tensor = torch.stack(completion_logprobs)
                
                # KL divergence: sum(old - new) - only for model-generated tokens
                kl = (old_logprob_tensor - new_logprob_tensor).sum()
                policy_loss = policy_loss + beta * kl
            
            total_loss = total_loss + policy_loss
            num_completions += 1
    
    # Average over all completions
    if num_completions > 0:
        total_loss = total_loss / num_completions
    
    return total_loss
