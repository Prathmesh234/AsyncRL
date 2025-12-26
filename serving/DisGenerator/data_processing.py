"""
Data processing functions for the DisGenerator.
Handles trajectory record building, message formatting, and logprob alignment.
"""

import time
from typing import List, Dict, Any, Optional


def messages_to_prompt_string(messages: List[Dict[str, str]], tokenizer) -> str:
    """
    Convert messages list to a plain prompt string.
    Uses the chat template if available, otherwise concatenates content.
    
    Args:
        messages: List of message dicts with 'role' and 'content'
        tokenizer: HuggingFace tokenizer with apply_chat_template
        
    Returns:
        Formatted prompt string
    """
    try:
        # Use chat template for proper formatting (Qwen uses this)
        prompt_str = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        return prompt_str
    except Exception:
        # Fallback: simple concatenation
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"{role}: {content}")
        return "\n".join(parts)


def extract_prompt_messages(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Extract the prompt messages (system + user) before any assistant response.
    
    Args:
        messages: Full list of conversation messages
        
    Returns:
        List containing only system and user messages before first assistant
    """
    prompt_messages = []
    for msg in messages:
        role = msg.get("role", "")
        if role in ["system", "user"]:
            prompt_messages.append(msg)
        else:
            break  # Stop at first non-prompt message
    return prompt_messages


def build_completion_text(messages: List[Dict[str, str]], completions: List[str]) -> str:
    """
    Build full completion text by interleaving completions with tool results.
    
    Args:
        messages: Full conversation messages (includes tool results)
        completions: List of assistant completion texts from each turn
        
    Returns:
        Full completion text with tool results interleaved
    """
    completion_parts = []
    completion_idx = 0
    
    for msg in messages:
        role = msg.get("role", "")
        if role in ["system", "user"]:
            continue  # Skip prompt messages
        elif role == "assistant":
            # Use accumulated completion content
            if completion_idx < len(completions):
                completion_parts.append(completions[completion_idx])
                completion_idx += 1
        elif role == "tool":
            # Tool result - add it to completion
            completion_parts.append(msg.get("content", ""))
    
    return "\n".join(completion_parts)


def align_logprobs(logprobs: List[float], target_length: int, pad_value: float = -1.0) -> List[float]:
    """
    Pad or truncate logprobs to match a target length.
    
    Args:
        logprobs: Original list of log probabilities
        target_length: Desired length to match (e.g., len(completion_ids))
        pad_value: Value to use for padding if logprobs is shorter
        
    Returns:
        List of logprobs with exactly target_length elements
    """
    result = logprobs.copy()
    
    if len(result) < target_length:
        # Pad with placeholder values
        result.extend([pad_value] * (target_length - len(result)))
    elif len(result) > target_length:
        # Truncate to match
        result = result[:target_length]
    
    return result


def align_action_mask(action_mask: List[int], target_length: int, pad_value: int = 1) -> List[int]:
    """
    Pad or truncate action_mask to match a target length.
    
    Args:
        action_mask: Original list of mask values (1 = include, 0 = skip)
        target_length: Desired length to match (e.g., len(completion_ids))
        pad_value: Value to use for padding (default 1 to include in loss)
        
    Returns:
        List of mask values with exactly target_length elements
    """
    result = action_mask.copy()
    
    if len(result) < target_length:
        # Pad with default value (1 = include in loss)
        result.extend([pad_value] * (target_length - len(result)))
    elif len(result) > target_length:
        # Truncate to match
        result = result[:target_length]
    
    return result


def build_trajectory_record(
    traj_id: str,
    group_id: str,
    messages: List[Dict[str, str]],
    completions: List[str],
    accumulated_logprobs: List[float],
    action_mask: List[int],
    tokenizer,
    reward: float = 0.0
) -> Dict[str, Any]:
    """
    Build the complete record for saving a trajectory to JSONL.
    
    Args:
        traj_id: Unique ID for this trajectory
        group_id: Shared group ID for GRPO grouping
        messages: Full conversation messages
        completions: List of assistant completion texts
        accumulated_logprobs: Streaming logprobs from generation
        action_mask: List of mask values (1 = model token, 0 = tool token)
        tokenizer: HuggingFace tokenizer for encoding
        reward: Pre-computed reward score for this completion
        
    Returns:
        Complete record dict ready for JSONL serialization
    """
    # Extract prompt messages
    prompt_messages = extract_prompt_messages(messages)
    
    # Convert prompt to string
    prompt_str = messages_to_prompt_string(prompt_messages, tokenizer)
    
    # Build full completion text
    full_completion = build_completion_text(messages, completions)
    
    # Tokenize prompt and completion
    prompt_encoding = tokenizer(prompt_str, add_special_tokens=False, return_tensors=None)
    prompt_ids = prompt_encoding["input_ids"]
    
    completion_encoding = tokenizer(full_completion, add_special_tokens=False, return_tensors=None)
    completion_ids = completion_encoding["input_ids"]
    
    # Align logprobs and action_mask to completion length
    old_logprobs = align_logprobs(accumulated_logprobs, len(completion_ids))
    aligned_mask = align_action_mask(action_mask, len(completion_ids))
    
    return {
        "gen_id": traj_id,
        "group_id": group_id,
        "prompt": prompt_str,
        "prompt_ids": prompt_ids,
        "completion": {
            "text": full_completion,
            "completion_ids": completion_ids,
            "old_logprobs": old_logprobs,
            "action_mask": aligned_mask,  # 1 = include in loss, 0 = skip
            "reward": reward  # Computed by reward functions
        },
        "metadata": {
            "timestamp": time.time(),
            "status": "COMPLETED",
            "num_turns": len(completions)
        }
    }

