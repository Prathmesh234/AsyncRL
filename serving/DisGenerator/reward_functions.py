"""
Reward Functions for DisGenerator.
Adapted from ToolGRPOTrainer to compute rewards for individual trajectories.

The main compute_reward() function combines all reward functions and returns
a single reward value for each trajectory before it's saved to batch.jsonl.

Includes DAPO Soft Overlong Punishment for length-based penalization.
Paper: https://arxiv.org/abs/2503.14476
"""

import logging
import re
import json
import sys
import os
from typing import Any, Optional

# Add parent directory to path for parser and grader imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from parser import extract_all_content
except ImportError:
    # Fallback if parser not available
    def extract_all_content(content):
        return {"has_tools": False, "valid_tools": [], "solution": None, "reasoning": None}

logger = logging.getLogger(__name__)


def tool_reward(content: str) -> float:
    """
    Reward for proper tool usage (web, code, azure tags).
    Rewards proper tag pairing and usage patterns.
    """
    r = 0.0
    
    # Reward for <think> tag (properly closed)
    if "<think>" in content and "</think>" in content:
        if content.count("<think>") == content.count("</think>"):
            r += 0.2
    
    # Reward for <web> tag (properly closed)
    if "<web>" in content and "</web>" in content:
        if content.count("<web>") == content.count("</web>"):
            r += 0.3
    
    # Reward for <code> tag (properly closed)
    if "<code>" in content and "</code>" in content:
        if content.count("<code>") == content.count("</code>"):
            r += 0.3
    
    # Reward for <azure> tag (properly closed)
    if "<azure>" in content and "</azure>" in content:
        if content.count("<azure>") == content.count("</azure>"):
            r += 0.3
    
    # Reward for <solution> tag (properly closed)
    if "<solution>" in content and "</solution>" in content:
        if content.count("<solution>") == content.count("</solution>"):
            r += 0.4
    
    # Reward for think -> solution pattern
    think_solution_pattern = r"<think>.*?</think>.*?<solution>.*?</solution>"
    if re.search(think_solution_pattern, content, re.DOTALL):
        r += 0.5
    
    # Length bonus
    if len(content) > 200:
        r += min(0.3, len(content) / 1000)
    
    # Parser-based bonus
    try:
        parsed = extract_all_content(content)
        if parsed["has_tools"]:
            r += 0.1
        for tool in parsed.get("valid_tools", []):
            if tool["type"] in ["web", "code", "azure"]:
                r += 0.3
            r += 0.1  # Bonus per valid tool
        if parsed.get("solution"):
            r += 0.2
        if parsed.get("reasoning"):
            r += 0.1
    except Exception:
        pass
    
    return r


def char_reward(content: str) -> float:
    """
    DAPO Soft Overlong Punishment - Length-based reward/penalty.
    
    Implements the formula from DAPO paper (arXiv:2503.14476, Section 3.4):
        R_overlong = -B^((L_generated - L_expected) / penalty)
    
    This is a "soft punishment" that:
    - Gives small positive rewards for reasonable length responses
    - Applies exponentially increasing penalty for overly long responses
    - Does not harshly penalize correct but lengthy reasoning
    - Stabilizes training by reducing reward noise for truncated samples
    
    Paper: https://arxiv.org/abs/2503.14476
    """
    r = 0.0
    length = len(content)
    
    # ==========================================================================
    # DAPO Configuration
    # ==========================================================================
    DAPO_BASE = 2.0              # B: Base of exponential penalty
    DAPO_EXPECTED_LENGTH = 8192  # L_expected: Max expected length (chars)
    DAPO_PENALTY_SCALE = 2048    # Scaling factor for penalty steepness
    DAPO_MAX_PENALTY = 2.0       # Cap on maximum penalty
    
    # ==========================================================================
    # Positive rewards for reasonable length (up to expected length)
    # ==========================================================================
    if length > 100:
        r += 0.1
    if length > 500:
        r += 0.1
    if length > 1000:
        r += 0.1
    if length > 2000:
        r += 0.1
    
    # ==========================================================================
    # DAPO Soft Overlong Punishment (exponential penalty for exceeding length)
    # ==========================================================================
    if length > DAPO_EXPECTED_LENGTH:
        # R_overlong = -B^((L_generated - L_expected) / penalty)
        try:
            exponent = (length - DAPO_EXPECTED_LENGTH) / DAPO_PENALTY_SCALE
            length_penalty = -(DAPO_BASE ** exponent)
            
            # Cap the penalty to avoid extreme values
            length_penalty = max(-DAPO_MAX_PENALTY, length_penalty)
            
            r += length_penalty
            
            logger.debug(
                f"DAPO length penalty: L_gen={length}, L_exp={DAPO_EXPECTED_LENGTH}, "
                f"exponent={exponent:.3f}, penalty={length_penalty:.3f}"
            )
        except (OverflowError, ValueError) as e:
            logger.warning(f"DAPO length penalty computation error: {e}")
            r -= DAPO_MAX_PENALTY  # Apply max penalty on error
    
    # ==========================================================================
    # Additional quality checks (kept from original)
    # ==========================================================================
    
    # Penalize excessive use of "Wait" (indicates rambling/unproductive thinking)
    wait_count = len(re.findall(r'\bWait\b', content, re.IGNORECASE))
    if wait_count > 3:
        r -= min(0.3, (wait_count - 3) * 0.05)  # Penalty grows with usage
    
    # ==========================================================================
    # Code patterns reward (restored from original)
    # ==========================================================================
    code_patterns = [
        r'`[^`]+`',        # Inline code
        r'```[\s\S]*?```', # Code blocks
        r'\w+\.\w+\(',     # Method calls
    ]
    for pattern in code_patterns:
        matches = len(re.findall(pattern, content))
        r += min(0.1, matches * 0.02)
    
    # Sentence structure
    sentences = content.split('.')
    if len(sentences) > 2:
        r += 0.05
    
    # Penalize repetition
    words = content.lower().split()
    if len(words) > 10:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.7:
            r -= 0.1
        elif unique_ratio > 0.9:
            r += 0.05
    
    return r


def format_reward(content: str) -> float:
    """
    Format-based reward for proper structure and formatting.
    """
    r = 0.0
    
    # Tag pairing
    tag_pairs = [
        ('<think>', '</think>'),
        ('<web>', '</web>'),
        ('<code>', '</code>'),
        ('<azure>', '</azure>'),
        ('<solution>', '</solution>')
    ]
    
    for start_tag, end_tag in tag_pairs:
        if start_tag in content and end_tag in content:
            if content.count(start_tag) == content.count(end_tag):
                r += 0.1
            else:
                r -= 0.05
    
    # JSON validation in tool tags
    json_patterns = [
        (r'<web>\s*(\{[^}]+\})\s*</web>', 'web'),
        (r'<code>\s*(\{[^}]+\})\s*</code>', 'code'),
        (r'<azure>\s*(\{[^}]+\})\s*</azure>', 'azure')
    ]
    
    for pattern, tool_type in json_patterns:
        matches = re.findall(pattern, content)
        for match in matches:
            try:
                data = json.loads(match)
                if str(data.get("type", "")).lower() == tool_type:
                    r += 0.1
            except json.JSONDecodeError:
                r -= 0.05
    
    # Workflow pattern
    has_think = '<think>' in content and '</think>' in content
    has_action = any(tag in content for tag in ['<web>', '<code>', '<azure>'])
    has_solution = '<solution>' in content and '</solution>' in content
    
    if has_think and has_action:
        r += 0.2
    if has_think and has_solution:
        r += 0.2
    if has_think and has_action and has_solution:
        r += 0.1
    
    # Multi-line structure
    if len(content.split('\n')) > 3:
        r += 0.05
    
    # Code blocks
    if '```' in content:
        code_blocks = re.findall(r'```[\s\S]*?```', content)
        for block in code_blocks:
            if len(block.split('\n')) > 2:
                r += 0.1
    
    return r


def compute_reward(completion_text: str, prompt: str = None, expected_answer: str = None) -> float:
    """
    Compute the total reward for a completion.
    
    Combines rewards from:
    - tool_reward: Tool tag usage and patterns
    - char_reward: DAPO Soft Overlong Punishment (length-based reward/penalty)
    - format_reward: Proper formatting
    - verification_reward: Prime Intellect solution verifier (replaces grader)
    
    Args:
        completion_text: The full completion text
        prompt: The original prompt
        expected_answer: Ground truth answer for verification (optional)
        
    Returns:
        Total reward as a float
    """
    total = 0.0
    
    # Combine all rewards
    total += tool_reward(completion_text)
    total += char_reward(completion_text)  # Includes DAPO length penalty
    total += format_reward(completion_text)
    
    # Use solution verifier instead of grader
    try:
        verification_score = verification_reward(completion_text, expected_answer)
        total += verification_score
    except Exception as e:
        logger.warning(f"Verification reward failed: {e}")
    
    logger.debug(f"Computed reward: {total:.3f}")
    return total


def verification_reward(completion_text: str, expected_answer: str = None) -> float:
    """
    Verify solution using Prime Intellect's verifiers.
    
    Extracts <solution> tag and verifies against expected answer
    using MathRubric for symbolic equivalence checking.
    
    Args:
        completion_text: Full completion text with <solution> tag
        expected_answer: Ground truth answer
        
    Returns:
        1.0 if correct, 0.0 if incorrect or no answer provided
    """
    if not expected_answer:
        logger.debug("No expected_answer provided, skipping verification")
        return 0.0
    
    try:
        from solution_verifier import compute_verification_reward
        reward = compute_verification_reward(completion_text, expected_answer)
        logger.info(f"Verification reward: {reward}")
        return reward
    except ImportError:
        logger.warning("solution_verifier not available")
        return 0.0
    except Exception as e:
        logger.error(f"Verification error: {e}")
        return 0.0


# ==========================================================================
# ORIGINAL GRADER (COMMENTED OUT - Using solution_verifier instead)
# ==========================================================================
# def grader_reward(completion_text: str, prompt: str, timeout_s: int = 30) -> float:
#     """
#     Call external grader service for reward.
#     Returns normalized score (0.0 to 1.0).
#     """
#     try:
#         from ToolGRPOTrainer.grader_command_sender import send_grader_command
#         
#         result = send_grader_command(prompt, completion_text, timeout_s=timeout_s)
#         
#         raw_score = None
#         if result:
#             text = str(result).strip()
#             match = re.search(r'\d+\.?\d*', text)
#             if match:
#                 raw_score = float(match.group())
#         
#         if raw_score is not None:
#             normalized = (raw_score - 1.0) / 4.0
#             return max(0.0, min(1.0, normalized))
#             
#         return 0.0
#         
#     except ImportError:
#         logger.warning("Grader command sender not available.")
#         return 0.0
#     except Exception as e:
#         logger.error(f"Grader error: {e}")
#         return 0.0
