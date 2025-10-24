import logging
import re
from typing import List, Any, Optional
import sys
import os

# Add parent directory to path for grader_command_sender import
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SERVING_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
TOOL_GRPO_DIR = os.path.join(SERVING_DIR, 'ToolGRPOTrainer')
if TOOL_GRPO_DIR not in sys.path:
    sys.path.insert(0, TOOL_GRPO_DIR)

try:
    from grader_command_sender import send_grader_command
except ImportError:
    from ToolGRPOTrainer.grader_command_sender import send_grader_command

# Configure logging
logger = logging.getLogger(__name__)

def grader_reward_fn(completions: List[Any], prompts: List[str] = None, **kwargs) -> List[float]:
    """
    Grader reward function that sends completions to Grader-2 service for evaluation.

    This function integrates with the remote Grader-2 service running on a separate machine.
    For each completion, it sends the query and completion to Grader-2 via Service Bus,
    which runs a prefill/decode grading process and returns a numeric score (1-5).

    Args:
        completions: List of completions from GRPO trainer
        prompts: List of prompts corresponding to the completions
        **kwargs: Additional arguments (may contain prompts if not passed directly)

    Returns:
        List of reward scores (floats normalized from 0.0 to 1.0)
    """
    logger.info(f"Grader reward function called with {len(completions)} completions")

    # Extract prompts from kwargs if not provided directly
    if prompts is None:
        prompts = kwargs.get('prompts', [])

    # If we still don't have prompts, create placeholder prompts
    if not prompts or len(prompts) != len(completions):
        logger.warning(f"Prompts not available or length mismatch. Using placeholders.")
        prompts = [f"Prompt for completion {i}" for i in range(len(completions))]

    rewards = []

    for i, completion in enumerate(completions):
        try:
            # Extract content from completion structure
            if isinstance(completion, list) and len(completion) > 0:
                content = completion[0]["content"] if "content" in completion[0] else str(completion[0])
            else:
                content = str(completion)

            prompt = prompts[i] if i < len(prompts) else f"Prompt {i}"

            logger.info(f"Sending completion {i} to Grader-2 service")
            logger.debug(f"Prompt: {prompt[:100]}...")
            logger.debug(f"Completion: {content[:100]}...")

            # Send to Grader-2 and get score (now returns a bare number string like "1", "2", etc.)
            result = send_grader_command(prompt, content, timeout_s=30)

            reward_score = 0.0
            raw_score = _extract_raw_score(result)

            if raw_score is not None:
                reward_score = _normalize_score(raw_score)
                logger.info(
                    f"Grader-2 score for completion {i}: {raw_score} (normalized: {reward_score:.3f})"
                )
            else:
                logger.warning(f"Could not extract score from grader result: {result}")

            rewards.append(reward_score)
            logger.info(f"Completion {i} grader reward: {reward_score:.3f}")

        except Exception as e:
            logger.error(f"Error in grader_reward for completion {i}: {e}")
            rewards.append(0.0)

    logger.info(f"Grader rewards: {[f'{r:.3f}' for r in rewards]}")
    return rewards


def _extract_raw_score(result: Any) -> Optional[float]:
    """Attempt to coerce the grader result into a numeric score."""
    if result is None:
        return None

    text = str(result).strip()
    if not text:
        return None

    score = _try_parse_float(text)
    if score is not None:
        return score

    match = re.search(r"\d+\.?\d*", text)
    if match:
        return _try_parse_float(match.group())

    return None


def _try_parse_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_score(raw_score: float) -> float:
    """Map the 1-5 grader score onto the [0, 1] interval."""
    normalized = (raw_score - 1.0) / 4.0
    if normalized < 0.0:
        return 0.0
    if normalized > 1.0:
        return 1.0
    return normalized
