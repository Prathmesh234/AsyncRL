import logging
import re
from typing import List, Any
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

            # Send to Grader-2 and get score
            result = send_grader_command(prompt, content, timeout_s=30)

            # Parse the result to extract numeric score
            reward_score = 0.0
            if "[grader-score]" in result:
                # Extract the numeric score from the result
                score_str = result.replace("[grader-score]", "").strip()
                try:
                    # Grader-2 returns scores from 1-5, normalize to 0.0-1.0
                    raw_score = float(score_str)
                    # Normalize: 1->0.0, 2->0.25, 3->0.5, 4->0.75, 5->1.0
                    reward_score = (raw_score - 1.0) / 4.0
                    logger.info(f"Grader-2 score for completion {i}: {raw_score} (normalized: {reward_score:.3f})")
                except ValueError:
                    logger.error(f"Failed to parse grader score: {score_str}")
                    reward_score = 0.0
            elif "[grader-result]" in result:
                # Try to extract score from JSON result
                try:
                    import json
                    result_json = result.replace("[grader-result]", "").strip()
                    parsed = json.loads(result_json)
                    raw_score = float(parsed.get("message", 0))
                    reward_score = (raw_score - 1.0) / 4.0
                    logger.info(f"Grader-2 score for completion {i}: {raw_score} (normalized: {reward_score:.3f})")
                except Exception as e:
                    logger.error(f"Failed to parse grader result: {e}")
                    reward_score = 0.0
            elif "[grader-error]" in result:
                logger.error(f"Grader-2 error for completion {i}: {result}")
                reward_score = 0.0
            else:
                # Try to extract any number from the result
                match = re.search(r'\d+\.?\d*', result)
                if match:
                    raw_score = float(match.group())
                    reward_score = (raw_score - 1.0) / 4.0
                    logger.info(f"Extracted grader score for completion {i}: {raw_score} (normalized: {reward_score:.3f})")
                else:
                    logger.warning(f"Could not extract score from grader result: {result}")
                    reward_score = 0.0

            rewards.append(reward_score)
            logger.info(f"Completion {i} grader reward: {reward_score:.3f}")

        except Exception as e:
            logger.error(f"Error in grader_reward for completion {i}: {e}")
            rewards.append(0.0)

    logger.info(f"Grader rewards: {[f'{r:.3f}' for r in rewards]}")
    return rewards
