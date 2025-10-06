import logging
import requests
import os
import time
from typing import List, Any

# Configure logging
logger = logging.getLogger(__name__)

def external_judge_reward_fn(completions: List[Any], prompts: List[str] = None, **kwargs) -> List[float]:
    """
    External judge reward function that sends completions to an external judging service.
    
    Args:
        completions: List of completions from GRPO trainer
        prompts: List of prompts corresponding to the completions
        **kwargs: Additional arguments (may contain prompts if not passed directly)
        
    Returns:
        List of reward scores (floats)
    """
    logger.info(f"External judge reward function called with {len(completions)} completions")
    
    # Get judge service IP from environment
    judge_ip = os.getenv("JUDGE_SERVICE_IP", "localhost:8080")
    send_endpoint = f"http://{judge_ip}/send-judge"
    recv_endpoint = f"http://{judge_ip}/recv-judge"
    
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
            
            # Prepare payload for judge service
            payload = {
                "completion_id": i,
                "prompt": prompts[i] if i < len(prompts) else f"Prompt {i}",
                "completion": content,
                "metadata": {
                    "timestamp": time.time(),
                    "completion_length": len(content)
                }
            }
            
            logger.info(f"Sending completion {i} to judge service at {send_endpoint}")
            
            # Send completion to judge service and get reward
            try:
                # Send completion to judge service
                send_response = requests.post(
                    send_endpoint,
                    json=payload,
                    timeout=30,
                    headers={"Content-Type": "application/json"}
                )
                send_response.raise_for_status()
                send_result = send_response.json()
                judge_id = send_result.get("judge_id", f"completion_{i}")
                logger.info(f"Successfully sent completion {i}, received judge_id: {judge_id}")
                
                # Wait a brief moment for processing
                time.sleep(1)
                
                # Get judgment result
                recv_response = requests.get(
                    recv_endpoint,
                    params={"judge_id": judge_id},
                    timeout=10
                )
                
                if recv_response.status_code == 200:
                    result = recv_response.json()
                    reward_score = float(result.get("score", 0.0))
                    logger.info(f"Received reward score {reward_score} for completion {i}")
                else:
                    logger.warning(f"Received status code {recv_response.status_code} when checking completion {i}")
                    reward_score = 0.0
                
            except Exception as e:
                logger.error(f"Failed to get judgment for completion {i}: {e}")
                reward_score = 0.0
            
            rewards.append(reward_score)
            logger.info(f"Completion {i} external judge reward: {reward_score:.3f}")
            
        except Exception as e:
            logger.error(f"Error in external_judge_reward for completion {i}: {e}")
            rewards.append(0.0)
    
    logger.info(f"External judge rewards: {rewards}")
    return rewards
