import logging
import re
import json
from parser import extract_all_content

# Configure logging
logger = logging.getLogger(__name__)

def tool_reward_fn(completions, **kwargs):
    """
    Reward function for tool-based completions.
    GRPO format: completions is a list of completion objects.
    
    Args:
        completions: List of completions from GRPO trainer
        **kwargs: Additional arguments
        
    Returns:
        List of reward scores (floats)
    """
    logger.info(f"Reward function called with {len(completions)} completions")
    
    rewards = []
    for i, completion in enumerate(completions):
        try:
            # Extract content from completion structure
            if isinstance(completion, list) and len(completion) > 0:
                # Format: [{"content": "..."}]
                content = completion[0]["content"] if "content" in completion[0] else str(completion[0])
            else:
                content = str(completion)
            
            logger.info(f"Processing completion {i}: {content[:100]}...")
            
            r = 0.0
            
            # 1. Reward for <think> tag presence
            if "<think>" in content and "</think>" in content:
                r += 0.2
                logger.info(f"<think> tag found in completion {i}")
            
            # 2. Reward for <web> tag presence
            if "<web>" in content and "</web>" in content:
                r += 0.3
                logger.info(f"<web> tag found in completion {i}")
                
                # Extra reward for proper web tag format
                web_pattern = r"<web>\s*\{\s*\"type\":\s*\"web\",\s*\"q\":\s*\"[^\"]+\",\s*\"k\":\s*\d+\s*\}\s*</web>"
                if re.search(web_pattern, content):
                    r += 0.2
                    logger.info(f"Proper web tag format found in completion {i}")
            
            # 3. Reward for <code> tag presence
            if "<code>" in content and "</code>" in content:
                r += 0.3
                logger.info(f"<code> tag found in completion {i}")
            
            # 4. Reward for <azure> tag presence
            if "<azure>" in content and "</azure>" in content:
                r += 0.3
                logger.info(f"<azure> tag found in completion {i}")
            
            # 5. Reward for proper pattern: <think>...<answer>...
            think_answer_pattern = r"<think>.*?</think>.*?<solution>.*?</solution>"
            if re.search(think_answer_pattern, content, re.DOTALL):
                r += 0.4
                logger.info(f"Think->Solution pattern found in completion {i}")
            
            # 6. Reward for longer answers (more detailed responses)
            if len(content) > 200:
                length_bonus = min(0.3, len(content) / 1000)  # Up to 0.3 bonus for long answers
                r += length_bonus
                logger.info(f"Length bonus {length_bonus:.2f} for completion {i} (length: {len(content)})")
            
            # Use the existing parser for additional validation
            try:
                parsed = extract_all_content(content)
                
                # Bonus for having valid tools detected by parser
                if parsed["has_tools"]:
                    r += 0.1
                    logger.info(f"Parser found tools in completion {i}")
                
                # Extra reward for valid parsed tools (original tool_type rewards)
                for tool in parsed["valid_tools"]:
                    if tool["type"] == "web":
                        r += 0.3  # Strong reward for valid web tools
                    elif tool["type"] == "code":
                        r += 0.3  # Strong reward for valid code tools  
                    elif tool["type"] == "azure":
                        r += 0.3  # Strong reward for valid azure tools
                    
                    r += 0.1  # Small bonus per any valid tool
                    logger.info(f"Valid {tool['type']} tool parsed in completion {i}")
                
                # Reward for solution content
                if parsed["solution"]:
                    r += 0.2
                    logger.info(f"Solution content found in completion {i}")
                    
                    # Extra reward for solution with documentation mentions
                    if "documentation" in parsed["solution"].lower():
                        r += 0.1
                        logger.info(f"Documentation mentioned in solution {i}")
                
                # Reward for reasoning content
                if parsed["reasoning"]:
                    r += 0.1
                    logger.info(f"Reasoning content found in completion {i}")
                    
            except Exception as parse_error:
                logger.warning(f"Parser error for completion {i}: {parse_error}")
            
            rewards.append(r)
            logger.info(f"Completion {i} total reward: {r:.2f}")
            
        except Exception as e:
            logger.error(f"Error processing completion {i}: {e}")
            rewards.append(0.0)
    
    logger.info(f"Final rewards: {[f'{r:.2f}' for r in rewards]}")
    return rewards
