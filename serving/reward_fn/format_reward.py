import logging
import re
import json

# Configure logging
logger = logging.getLogger(__name__)

def format_reward_fn(completions, **kwargs):
    """
    Format-based reward function focusing on proper structure and formatting.
    
    Args:
        completions: List of completions from GRPO trainer
        **kwargs: Additional arguments
        
    Returns:
        List of reward scores (floats)
    """
    logger.info(f"Format reward function called with {len(completions)} completions")
    
    rewards = []
    for i, completion in enumerate(completions):
        try:
            # Extract content from completion structure
            if isinstance(completion, list) and len(completion) > 0:
                content = completion[0]["content"] if "content" in completion[0] else str(completion[0])
            else:
                content = str(completion)
            
            r = 0.0
            
            # 1. Reward for proper tag structure
            tag_pairs = [
                ('<think>', '</think>'),
                ('<web>', '</web>'),
                ('<code>', '</code>'),
                ('<azure>', '</azure>'),
                ('<solution>', '</solution>')
            ]
            
            for start_tag, end_tag in tag_pairs:
                if start_tag in content and end_tag in content:
                    # Check if tags are properly paired
                    start_count = content.count(start_tag)
                    end_count = content.count(end_tag)
                    if start_count == end_count:
                        r += 0.1
                        logger.info(f"Proper {start_tag} tag pairing in completion {i}")
                    else:
                        r -= 0.05  # Penalty for mismatched tags
            
            # 2. Reward for proper tool tag format patterns
            # Check for exact format patterns specified in system prompt
            tool_format_patterns = [
                (r"<web>\s*\{\s*\"type\":\s*\"web\",\s*\"q\":\s*\"[^\"]+\",\s*\"k\":\s*([1-9]|10)\s*\}\s*</web>", 'web'),
                (r"<code>\s*\{\s*\"type\":\s*\"code\",\s*\"code_command\":\s*\"[^\"]+\"\s*\}\s*</code>", 'code'),
                (r"<azure>\s*\{\s*\"type\":\s*\"azure\",\s*\"azure_command\":\s*\"[^\"]+\"\s*\}\s*</azure>", 'azure')
            ]
            
            for pattern, tool_type in tool_format_patterns:
                if re.search(pattern, content):
                    r += 0.25  # Strong reward for exact format compliance
                    logger.info(f"Perfect {tool_type} tag format found in completion {i}")
            
            # 3. General JSON format validation in tool tags
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
                    except json.JSONDecodeError:
                        r -= 0.05  # Penalty for invalid JSON
                        logger.warning(f"Invalid JSON in {tool_type} tag in completion {i}")
                        continue

                    payload_type = str(data.get("type", "")).strip().lower()
                    if payload_type == tool_type:
                        r += 0.10  # Moderate reward for valid JSON with correct type
                        logger.info(f"Valid JSON in {tool_type} tag in completion {i}")
                    else:
                        r -= 0.05  # Penalize missing/mismatched type to reinforce schema
                        logger.warning(
                            f"Incorrect or missing type in {tool_type} tag in completion {i}: {payload_type!r}"
                        )
            
            # 4. Reward for proper workflow structure
            # Check for think -> action -> solution pattern
            has_think = '<think>' in content and '</think>' in content
            has_action = any(tag in content for tag in ['<web>', '<code>', '<azure>'])
            has_solution = '<solution>' in content and '</solution>' in content
            
            if has_think and has_action:
                r += 0.2
                logger.info(f"Think -> Action pattern in completion {i}")
            
            if has_think and has_solution:
                r += 0.2
                logger.info(f"Think -> Solution pattern in completion {i}")
            
            if has_think and has_action and has_solution:
                r += 0.1  # Bonus for complete workflow
                logger.info(f"Complete Think -> Action -> Solution workflow in completion {i}")
            
            # 5. Reward for proper line breaks and structure
            lines = content.split('\n')
            if len(lines) > 3:
                r += 0.05  # Bonus for multi-line structure
            
            # 6. Reward for code formatting
            if '```' in content:
                code_blocks = re.findall(r'```[\s\S]*?```', content)
                for block in code_blocks:
                    if len(block.split('\n')) > 2:  # Multi-line code block
                        r += 0.1
                        logger.info(f"Multi-line code block found in completion {i}")
            
            # 7. Reward for list formatting
            list_patterns = [
                r'^\s*[-*+]\s+',  # Bullet points
                r'^\s*\d+\.\s+',  # Numbered lists
            ]
            
            for pattern in list_patterns:
                matches = len(re.findall(pattern, content, re.MULTILINE))
                if matches > 1:
                    r += 0.05
                    logger.info(f"List formatting found in completion {i}")
                    break
            
            # 8. Penalty for overly complex nested structures
            total_tags = sum(content.count(f'<{tag}>') for tag in ['think', 'web', 'code', 'azure', 'solution'])
            if total_tags > 6:  # Too many tags might indicate confusion
                r -= 0.1
                logger.warning(f"Too many tags ({total_tags}) in completion {i}")
            
            # 9. Reward for proper spacing around tags
            well_spaced_tags = 0
            for tag in ['think', 'web', 'code', 'azure', 'solution']:
                pattern = rf'\n\s*<{tag}>.*?</{tag}>\s*\n'
                if re.search(pattern, content, re.DOTALL):
                    well_spaced_tags += 1
            
            if well_spaced_tags > 0:
                r += well_spaced_tags * 0.02  # Small bonus per well-spaced tag
            
            rewards.append(r)
            logger.info(f"Completion {i} format reward: {r:.3f}")
            
        except Exception as e:
            logger.error(f"Error in format_reward for completion {i}: {e}")
            rewards.append(0.0)
    
    return rewards
