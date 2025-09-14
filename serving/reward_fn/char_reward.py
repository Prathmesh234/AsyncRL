import logging
import re

# Configure logging
logger = logging.getLogger(__name__)

def char_reward_fn(completions, **kwargs):
    """
    Character-based reward function focusing on response quality indicators.
    
    Args:
        completions: List of completions from GRPO trainer
        **kwargs: Additional arguments
        
    Returns:
        List of reward scores (floats)
    """
    logger.info(f"Character reward function called with {len(completions)} completions")
    
    rewards = []
    for i, completion in enumerate(completions):
        try:
            # Extract content from completion structure
            if isinstance(completion, list) and len(completion) > 0:
                content = completion[0]["content"] if "content" in completion[0] else str(completion[0])
            else:
                content = str(completion)
            
            r = 0.0
            
            # 1. Length-based rewards (encourage detailed responses)
            length = len(content)
            if length > 100:
                r += 0.1
            if length > 300:
                r += 0.1
            if length > 500:
                r += 0.1
            # Cap at reasonable length to avoid overly verbose responses
            if length > 2000:
                r -= 0.1
                
            # 2. Reward for technical vocabulary
            technical_terms = [
                'function', 'method', 'class', 'variable', 'parameter', 'argument',
                'API', 'endpoint', 'database', 'query', 'authentication', 'authorization',
                'configuration', 'deployment', 'container', 'service', 'application',
                'implementation', 'documentation', 'tutorial', 'example', 'guide'
            ]
            
            tech_count = sum(1 for term in technical_terms if term.lower() in content.lower())
            r += min(0.2, tech_count * 0.02)  # Up to 0.2 bonus
            
            # 3. Reward for code-like patterns
            code_patterns = [
                r'`[^`]+`',  # Inline code
                r'```[\s\S]*?```',  # Code blocks
                r'\w+\.\w+\(',  # Method calls
                r'import \w+',  # Import statements
                r'def \w+\(',  # Function definitions
                r'class \w+',  # Class definitions
            ]
            
            for pattern in code_patterns:
                matches = len(re.findall(pattern, content))
                r += min(0.1, matches * 0.02)  # Small bonus per code pattern
            
            # 4. Reward for proper sentence structure
            sentences = content.split('.')
            if len(sentences) > 2:
                r += 0.05  # Bonus for multiple sentences
            
            # 5. Penalize repetitive content
            words = content.lower().split()
            if len(words) > 10:
                unique_words = len(set(words))
                repetition_ratio = unique_words / len(words)
                if repetition_ratio < 0.7:  # High repetition
                    r -= 0.1
                elif repetition_ratio > 0.9:  # Good variety
                    r += 0.05
            
            # 6. Reward for question words (shows engagement)
            question_words = ['what', 'how', 'why', 'when', 'where', 'which', 'who']
            question_count = sum(1 for word in question_words if word in content.lower())
            r += min(0.05, question_count * 0.01)
            
            rewards.append(r)
            logger.info(f"Completion {i} character reward: {r:.3f} (length: {length})")
            
        except Exception as e:
            logger.error(f"Error in char_reward for completion {i}: {e}")
            rewards.append(0.0)
    
    return rewards
