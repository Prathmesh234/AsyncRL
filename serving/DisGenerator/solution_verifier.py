"""
Solution Verifier Integration for DisGenerator using Prime Intellect's Verifiers.

Uses Prime Intellect's built-in utility functions for math verification.
Simple and clean implementation following official documentation.

Installation:
    pip install verifiers

No API keys required for local verification.
"""

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def extract_solution(completion_text: str) -> Optional[str]:
    """
    Extract text from <solution>...</solution> tags.
    
    Args:
        completion_text: Full completion text from model
        
    Returns:
        Solution text or None if not found
    """
    pattern = r'<solution>(.*?)</solution>'
    match = re.search(pattern, completion_text, re.DOTALL)
    if match:
        solution = match.group(1).strip()
        logger.debug(f"Extracted solution: {solution[:100]}...")
        return solution
    
    logger.debug("No <solution> tag found in completion")
    return None


def verify_math_solution(solution: str, expected_answer: str) -> float:
    """
    Verify mathematical solution using Prime Intellect's verifiers utilities.
    
    Uses vf.extract_boxed_answer() for LaTeX format and simple string matching.
    
    Args:
        solution: Extracted solution text (may contain \\boxed{} format)
        expected_answer: Ground truth answer from dataset
        
    Returns:
        Reward score (1.0 for correct, 0.0 for incorrect)
        
    Raises:
        ImportError: If verifiers library is not installed
    """
    try:
        import verifiers as vf
    except ImportError:
        raise ImportError(
            "Prime Intellect verifiers library is required for solution verification.\n"
            "Install with: uv add verifiers\n"
            "or: pip install verifiers"
        )
    
    # Use Prime Intellect's built-in utility to extract boxed answer
    solution_answer = vf.extract_boxed_answer(solution)
    if not solution_answer:
        # Fallback: use entire solution if no \boxed{} found
        solution_answer = solution.strip()
    
    # Also try to extract from expected answer
    expected = vf.extract_boxed_answer(expected_answer)
    if not expected:
        expected = expected_answer.strip()
    
    logger.info(f"Verifying: '{solution_answer}' == '{expected}'")
    
    # Normalize and compare
    solution_norm = solution_answer.lower().strip()
    expected_norm = expected.lower().strip()
    
    if solution_norm == expected_norm:
        logger.info("Verification: EXACT MATCH")
        return 1.0
    
    # Partial match
    if expected_norm in solution_norm:
        logger.info("Verification: PARTIAL MATCH")
        return 0.5
    
    logger.info("Verification: NO MATCH")
    return 0.0


def compute_verification_reward(completion_text: str, expected_answer: str = None) -> float:
    """
    Main entry point for solution verification.
    
    Extracts solution from completion and verifies against expected answer
    using Prime Intellect's verifiers utilities.
    
    Args:
        completion_text: Full completion text from model
        expected_answer: Ground truth answer (optional)
        
    Returns:
        Verification reward (0.0 to 1.0)
        
    Raises:
        ImportError: If verifiers library is not installed
    """
    # Extract solution
    solution = extract_solution(completion_text)
    
    if not solution:
        logger.debug("No solution found, returning 0.0 reward")
        return 0.0
    
    if not expected_answer:
        logger.debug("No expected answer provided, cannot verify")
        return 0.0
    
    # Verify solution (will raise ImportError if verifiers not installed)
    reward = verify_math_solution(solution, expected_answer)
    
    return reward


