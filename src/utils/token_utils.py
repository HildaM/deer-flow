# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import logging
import tiktoken
from typing import List, Union, Dict, Any

from src.config import load_yaml_config

logger = logging.getLogger(__name__)


def get_max_token_limit() -> int:
    """Get MAX_TOKEN_LIMIT from configuration file."""
    try:
        config = load_yaml_config("conf.yaml")
        max_token_limit = config.get("MAX_TOKEN_LIMIT", 100000)
        if isinstance(max_token_limit, str):
            try:
                return int(max_token_limit)
            except ValueError:
                logger.warning(f"Invalid MAX_TOKEN_LIMIT value: {max_token_limit}, using default 100000")
                return 100000
        return max_token_limit
    except Exception as e:
        logger.warning(f"Failed to load MAX_TOKEN_LIMIT from conf.yaml: {e}, using default 100000")
        return 100000


def estimate_tokens(text: str) -> int:
    """
    Estimate the number of tokens in a text string.
    Uses a simple heuristic: 1 token ≈ 4 characters for most languages.
    This is an approximation and may not be 100% accurate.
    
    Args:
        text: Input text string
        
    Returns:
        Estimated number of tokens
    """
    if not text:
        return 0
    return len(text) // 4


def truncate_text_by_tokens(text: str, max_tokens: int) -> str:
    """
    Truncate text to fit within the specified token limit.
    
    Args:
        text: Input text to truncate
        max_tokens: Maximum number of tokens allowed
        
    Returns:
        Truncated text that fits within the token limit
    """
    if not text or max_tokens <= 0:
        return ""
    
    estimated_tokens = estimate_tokens(text)
    if estimated_tokens <= max_tokens:
        return text
    
    # Calculate approximate character limit
    char_limit = max_tokens * 4
    
    # Truncate and add indication
    if len(text) > char_limit:
        truncated = text[:char_limit]
        # Try to truncate at word boundary
        last_space = truncated.rfind(' ')
        if last_space > char_limit * 0.8:  # Only if we don't lose too much content
            truncated = truncated[:last_space]
        
        truncated += "\n\n[内容因token限制被截断...]"  
        logger.warning(f"Text truncated from {estimated_tokens} to ~{estimate_tokens(truncated)} tokens")
        return truncated
    
    return text


def truncate_search_results(search_results: Union[str, List[dict]], max_tokens: int = None) -> str:
    """
    Truncate search results to fit within token limit.
    
    Args:
        search_results: Search results (string or list of dicts)
        max_tokens: Maximum number of tokens allowed (defaults to MAX_TOKEN_LIMIT from config)
        
    Returns:
        Truncated search results as string
    """
    if max_tokens is None:
        max_tokens = get_max_token_limit()
    if not search_results:
        return ""
    
    # Convert to string if it's a list
    if isinstance(search_results, list):
        # Process each result and combine
        combined_results = []
        current_tokens = 0
        
        for result in search_results:
            if isinstance(result, dict):
                title = result.get('title', '')
                content = result.get('content', '')
                result_text = f"## {title}\n\n{content}"
            else:
                result_text = str(result)
            
            result_tokens = estimate_tokens(result_text)
            
            # Check if adding this result would exceed the limit
            if current_tokens + result_tokens > max_tokens:
                # Try to add a truncated version of this result
                remaining_tokens = max_tokens - current_tokens
                if remaining_tokens > 100:  # Only if we have reasonable space left
                    truncated_result = truncate_text_by_tokens(result_text, remaining_tokens)
                    combined_results.append(truncated_result)
                break
            
            combined_results.append(result_text)
            current_tokens += result_tokens
        
        final_text = "\n\n".join(combined_results)
    else:
        final_text = str(search_results)
    
    # Final truncation to ensure we're within limits
    return truncate_text_by_tokens(final_text, max_tokens)