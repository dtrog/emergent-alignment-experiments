"""
Utility functions for the emergent alignment experiments.

This module contains helper functions for:
- Safe logging configuration
- Conversation history pruning
- Retry logic with exponential backoff
- Response cleaning and JSON file loading
"""

import json
import time
import random
from typing import Dict, List, Any, Optional
from pathlib import Path
from functools import wraps
from openai import RateLimitError
from colorama import Fore
from tqdm import tqdm
import logging


def get_safe_config_for_logging(config: Any) -> Any:
    """
    Sanitize configuration for logging by redacting sensitive information.
    
    Args:
        config: Configuration object to sanitize
        
    Returns:
        Sanitized configuration safe for logging
    """
    if isinstance(config, dict):
        safe_config = {}
        for key, value in config.items():
            if any(sensitive in key.lower() for sensitive in ['key', 'secret', 'token', 'password']):
                safe_config[key] = "REDACTED"
            else:
                safe_config[key] = get_safe_config_for_logging(value)
        return safe_config
    elif isinstance(config, list):
        return [get_safe_config_for_logging(item) for item in config]
    else:
        return config


def prune_conversation_history(history: List[Dict[str, str]], model_max_tokens: int = 4000) -> List[Dict[str, str]]:
    """
    Prune conversation history to fit within model token limits.
    
    Args:
        history: List of message dictionaries
        model_max_tokens: Maximum tokens allowed for the model
        
    Returns:
        Pruned conversation history
    """
    # Rough estimation: 1 token ≈ 3.5 characters
    chars_per_token = 3.5
    max_chars = int(model_max_tokens * 0.8 * chars_per_token)  # Leave 20% buffer
    
    # Always keep system message if present
    if history and history[0].get('role') == 'system':
        pruned = [history[0]]
        remaining_history = history[1:]
    else:
        pruned = []
        remaining_history = history
    
    # Calculate current character count
    current_chars = sum(len(msg.get('content', '')) for msg in pruned)
    
    # Add messages from the end, working backwards
    for msg in reversed(remaining_history):
        msg_chars = len(msg.get('content', ''))
        if current_chars + msg_chars <= max_chars:
            pruned.insert(-len(pruned) if pruned and pruned[0].get('role') == 'system' else 0, msg)
            current_chars += msg_chars
        else:
            break
    
    return pruned


def retry_on_overload(max_attempts=5, base_delay=1.0, backoff=2.0):
    """
    A decorator that retries a function upon encountering RateLimitError or overloaded API errors.

    Args:
        max_attempts (int): Maximum number of retry attempts.
        base_delay (float): Initial delay between retries in seconds.
        backoff (float): Multiplier for exponential backoff.

    Returns:
        callable: The decorator function.
    """
    def decorator(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            """
            Wrapper function to handle retries for API calls with exponential backoff.

            Args:
                *args: Positional arguments to pass to the wrapped function.
                **kwargs: Keyword arguments to pass to the wrapped function.

            Returns:
                The result of the wrapped function if successful.

            Raises:
                RateLimitError: If the maximum number of retry attempts due to rate limiting is exceeded.
                Exception: Re-raises any other exception unless it is an "overloaded" error and retries remain.
            """
            delay = base_delay
            arm_name = getattr(self, 'name', "Unknown Actor")
            for attempt in range(1, max_attempts + 1):
                try:
                    return fn(self, *args, **kwargs)
                except RateLimitError:
                    if attempt == max_attempts: 
                        raise
                    tqdm.write(Fore.YELLOW + f"[{arm_name}] Rate limit hit. Retrying in {delay:.1f}s...")
                    time.sleep(delay)
                    delay *= backoff
                except Exception as e:
                    tqdm.write(Fore.RED + f"[{arm_name}] API Error: {e}")
                    if "overloaded" in str(e).lower() and attempt < max_attempts:
                        tqdm.write(Fore.YELLOW + f"[{arm_name}] Overloaded error. Retrying in {delay:.1f}s...")
                        time.sleep(delay)
                        delay *= backoff
                    else:
                        raise
        return wrapper
    return decorator


def strip_and_clean_response(response: str) -> str:
    """
    Clean and strip response text, removing JSON markers and extra whitespace.
    
    Args:
        response: Raw response string
        
    Returns:
        Cleaned response string
    """
    # Remove common JSON code block markers
    response = response.strip()
    
    # Remove ```json and ``` markers
    if response.startswith('```json'):
        response = response[7:]  # Remove ```json
    if response.startswith('```'):
        response = response[3:]   # Remove ```
    if response.endswith('```'):
        response = response[:-3]  # Remove ```
    
    return response.strip()


def load_json_file(filepath: str) -> Any:
    """
    Load and parse a JSON file.
    
    Args:
        filepath: Path to the JSON file
        
    Returns:
        Parsed JSON data
        
    Raises:
        Exception: If file cannot be read or parsed
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, IOError) as e:
        raise Exception(f"Failed to load JSON file {filepath}: {str(e)}")
