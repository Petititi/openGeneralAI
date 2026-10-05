"""
JSON Parser Utility Module - Shared JSON parsing with fallback logic.

This module provides a centralized JSON parsing function used across the codebase
to avoid duplication of JSON parsing logic with similar fallback patterns.

Responsibility: Pure JSON parsing utility functions.
No LLM, storage, or business logic concerns.
"""

import json
import re
from typing import Any, Optional, Tuple, Union

# Optional import for advanced JSON parsing
try:
    import json5
    JSON5_AVAILABLE = True
except ImportError:
    JSON5_AVAILABLE = False


class JSONParseError(Exception):
    """Custom exception for JSON parsing errors."""
    def __init__(self, message: str, original_text: str = ""):
        super().__init__(message)
        self.original_text = original_text


def parse_json_safe(
    text: str,
    require_dict: bool = False,
    allow_json5: bool = True,
    fix_unicode: bool = True
) -> Union[dict, list, Any]:
    """
    Parse JSON text with multiple fallback strategies.
    
    This function provides a robust JSON parsing approach with:
    - Primary: Standard json.loads()
    - Fallback 1: json5.loads() if available
    - Fallback 2: Unicode escape fix + retry
    - Fallback 3: Extract JSON object from text using regex
    
    Args:
        text: The text to parse as JSON
        require_dict: If True, raises error if parsed result is not a dict
        allow_json5: If True, tries json5 parsing as fallback
        fix_unicode: If True, attempts to fix unicode escape issues
        
    Returns:
        Parsed JSON (typically dict, but can be any JSON type)
        
    Raises:
        JSONParseError: If all parsing strategies fail
    """
    if not text or not text.strip():
        raise JSONParseError("Empty text provided", text)
    
    # Strategy 1: Standard JSON parsing
    try:
        result = json.loads(text)
        if require_dict and not isinstance(result, dict):
            raise JSONParseError(f"Expected dict but got {type(result).__name__}", text)
        return result
    except (json.JSONDecodeError, ValueError):
        pass
    
    # Strategy 2: json5 parsing (more lenient)
    if allow_json5 and JSON5_AVAILABLE:
        try:
            result = json5.loads(text)
            if require_dict and not isinstance(result, dict):
                raise JSONParseError(f"Expected dict but got {type(result).__name__}", text)
            return result
        except (json5.JSONDecodeError, ValueError):
            pass
    
    # Strategy 3: Fix unicode escape issues
    if fix_unicode:
        try:
            fixed_text = text.encode("utf-8").decode("unicode_escape")
            result = json.loads(fixed_text)
            if require_dict and not isinstance(result, dict):
                raise JSONParseError(f"Expected dict but got {type(result).__name__}", text)
            return result
        except (json.JSONDecodeError, ValueError, UnicodeDecodeError):
            pass
    
    # Strategy 4: Extract JSON object from text using regex
    try:
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            if require_dict and not isinstance(result, dict):
                raise JSONParseError(f"Expected dict but got {type(result).__name__}", text)
            return result
    except (json.JSONDecodeError, ValueError, AttributeError):
        pass
    
    # All strategies failed
    raise JSONParseError("Failed to parse JSON with any known strategy", text)


def parse_json_with_fallback(
    text: str,
    default: Optional[Any] = None,
    require_dict: bool = False
) -> Tuple[bool, Any]:
    """
    Parse JSON text with fallback to default value on failure.
    
    This is a convenience wrapper around parse_json_safe that returns
    a tuple indicating success and the parsed result or default.
    
    Args:
        text: The text to parse as JSON
        default: Value to return if parsing fails (default: None)
        require_dict: If True, raises error if parsed result is not a dict
        
    Returns:
        Tuple of (success: bool, result: Any)
    """
    try:
        result = parse_json_safe(text, require_dict=require_dict)
        return True, result
    except JSONParseError:
        return False, default


def extract_json_from_markdown(text: str) -> Optional[dict]:
    """
    Extract JSON from markdown-formatted text (e.g., ```json ... ```).
    
    Args:
        text: The text that may contain markdown-formatted JSON
        
    Returns:
        Parsed JSON dict if found, None otherwise
    """
    # Remove markdown code blocks
    cleaned = text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    
    cleaned = cleaned.strip()
    
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    
    # Try with json5 if available
    if JSON5_AVAILABLE:
        try:
            return json5.loads(cleaned)
        except (json5.JSONDecodeError, ValueError):
            pass
    
    return None


def clean_json_response(raw_response: str) -> str:
    """
    Clean a raw LLM response to extract clean JSON text.
    
    Handles common formatting issues:
    - Markdown code blocks (```json ... ```)
    - Plain markdown code blocks (``` ... ```)
    - Leading/trailing whitespace
    
    Args:
        raw_response: The raw response from LLM
        
    Returns:
        Cleaned JSON text
    """
    if raw_response.startswith("```json"):
        raw_response = raw_response[7:-3]
    elif raw_response.startswith("```"):
        raw_response = raw_response[3:-3]
    
    return raw_response.strip()
