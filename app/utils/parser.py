# app/utils/parser.py
"""
Parsing utilities for extracting structured data from model responses
"""

import json
import re
from typing import Any, Optional, Dict
import logging

logger = logging.getLogger(__name__)

def extract_json_from_response(response: str) -> Optional[Dict]:
    """
    Extract JSON from model response robustly.
    Prioritizes full structure over regex fragments to handle nested JSON.
    """
    if not response:
        return None

    # 1. Fast Path: Try parsing the cleaned string directly
    # This is what fixes your specific example (which is valid JSON just with whitespace)
    cleaned_response = response.strip()
    try:
        return json.loads(cleaned_response)
    except json.JSONDecodeError:
        pass

    # 2. Markdown Block Strategy (Common for LLMs)
    # Use re.DOTALL to match across newlines inside the block
    patterns = [
        r'```json\s*(.*?)\s*```',
        r'```\s*(.*?)\s*```'
    ]
    for pattern in patterns:
        match = re.search(pattern, response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

    # 3. Substring Strategy (The "Nuclear Option")
    # Instead of regex (which struggles with nesting), find the first '{' and last '}'
    try:
        start_idx = response.find('{')
        end_idx = response.rfind('}')
        
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            possible_json = response[start_idx : end_idx + 1]
            return json.loads(possible_json)
    except json.JSONDecodeError:
        pass

    return None


def safe_parse_json(text: str, default: Any = None) -> Any:
    """
    Safely parse JSON with fallback
    
    Args:
        text: JSON string
        default: Default value if parsing fails
        
    Returns:
        Parsed JSON or default value
    """
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError, TypeError):
        return default

_YES_RE = re.compile(r"\b(yes|true)\b", re.IGNORECASE)
_NO_RE = re.compile(r"\b(no|false)\b", re.IGNORECASE)
_VALID_RE = re.compile(r"\bvalid\b", re.IGNORECASE)
_INVALID_RE = re.compile(r"\binvalid\b", re.IGNORECASE)


def extract_boolean_from_response(response: str) -> bool:
    """
    Extract a boolean verdict from a free-form model response.

    Uses word boundaries so "Yesterday we observed…" doesn't falsely match
    "yes", and "invalid" doesn't falsely match "valid". Yes/no takes
    precedence; valid/invalid is a secondary hint when neither appears.
    """
    if not response:
        return False

    yes = _YES_RE.search(response)
    no = _NO_RE.search(response)
    if yes and not no:
        return True
    if no and not yes:
        return False
    if yes and no:
        # Both appear — prefer whichever comes first (models often lead
        # with the verdict and then explain what wasn't the case).
        return yes.start() < no.start()

    if _INVALID_RE.search(response):
        return False
    if _VALID_RE.search(response):
        return True
    return False

def extract_key_value_pairs(response: str) -> Dict[str, str]:
    """
    Extract key-value pairs from structured text
    
    Args:
        response: Model response string
        
    Returns:
        Dictionary of extracted key-value pairs
    """
    
    pairs = {}
    
    # Pattern for "Key: Value" format
    pattern = r'(?:^|\n)\s*\**([A-Za-z][A-Za-z\s]+?):\**\s*(.+?)(?=\n|$)'
    
    matches = re.findall(pattern, response, re.MULTILINE)
    
    for key, value in matches:
        key = key.strip().strip('*').lower().replace(' ', '_')
        value = value.strip().strip('*')
        pairs[key] = value
    
    return pairs