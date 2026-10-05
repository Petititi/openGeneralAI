"""Parsing of the JSON answers of the LLM, which often come with markdown fences or small
syntax liberties (comments, trailing commas, single quotes)."""

import json
import re
from typing import Any

import json5


class JSONParseError(ValueError):
    """The text holds no JSON value we could read."""

    def __init__(self, message: str, original_text: str = ""):
        super().__init__(message)
        self.original_text = original_text


def strip_code_fences(text: str) -> str:
    """Remove a surrounding ```json ... ``` (or ``` ... ```) block."""
    text = text.strip()
    match = re.fullmatch(r"```[a-zA-Z0-9]*\s*\n?(.*?)\n?```", text, re.DOTALL)
    return match.group(1).strip() if match else text


def parse_json_safe(text: str, require_dict: bool = False) -> Any:
    """Parse JSON with fallbacks: strict JSON, then JSON5, then the first {...} block of the text.

    Raises JSONParseError when every strategy fails (or when require_dict and the value is
    not an object).
    """
    if not text or not text.strip():
        raise JSONParseError("Empty answer", text)

    candidates = [text]
    block = re.search(r"\{.*\}", text, re.DOTALL)
    if block and block.group() != text:
        candidates.append(block.group())

    for candidate in candidates:
        for loads in (json.loads, json5.loads):
            try:
                value = loads(candidate)
            except ValueError:
                continue
            if require_dict and not isinstance(value, dict):
                raise JSONParseError(f"Expected a JSON object, got {type(value).__name__}", text)
            return value
    raise JSONParseError("The answer is not valid JSON", text)


def parse_llm_json(text: str) -> dict:
    """The JSON object of an LLM answer, fences removed."""
    return parse_json_safe(strip_code_fences(text), require_dict=True)
