import re
import json
from typing import Dict, Any
from .schema import sanitize_annotation_dict, MISSING_LABEL

def _failed_parse_record(explanation: str, error: str) -> Dict[str, Any]:
    """Returns an explicit UNKNOWN annotation record when JSON extraction fails."""
    return {
        "cyberbullying": MISSING_LABEL,
        "type": MISSING_LABEL,
        "emotion": MISSING_LABEL,
        "cultural_context": MISSING_LABEL,
        "implicit_level": MISSING_LABEL,
        "severity": MISSING_LABEL,
        "confidence": None,
        "explanation": explanation,
        "error": error,
    }

def extract_json_from_response(output: str) -> Dict[str, Any]:
    """
    Robust JSON extraction from SumoPod LLM text generation.
    Handles:
    1. Markdown code blocks (```json ... ```)
    2. Unquoted numeric values (e.g. "confidence": 0.95)
    3. Trailing commas & syntax errors
    4. Truncated / unclosed JSON (auto-repair if model cut off before '}')
    5. Regex-based field extraction fallback
    """
    if not output or not isinstance(output, str):
        return _failed_parse_record("No output produced", "Empty model output")

    # Step 1: Strip markdown code blocks if present
    cleaned = re.sub(r'```(?:json)?\s*', '', output)
    cleaned = cleaned.replace('```', '').strip()

    # Step 2: Find outermost curly braces { ... }
    json_str = None
    match = re.search(r'(\{[\s\S]*\})', cleaned)
    if match:
        json_str = match.group(1).strip()
    elif "{" in cleaned:
        # Auto-repair truncated JSON: take from first '{' and append '}'
        start_idx = cleaned.find("{")
        snippet = cleaned[start_idx:].strip()
        # If open quote without closing quote, close quote first
        if snippet.count('"') % 2 != 0:
            snippet += '"'
        json_str = snippet + "\n}"

    if not json_str:
        return _failed_parse_record(
            f"Failed to locate JSON brackets. Raw snippet: {output[:150]}...",
            "No JSON brackets found",
        )

    # Step 3: Attempt direct parsing
    try:
        data = json.loads(json_str)
        if isinstance(data, dict):
            return sanitize_annotation_dict(data)
    except Exception:
        pass

    # Step 4: Fix common LLM JSON syntax mistakes (trailing commas, trailing quotes)
    try:
        fixed_json = re.sub(r',\s*\}', '}', json_str)
        fixed_json = re.sub(r',\s*\]', ']', fixed_json)
        data = json.loads(fixed_json)
        if isinstance(data, dict):
            return sanitize_annotation_dict(data)
    except Exception:
        pass

    # Step 5: Regex field-by-field extraction fallback
    # Supports both quoted string values and unquoted numeric/boolean values
    fallback = {}
    fields = [
        "cyberbullying", "type", "emotion", "cultural_context",
        "implicit_level", "severity", "confidence", "explanation"
    ]
    for f in fields:
        # Regex matches either "field": "value" or "field": 0.95 / true / etc.
        pattern = rf'"{f}"\s*:\s*(?:"([^"]*)"|([^,\s\}}\n]+))'
        field_match = re.search(pattern, json_str, re.IGNORECASE)
        if field_match:
            val = field_match.group(1) if field_match.group(1) is not None else field_match.group(2)
            fallback[f] = val.strip().strip('"')
        else:
            fallback[f] = ""

    # If key fields are extracted, sanitize and return
    if any(fallback.get(k) for k in ["cyberbullying", "type", "confidence", "explanation"]):
        sanitized = sanitize_annotation_dict(fallback)
        return sanitized

    return _failed_parse_record(
        f"JSON syntax error. Model response: {output[:200]}",
        "Malformed JSON",
    )


# Backward-compatible alias for legacy utilities
extract_json_from_llama = extract_json_from_response
