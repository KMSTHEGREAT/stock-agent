"""
LLM helper for V2 agents.
Works with Phi-4-mini (content output) and Qwen3-4B (thinking output).
"""
import json
import re
import logging
import ollama
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# V2 config
MODEL = "phi4-mini"
BASE_URL = "http://localhost:11434"


def call_llm(prompt: str, system: str = None,
             temperature: float = 0.3,
             num_predict: int = 500) -> Optional[str]:
    """
    Call LLM and return the actual output.
    Handles both Phi-4-mini (content) and Qwen3-4B (thinking) response formats.
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    
    try:
        response = ollama.chat(
            model=MODEL,
            messages=messages,
            options={
                "temperature": temperature,
                "num_predict": num_predict,
            }
        )
        
        msg = response.message
        content = msg.content or ""
        thinking = msg.thinking or ""
        
        # Phi-4-mini: content has the answer
        if content.strip():
            return content
        
        # Qwen3-4B: answer is in thinking
        if thinking.strip():
            logger.debug(f"Content empty, using thinking ({len(thinking)} chars)")
            return thinking
        
        return None
        
    except Exception as e:
        logger.error(f"LLM call error: {e}")
        return None


def extract_json(text: str) -> Optional[Dict]:
    """Extract JSON from LLM response text."""
    if not text:
        return None
    
    # Clean
    text = text.strip()
    
    # Remove markdown code blocks
    code_match = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL)
    if code_match:
        text = code_match.group(1).strip()
    
    # Try direct parse
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass
    
    # Find JSON object boundaries
    start = text.find('{')
    if start == -1:
        return None
    
    depth = 0
    in_string = False
    escape = False
    
    for i in range(start, len(text)):
        char = text[i]
        
        if escape:
            escape = False
            continue
        if char == '\\':
            escape = True
            continue
        if char == '"':
            in_string = not in_string
        if not in_string:
            if char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0:
                    candidate = text[start:i+1]
                    try:
                        result = json.loads(candidate)
                        if isinstance(result, dict):
                            return result
                    except:
                        # Fix common issues
                        fixed = re.sub(r',\s*}', '}', candidate)
                        fixed = re.sub(r',\s*]', ']', fixed)
                        try:
                            result = json.loads(fixed)
                            if isinstance(result, dict):
                                return result
                        except:
                            pass
    
    return None


def call_llm_json(prompt: str, system: str = None,
                  temperature: float = 0.2,
                  num_predict: int = 500) -> Optional[Dict]:
    """Call LLM and try to extract JSON from response."""
    text = call_llm(prompt, system, temperature, num_predict)
    if text:
        return extract_json(text)
    return None


def safe_float(value, default: float = 0.0) -> float:
    """Safely convert to float."""
    try:
        if value is None:
            return default
        return float(value)
    except (ValueError, TypeError):
        return default


def safe_str(value, default: str = "") -> str:
    """Safely convert to string."""
    if value is None:
        return default
    return str(value)[:500]