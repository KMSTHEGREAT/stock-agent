"""
Debug script to understand what Qwen 3 4B actually returns.
Run: python debug_llm.py
"""
import ollama
import json
import re

def test_raw_response():
    """See the raw response without any format constraint."""
    print("=" * 60)
    print("TEST 1: Raw response (no format)")
    print("=" * 60)
    
    response = ollama.chat(
        model='qwen3:4b',
        messages=[
            {'role': 'user', 'content': 'What is 2+2? Reply with just the number.'}
        ],
        options={'num_predict': 50, 'temperature': 0.1}
    )
    
    content = response['message']['content']
    print(f"Content: {repr(content)}")
    print(f"Length: {len(content)}")
    print()


def test_format_response():
    """Test with format parameter for structured output."""
    print("=" * 60)
    print("TEST 2: Response with format parameter")
    print("=" * 60)
    
    schema = {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "number": {"type": "number"}
        },
        "required": ["answer", "number"]
    }
    
    response = ollama.chat(
        model='qwen3:4b',
        messages=[
            {'role': 'user', 'content': 'What is 2+2? Give me the answer as a word and as a number.'}
        ],
        format=schema,
        options={'num_predict': 100, 'temperature': 0.1}
    )
    
    content = response['message']['content']
    print(f"Content: {repr(content)}")
    print(f"Length: {len(content)}")
    
    # Try to parse
    try:
        parsed = json.loads(content)
        print(f"✓ Direct JSON parse: {parsed}")
    except json.JSONDecodeError as e:
        print(f"✗ Direct parse failed: {e}")
        
        # Try cleaning
        cleaned = re.sub(r'output.*?output', '', content, flags=re.DOTALL).strip()
        print(f"After cleaning thinking tags: {repr(cleaned[:200])}")
        
        try:
            parsed = json.loads(cleaned)
            print(f"✓ Cleaned JSON parse: {parsed}")
        except:
            print("✗ Cleaned parse also failed")
    print()


def test_long_analysis():
    """Test with a longer, more realistic prompt."""
    print("=" * 60)
    print("TEST 3: Realistic analysis prompt")
    print("=" * 60)
    
    schema = {
        "type": "object",
        "properties": {
            "analysis_summary": {
                "type": "string",
                "description": "Brief analysis summary"
            },
            "sentiment": {
                "type": "string",
                "enum": ["positive", "negative", "neutral"]
            },
            "confidence": {
                "type": "number",
                "minimum": 0,
                "maximum": 1
            }
        },
        "required": ["analysis_summary", "sentiment", "confidence"]
    }
    
    prompt = """Analyze TCS (Tata Consultancy Services) stock:

Current Price: ₹3200
RSI: 32 (bearish)
Trend: Down
P/E Ratio: 25
Market Cap: Large Cap

Provide analysis in JSON format with analysis_summary, sentiment, and confidence."""
    
    response = ollama.chat(
        model='qwen3:4b',
        messages=[
            {'role': 'user', 'content': prompt}
        ],
        format=schema,
        options={'num_predict': 300, 'temperature': 0.3}
    )
    
    content = response['message']['content']
    print(f"Content (first 500 chars): {repr(content[:500])}")
    print(f"Total length: {len(content)}")
    
    # Check for thinking tags
    has_thinking = 'output' in content
    print(f"Contains thinking tags: {has_thinking}")
    
    # Try various extraction methods
    print("\n--- Extraction attempts ---")
    
    # Method 1: Direct parse
    try:
        parsed = json.loads(content)
        print(f"✓ Method 1 (direct): {json.dumps(parsed, indent=2)[:200]}")
        return
    except:
        print("✗ Method 1 (direct) failed")
    
    # Method 2: Remove thinking tags
    cleaned = re.sub(r'output.*?output', '', content, flags=re.DOTALL).strip()
    try:
        parsed = json.loads(cleaned)
        print(f"✓ Method 2 (remove thinking): {json.dumps(parsed, indent=2)[:200]}")
        return
    except:
        print(f"✗ Method 2 failed. Cleaned content: {repr(cleaned[:200])}")
    
    # Method 3: Find JSON in text
    json_match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            print(f"✓ Method 3 (regex simple): {json.dumps(parsed, indent=2)[:200]}")
            return
        except:
            pass
    
    # Method 4: Find JSON with nested objects
    json_match = re.search(r'\{.*\}', content, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            print(f"✓ Method 4 (regex greedy): {json.dumps(parsed, indent=2)[:200]}")
            return
        except:
            pass
    
    # Method 5: Try to fix common JSON issues
    # Remove trailing commas, fix quotes
    fixed = content
    fixed = re.sub(r',\s*}', '}', fixed)  # Remove trailing commas
    fixed = re.sub(r',\s*]', ']', fixed)
    
    try:
        parsed = json.loads(fixed)
        print(f"✓ Method 5 (fix JSON): {json.dumps(parsed, indent=2)[:200]}")
        return
    except:
        print("✗ All methods failed")
    
    # If nothing works, print the raw content for inspection
    print(f"\nRAW CONTENT FOR INSPECTION:")
    print(content)


if __name__ == "__main__":
    test_raw_response()
    test_format_response()
    test_long_analysis()