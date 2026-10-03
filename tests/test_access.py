"""
Quick test to find the correct way to access Ollama response.
Run: python test_access.py
"""
import ollama

response = ollama.chat(
    model='qwen3:4b',
    messages=[{'role': 'user', 'content': 'What is 2+2? Reply with just the number.'}],
    options={'num_predict': 50, 'temperature': 0.1}
)

print("=== Method 1: Dict access ===")
try:
    content = response['message']['content']
    print(f"  Result: {repr(content)}")
except Exception as e:
    print(f"  Error: {e}")

print("\n=== Method 2: .get() access ===")
try:
    message = response.get('message', {})
    content = message.get('content', '')
    print(f"  Result: {repr(content)}")
except Exception as e:
    print(f"  Error: {e}")

print("\n=== Method 3: Attribute access ===")
try:
    content = response.message.content
    print(f"  Result: {repr(content)}")
except Exception as e:
    print(f"  Error: {e}")

print("\n=== Method 4: Direct inspection ===")
print(f"  Type: {type(response)}")
print(f"  Has message attr: {hasattr(response, 'message')}")
if hasattr(response, 'message'):
    msg = response.message
    print(f"  Message type: {type(msg)}")
    print(f"  Has content attr: {hasattr(msg, 'content')}")
    if hasattr(msg, 'content'):
        print(f"  Content: {repr(msg.content)}")
    print(f"  Has thinking attr: {hasattr(msg, 'thinking')}")
    if hasattr(msg, 'thinking'):
        print(f"  Thinking: {repr(msg.thinking[:100]) if msg.thinking else 'None'}")

print("\n=== Method 5: Convert to dict ===")
try:
    if hasattr(response, 'model_dump'):
        d = response.model_dump()
        content = d['message']['content']
        print(f"  Result: {repr(content)}")
    elif hasattr(response, 'dict'):
        d = response.dict()
        content = d['message']['content']
        print(f"  Result: {repr(content)}")
    else:
        print("  No dict conversion method found")
except Exception as e:
    print(f"  Error: {e}")