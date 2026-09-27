"""
Debug Qwen 3 4B response structure.
Run: python debug_llm2.py
"""
import ollama
import json

def inspect_response():
    """Inspect the full response object structure."""
    print("=" * 60)
    print("TEST: Full response inspection")
    print("=" * 60)
    
    response = ollama.chat(
        model='qwen3:4b',
        messages=[
            {'role': 'user', 'content': 'What is 2+2? Reply with just the number.'}
        ],
        options={'num_predict': 100}
    )
    
    print(f"Type: {type(response)}")
    print(f"Keys: {response.keys() if hasattr(response, 'keys') else 'N/A'}")
    print(f"\nFull response:")
    print(json.dumps(response, indent=2, default=str))
    
    # Check message field
    message = response.get('message', {})
    print(f"\nMessage keys: {message.keys() if hasattr(message, 'keys') else 'N/A'}")
    print(f"Content: {repr(message.get('content', ''))}")
    print(f"Thinking: {repr(message.get('thinking', ''))[:500] if message.get('thinking') else 'None'}")


def test_no_think():
    """Test with /no_think tag to disable thinking mode."""
    print("\n" + "=" * 60)
    print("TEST: /no_think tag")
    print("=" * 60)
    
    response = ollama.chat(
        model='qwen3:4b',
        messages=[
            {'role': 'user', 'content': 'What is 2+2? Reply with just the number. /no_think'}
        ],
        options={'num_predict': 50}
    )
    
    content = response['message']['content']
    thinking = response['message'].get('thinking', '')
    
    print(f"Content: {repr(content)}")
    print(f"Thinking: {repr(thinking[:200]) if thinking else 'None'}")


def test_think_parameter():
    """Test with think=False parameter if supported."""
    print("\n" + "=" * 60)
    print("TEST: think parameter")
    print("=" * 60)
    
    try:
        response = ollama.chat(
            model='qwen3:4b',
            messages=[
                {'role': 'user', 'content': 'What is 2+2? Reply with just the number.'}
            ],
            think=False,
            options={'num_predict': 50}
        )
        
        content = response['message']['content']
        print(f"Content: {repr(content)}")
        
    except Exception as e:
        print(f"think parameter not supported: {e}")
        
        # Try with options
        try:
            response = ollama.chat(
                model='qwen3:4b',
                messages=[
                    {'role': 'user', 'content': 'What is 2+2? Reply with just the number.'}
                ],
                options={'num_predict': 50, 'think': False}
            )
            content = response['message']['content']
            print(f"Content (via options): {repr(content)}")
        except Exception as e2:
            print(f"options think also failed: {e2}")


def test_generate():
    """Test using generate API instead of chat."""
    print("\n" + "=" * 60)
    print("TEST: Generate API")
    print("=" * 60)
    
    try:
        response = ollama.generate(
            model='qwen3:4b',
            prompt='What is 2+2? Reply with just the number.',
            options={'num_predict': 50}
        )
        
        print(f"Response: {response}")
        print(f"Response text: {repr(response.get('response', ''))}")
        
    except Exception as e:
        print(f"Generate failed: {e}")


def test_different_model():
    """Test if another model works to isolate the issue."""
    print("\n" + "=" * 60)
    print("TEST: Check available models")
    print("=" * 60)
    
    try:
        models = ollama.list()
        print("Available models:")
        
        if hasattr(models, 'models'):
            model_list = models.models
        elif isinstance(models, dict):
            model_list = models.get('models', [])
        else:
            model_list = models
        
        for m in model_list:
            name = m.get('model', m.get('name', 'unknown')) if isinstance(m, dict) else str(m)
            print(f"  - {name}")
        
    except Exception as e:
        print(f"Error listing models: {e}")


if __name__ == "__main__":
    inspect_response()
    test_no_think()
    test_think_parameter()
    test_generate()
    test_different_model()