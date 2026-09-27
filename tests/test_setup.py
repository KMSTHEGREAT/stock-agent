"""
Quick test to verify the setup is working.
Run: python test_setup.py
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def test_imports():
    """Test all critical imports."""
    print("Testing imports...")
    
    try:
        import yfinance
        print(f"  ✓ yfinance {yfinance.__version__}")
    except ImportError:
        print("  ✗ yfinance not installed")
        return False
    
    try:
        import pandas
        print(f"  ✓ pandas {pandas.__version__}")
    except ImportError:
        print("  ✗ pandas not installed")
        return False
    
    try:
        import pandas_ta
        print(f"  ✓ pandas-ta")
    except ImportError:
        print("  ✗ pandas-ta not installed")
        return False
    
    try:
        import ollama
        print(f"  ✓ ollama")
    except ImportError:
        print("  ✗ ollama not installed")
        return False
    
    try:
        import pydantic
        print(f"  ✓ pydantic {pydantic.__version__}")
    except ImportError:
        print("  ✗ pydantic not installed")
        return False
    
    return True


def test_ollama():
    """Test Ollama connection and model availability."""
    print("\nTesting Ollama...")
    
    try:
        import ollama
        
        # List models - handle different API response formats
        try:
            models_response = ollama.list()
            
            # Handle different response structures
            if hasattr(models_response, 'models'):
                model_list = models_response.models
            elif isinstance(models_response, dict) and 'models' in models_response:
                model_list = models_response['models']
            else:
                model_list = models_response
            
            model_names = []
            for m in model_list:
                # Handle different attribute/key names
                if hasattr(m, 'model'):
                    model_names.append(m.model)
                elif hasattr(m, 'name'):
                    model_names.append(m.name)
                elif isinstance(m, dict):
                    model_names.append(m.get('model', m.get('name', str(m))))
                else:
                    model_names.append(str(m))
            
            print(f"  Available models: {model_names}")
            
            # Check for qwen3:4b (name might be "qwen3:4b" or "qwen3:4b-instruct" etc.)
            has_qwen = any('qwen3' in name.lower() and '4b' in name.lower() 
                          for name in model_names)
            
            if has_qwen:
                print("  ✓ Qwen 3 4B is available")
                
                # Test actual inference
                test_response = ollama.chat(
                    model='qwen3:4b',
                    messages=[{'role': 'user', 'content': 'Say "OK" if you can read this.'}],
                    options={'num_predict': 10}
                )
                content = test_response['message']['content']
                print(f"  ✓ Model responds: {content[:50]}")
                return True
            else:
                print("  ✗ Qwen 3 4B not found")
                print(f"  Run: ollama pull qwen3:4b")
                return False
                
        except Exception as e:
            print(f"  ✗ Ollama list error: {e}")
            # Try direct connection test
            try:
                import requests
                response = requests.get("http://localhost:11434/api/tags", timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    models = data.get('models', [])
                    model_names = [m.get('name', m.get('model', '')) for m in models]
                    print(f"  ✓ Ollama is running. Models: {model_names}")
                    
                    has_qwen = any('qwen3' in name.lower() and '4b' in name.lower() 
                                  for name in model_names)
                    if has_qwen:
                        print("  ✓ Qwen 3 4B is available")
                        return True
                    else:
                        print("  ✗ Qwen 3 4B not found")
                        return False
                else:
                    print(f"  ✗ Ollama HTTP status: {response.status_code}")
                    return False
            except Exception as e2:
                print(f"  ✗ Cannot connect to Ollama: {e2}")
                return False
            
    except ImportError:
        print("  ✗ ollama not installed")
        return False
    except Exception as e:
        print(f"  ✗ Ollama error: {e}")
        return False


def test_database():
    """Test database and cache."""
    print("\nTesting database...")
    
    try:
        from src.data.cache import DataCache
        from src.data.symbol_mapper import SymbolMapper
        
        # Test cache
        cache = DataCache()
        cache.set("test_key", {"data": "test"}, ttl_hours=1)
        result = cache.get("test_key")
        
        if result == {"data": "test"}:
            print("  ✓ Cache working")
        else:
            print("  ✗ Cache failed")
            return False
        
        # Test symbol mapper
        mapper = SymbolMapper()
        info = mapper.validate_symbol("RELIANCE")
        
        if info["valid"] and info["nse_symbol"] == "RELIANCE":
            print("  ✓ Symbol mapper working")
        else:
            print("  ✗ Symbol mapper failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"  ✗ Database error: {e}")
        return False


def test_yfinance():
    """Test yfinance with Indian stock."""
    print("\nTesting yfinance with NSE data...")
    
    try:
        import yfinance as yf
        
        # Test with RELIANCE
        ticker = yf.Ticker("RELIANCE.NS")
        hist = ticker.history(period="5d")
        
        if not hist.empty and len(hist) > 0:
            latest_close = hist["Close"].iloc[-1]
            print(f"  ✓ RELIANCE.NS last close: ₹{latest_close:.2f}")
            return True
        else:
            print("  ✗ No data returned for RELIANCE.NS")
            return False
            
    except Exception as e:
        print(f"  ✗ yfinance error: {e}")
        return False


def test_price_fetcher():
    """Test the price fetcher module."""
    print("\nTesting PriceFetcher...")
    
    try:
        from src.data.price_fetcher import PriceFetcher
        
        fetcher = PriceFetcher()
        
        # Test current price
        price = fetcher.get_current_price("RELIANCE")
        if price:
            print(f"  ✓ Current price for RELIANCE: ₹{price:.2f}")
        else:
            print("  ✗ Failed to get current price")
            return False
        
        # Test historical data
        df = fetcher.get_historical_data("TCS", period="1mo")
        if df is not None and not df.empty:
            print(f"  ✓ Historical data for TCS: {len(df)} rows")
        else:
            print("  ✗ Failed to get historical data")
            return False
        
        return True
        
    except Exception as e:
        print(f"  ✗ PriceFetcher error: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 50)
    print("Stock Agent Setup Verification")
    print("=" * 50)
    
    all_passed = True
    
    all_passed &= test_imports()
    all_passed &= test_ollama()
    all_passed &= test_database()
    all_passed &= test_yfinance()
    all_passed &= test_price_fetcher()
    
    print("\n" + "=" * 50)
    if all_passed:
        print("✓ ALL TESTS PASSED - Ready to proceed!")
    else:
        print("✗ SOME TESTS FAILED - Fix issues above")
    print("=" * 50)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
