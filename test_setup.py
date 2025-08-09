# test_setup.py - Test if everything is installed correctly

import sys
import importlib

def test_imports():
    """Test if all required packages are installed"""
    
    packages = {
        "Core": ["together", "requests", "dotenv"],
        "Web UI": ["streamlit", "pandas", "plotly"],
        "API": ["fastapi", "uvicorn", "pydantic"],
        "Web Scraping": ["bs4", "httpx", "playwright"],
        "Async": ["aiohttp", "aiofiles"],
        "Utils": ["loguru", "psutil", "rich"]
    }
    
    print("🧪 Testing EdgeMind Dependencies")
    print("=" * 40)
    
    all_good = True
    
    for category, modules in packages.items():
        print(f"\n{category}:")
        for module in modules:
            try:
                if module == "dotenv":
                    importlib.import_module("dotenv")
                elif module == "bs4":
                    importlib.import_module("bs4")
                else:
                    importlib.import_module(module)
                print(f"  ✅ {module}")
            except ImportError:
                print(f"  ❌ {module} - Not installed")
                all_good = False
    
    print("\n" + "=" * 40)
    
    if all_good:
        print("✅ All dependencies installed successfully!")
    else:
        print("⚠️ Some dependencies are missing.")
        print("Run: pip install -r requirements_core.txt")
    
    return all_good

def test_api_connection():
    """Test if API is running"""
    try:
        import requests
        response = requests.get("http://localhost:8000/health", timeout=2)
        if response.status_code == 200:
            print("✅ API is running at http://localhost:8000")
            return True
    except:
        print("⚠️ API is not running. Start with: .\\run.ps1 api")
        return False

def test_redis_connection():
    """Test if Redis is available"""
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379)
        r.ping()
        print("✅ Redis is connected")
        return True
    except:
        print("ℹ️ Redis not available (optional for basic functionality)")
        return False

if __name__ == "__main__":
    print("\n🚀 EdgeMind Platform - System Check\n")
    
    # Test imports
    imports_ok = test_imports()
    
    print("\n🔌 Testing Services:")
    print("-" * 40)
    
    # Test API
    api_ok = test_api_connection()
    
    # Test Redis
    redis_ok = test_redis_connection()
    
    print("\n" + "=" * 40)
    print("📊 Summary:")
    print(f"  Dependencies: {'✅' if imports_ok else '❌'}")
    print(f"  API Server: {'✅' if api_ok else '❌'}")
    print(f"  Redis Cache: {'✅' if redis_ok else 'ℹ️ Optional'}")
    
    if imports_ok:
        print("\n✨ Your EdgeMind platform is ready to run!")
        print("\nStart with: .\\run.ps1 both")
    else:
        print("\n⚠️ Please install missing dependencies first")