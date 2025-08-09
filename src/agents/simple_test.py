# src/agents/simple_test.py
"""
Simple test to verify autonomous research components work
"""
import sys
from pathlib import Path
import time
import asyncio

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Test imports
def test_imports():
    """Test that all required packages import correctly"""
    print("🧪 Testing imports...")
    
    try:
        import selenium
        print("✅ Selenium imported")
    except ImportError as e:
        print(f"❌ Selenium failed: {e}")
        return False
    
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        print("✅ WebDriver imported")
    except ImportError as e:
        print(f"❌ WebDriver failed: {e}")
        return False
    
    try:
        import docker
        print("✅ Docker imported")
    except ImportError as e:
        print(f"❌ Docker failed: {e}")
        return False
    
    try:
        import requests
        import pandas as pd
        import plotly.express as px
        print("✅ Data tools imported")
    except ImportError as e:
        print(f"❌ Data tools failed: {e}")
        return False
    
    try:
        from src.core.working_ai_playground import AIPlayground
        print("✅ AI Playground imported")
    except ImportError as e:
        print(f"❌ AI Playground failed: {e}")
        print("Make sure you have the working_ai_playground.py file")
        return False
    
    return True

def test_ai_connection():
    """Test AI API connection"""
    print("\n🤖 Testing AI connection...")
    
    try:
        from src.core.working_ai_playground import AIPlayground
        ai = AIPlayground()
        
        response = ai.chat("Hello! Just testing the connection.")
        if response and "❌" not in response:
            print("✅ AI connection working")
            print(f"Response: {response[:100]}...")
            return True
        else:
            print(f"❌ AI connection issue: {response}")
            return False
            
    except Exception as e:
        print(f"❌ AI connection failed: {e}")
        return False

def test_browser_setup():
    """Test if browser automation works"""
    print("\n🌐 Testing browser setup...")
    
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from webdriver_manager.chrome import ChromeDriverManager
        from selenium.webdriver.chrome.service import Service
        
        # Set up Chrome options
        chrome_options = Options()
        chrome_options.add_argument("--headless")  # Run headless for testing
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        
        # Try to create driver
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        
        # Test navigation
        driver.get("https://www.google.com")
        title = driver.title
        driver.quit()
        
        if "Google" in title:
            print("✅ Browser automation working")
            return True
        else:
            print(f"❌ Browser test failed, got title: {title}")
            return False
            
    except Exception as e:
        print(f"❌ Browser setup failed: {e}")
        print("💡 Try installing Chrome or updating ChromeDriver")
        return False

def test_docker_connection():
    """Test Docker connection (optional)"""
    print("\n🐳 Testing Docker connection...")
    
    try:
        import docker
        client = docker.from_env()
        
        # Try to get Docker info
        info = client.info()
        print(f"✅ Docker connected - Version: {info.get('ServerVersion', 'Unknown')}")
        return True
        
    except Exception as e:
        print(f"⚠️ Docker not available: {e}")
        print("💡 Docker is optional for basic web research")
        return False

async def test_basic_research():
    """Test a simple research task"""
    print("\n🔬 Testing basic research capability...")
    
    try:
        # Simple web search simulation
        import requests
        from bs4 import BeautifulSoup
        
        # Test Google search (simple version)
        search_url = "https://www.google.com/search?q=python+programming"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(search_url, headers=headers, timeout=10)
        if response.status_code == 200:
            print("✅ Basic web requests working")
        else:
            print(f"⚠️ Web request returned {response.status_code}")
        
        # Test AI analysis
        from src.core.working_ai_playground import AIPlayground
        ai = AIPlayground()
        
        analysis = ai.business_advisor("Analyze the current trends in AI development for 2025")
        if analysis and len(analysis) > 50:
            print("✅ AI analysis capability working")
            return True
        else:
            print("❌ AI analysis failed")
            return False
            
    except Exception as e:
        print(f"❌ Basic research test failed: {e}")
        return False

async def main():
    """Run all tests"""
    print("🚀 AUTONOMOUS RESEARCH SYSTEM TEST")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("AI Connection", test_ai_connection),
        ("Browser Setup", test_browser_setup),
        ("Docker Connection", test_docker_connection),
        ("Basic Research", test_basic_research)
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name}...")
        if asyncio.iscoroutinefunction(test_func):
            results[test_name] = await test_func()
        else:
            results[test_name] = test_func()
        time.sleep(1)  # Brief pause between tests
    
    # Summary
    print("\n" + "=" * 50)
    print("🎯 TEST RESULTS SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20} {status}")
        if result:
            passed += 1
    
    print(f"\nScore: {passed}/{total} tests passed")
    
    if passed >= 3:  # At least basic functionality
        print("\n🎉 SYSTEM READY FOR AUTONOMOUS RESEARCH!")
        print("💡 Next step: Run the web interface")
        print("   streamlit run web/agent_control_app.py")
    else:
        print("\n⚠️ SYSTEM NEEDS SETUP")
        print("💡 Fix the failed tests before proceeding")
    
    return results

if __name__ == "__main__":
    asyncio.run(main())