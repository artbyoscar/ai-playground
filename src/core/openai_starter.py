# src/core/openai_starter.py
import os
import requests
from dotenv import load_dotenv

load_dotenv()

def test_openai_api():
    """Test OpenAI API - much more reliable than HuggingFace"""
    print("🤖 Testing OpenAI API...")
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ No OpenAI API key found")
        return False
    
    print(f"✅ API key found: {api_key[:20]}...")
    
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    
    payload = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "user", "content": "Hello! I'm building my AI playground. Say something encouraging!"}
        ],
        "max_tokens": 100
    }
    
    try:
        response = requests.post(
            'https://api.openai.com/v1/chat/completions',
            headers=headers,
            json=payload
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            message = result['choices'][0]['message']['content']
            print("🎉 SUCCESS!")
            print(f"AI Response: {message}")
            return True
        else:
            print(f"❌ Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False

def test_free_alternatives():
    """Test free APIs that don't require auth"""
    print("\n🆓 Testing Free Alternatives...")
    
    # Ollama Web UI (if running locally)
    try:
        response = requests.get('http://localhost:11434/api/tags', timeout=2)
        if response.status_code == 200:
            print("✅ Ollama running locally!")
            return True
    except:
        print("⚪ Ollama not running locally")
    
    # Alternative free APIs
    free_apis = [
        "https://api.deepinfra.com/v1/openai/chat/completions",  # Free tier
        "https://api.together.xyz/v1/chat/completions"           # Free tier
    ]
    
    for api in free_apis:
        try:
            # Test without auth first
            response = requests.get(api.replace('/chat/completions', '/models'), timeout=3)
            if response.status_code != 404:
                print(f"✅ {api} responding")
        except:
            print(f"⚪ {api} not available")
    
    return False

if __name__ == "__main__":
    print("🎮 AI Playground - Alternative API Test")
    print("=" * 50)
    
    # Test OpenAI first (most likely to work)
    if test_openai_api():
        print("\n🎉 You're ready to build! OpenAI API working.")
    else:
        print("\n🔄 Trying free alternatives...")
        test_free_alternatives()
        print("\n💡 Consider getting OpenAI credits or running local models")