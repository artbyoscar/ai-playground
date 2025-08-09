# src/core/api_debugger.py
import os
import requests
from dotenv import load_dotenv
import json

load_dotenv()

def debug_together():
    """Debug Together.ai with different models"""
    print("🔧 Debugging Together.ai...")
    
    api_key = os.getenv('TOGETHER_API_KEY')
    if not api_key:
        print("❌ No API key")
        return
    
    # Try free models first
    free_models = [
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "togethercomputer/RedPajama-INCITE-Chat-3B-v1",
        "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
        "microsoft/DialoGPT-medium"
    ]
    
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    
    for model in free_models:
        print(f"\n   Testing model: {model}")
        data = {
            "model": model,
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 50
        }
        
        try:
            response = requests.post(
                'https://api.together.xyz/v1/chat/completions',
                headers=headers,
                json=data,
                timeout=10
            )
            print(f"   Status: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ SUCCESS: {result['choices'][0]['message']['content'][:50]}...")
                return model
            else:
                print(f"   ❌ Error: {response.text[:100]}...")
        except Exception as e:
            print(f"   ❌ Exception: {e}")
    
    return None

def debug_grok():
    """Debug Grok/xAI API"""
    print("\n🔧 Debugging Grok (xAI)...")
    
    api_key = os.getenv('XAI_API_KEY')
    if not api_key:
        print("❌ No API key")
        return
    
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    
    # Try different models and endpoints
    test_configs = [
        {"model": "grok-4-latest", "url": "https://api.x.ai/v1/chat/completions"},
        {"model": "grok-beta", "url": "https://api.x.ai/v1/chat/completions"},
        {"model": "grok-4", "url": "https://api.x.ai/v1/chat/completions"}
    ]
    
    for config in test_configs:
        print(f"\n   Testing: {config['model']}")
        data = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello"}
            ],
            "model": config['model'],
            "stream": False,
            "temperature": 0
        }
        
        try:
            response = requests.post(config['url'], headers=headers, json=data, timeout=10)
            print(f"   Status: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ SUCCESS: {result['choices'][0]['message']['content'][:50]}...")
                return config
            else:
                print(f"   ❌ Error: {response.text[:200]}...")
        except Exception as e:
            print(f"   ❌ Exception: {e}")
    
    return None

def debug_deepseek():
    """Debug DeepSeek API"""
    print("\n🔧 Debugging DeepSeek...")
    
    api_key = os.getenv('DEEPSEEK_API_KEY')
    if not api_key:
        print("❌ No API key")
        return
    
    # Try different endpoints and models
    test_configs = [
        {"model": "deepseek-coder", "url": "https://api.deepseek.com/v1/chat/completions"},
        {"model": "deepseek-chat", "url": "https://api.deepseek.com/v1/chat/completions"},
        {"model": "deepseek-coder", "url": "https://api.deepseek.com/chat/completions"}
    ]
    
    for config in test_configs:
        print(f"\n   Testing: {config['model']} at {config['url']}")
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        data = {
            "model": config['model'],
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False
        }
        
        try:
            response = requests.post(config['url'], headers=headers, json=data, timeout=10)
            print(f"   Status: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ SUCCESS: {result['choices'][0]['message']['content'][:50]}...")
                return config
            else:
                print(f"   ❌ Error: {response.text[:200]}...")
        except Exception as e:
            print(f"   ❌ Exception: {e}")
    
    return None

def debug_openai():
    """Debug OpenAI API"""
    print("\n🔧 Debugging OpenAI...")
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ No API key")
        return
    
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    
    # Try different models
    models = ["gpt-3.5-turbo", "gpt-4o-mini"]
    
    for model in models:
        print(f"\n   Testing: {model}")
        data = {
            "model": model,
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 50
        }
        
        try:
            response = requests.post(
                'https://api.openai.com/v1/chat/completions',
                headers=headers,
                json=data,
                timeout=10
            )
            print(f"   Status: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ SUCCESS: {result['choices'][0]['message']['content'][:50]}...")
                return model
            else:
                print(f"   ❌ Error: {response.text[:200]}...")
        except Exception as e:
            print(f"   ❌ Exception: {e}")
    
    return None

def find_free_alternatives():
    """Find truly free APIs that work without registration"""
    print("\n🆓 Searching for Free Alternatives...")
    
    free_apis = [
        {
            "name": "Hugging Face (public)",
            "url": "https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium",
            "method": "POST",
            "headers": {},
            "data": {"inputs": "Hello, how are you?"}
        },
        {
            "name": "Ollama (local)",
            "url": "http://localhost:11434/api/generate",
            "method": "POST", 
            "headers": {"Content-Type": "application/json"},
            "data": {"model": "llama2", "prompt": "Hello", "stream": False}
        }
    ]
    
    working_free = []
    
    for api in free_apis:
        print(f"\n   Testing {api['name']}...")
        try:
            if api['method'] == 'POST':
                response = requests.post(
                    api['url'], 
                    headers=api['headers'], 
                    json=api['data'],
                    timeout=5
                )
            else:
                response = requests.get(api['url'], timeout=5)
            
            print(f"   Status: {response.status_code}")
            if response.status_code == 200:
                print(f"   ✅ {api['name']} is working!")
                working_free.append(api)
            else:
                print(f"   Response: {response.text[:100]}...")
                
        except Exception as e:
            print(f"   ❌ {api['name']} failed: {e}")
    
    return working_free

def main():
    print("🔧 AI API Debugging Tool")
    print("=" * 50)
    
    working_apis = []
    
    # Debug each API
    together_result = debug_together()
    if together_result:
        working_apis.append(("Together.ai", together_result))
    
    grok_result = debug_grok()
    if grok_result:
        working_apis.append(("Grok", grok_result))
    
    deepseek_result = debug_deepseek()
    if deepseek_result:
        working_apis.append(("DeepSeek", deepseek_result))
    
    openai_result = debug_openai()
    if openai_result:
        working_apis.append(("OpenAI", openai_result))
    
    # Find free alternatives
    free_alternatives = find_free_alternatives()
    
    # Summary
    print("\n" + "=" * 50)
    print("🎯 RESULTS SUMMARY")
    print("=" * 50)
    
    if working_apis:
        print("✅ Working APIs:")
        for name, config in working_apis:
            print(f"   • {name}: {config}")
    else:
        print("❌ No paid APIs working")
    
    if free_alternatives:
        print("\n✅ Free alternatives:")
        for api in free_alternatives:
            print(f"   • {api['name']}")
    else:
        print("\n❌ No free alternatives found")
    
    if not working_apis and not free_alternatives:
        print("\n💡 RECOMMENDATIONS:")
        print("   1. Install Ollama for local models (100% free)")
        print("   2. Check API key formats and account status")
        print("   3. Try adding credits to Together.ai ($5 minimum)")

if __name__ == "__main__":
    main()