# src/core/multi_api_manager.py
import os
import requests
from dotenv import load_dotenv
import json
from datetime import datetime

load_dotenv()

class MultiAPIManager:
    """Smart API manager that tries providers in order of cost-effectiveness"""
    
    def __init__(self):
        self.providers = [
            {
                'name': 'Together.ai',
                'api_key': os.getenv('TOGETHER_API_KEY'),
                'cost': 'FREE/VERY_CHEAP',
                'url': 'https://api.together.xyz/v1/chat/completions',
                'model': 'openai/gpt-oss-120b',
                'test_func': self.test_together
            },
            {
                'name': 'Grok (xAI)',
                'api_key': os.getenv('XAI_API_KEY'),
                'cost': 'FREE_TRIAL',
                'url': 'https://api.x.ai/v1/chat/completions',
                'model': 'grok-4-latest',
                'test_func': self.test_grok
            },
            {
                'name': 'DeepSeek',
                'api_key': os.getenv('DEEPSEEK_API_KEY'),
                'cost': 'VERY_CHEAP',
                'url': 'https://api.deepseek.com/v1/chat/completions',
                'model': 'deepseek-coder',
                'test_func': self.test_deepseek
            },
            {
                'name': 'OpenAI',
                'api_key': os.getenv('OPENAI_API_KEY'),
                'cost': 'EXPENSIVE',
                'url': 'https://api.openai.com/v1/chat/completions',
                'model': 'gpt-3.5-turbo',
                'test_func': self.test_openai
            }
        ]
        
        # Filter out providers without API keys
        self.providers = [p for p in self.providers if p['api_key']]
        
    def test_together(self, message="Hello! Test response please."):
        """Test Together.ai API"""
        try:
            # Try the SDK approach first
            try:
                from together import Together
                client = Together(api_key=os.getenv('TOGETHER_API_KEY'))
                response = client.chat.completions.create(
                    model="openai/gpt-oss-120b",
                    messages=[{"role": "user", "content": message}]
                )
                return response.choices[0].message.content
            except ImportError:
                # Fallback to requests
                headers = {
                    'Authorization': f'Bearer {os.getenv("TOGETHER_API_KEY")}',
                    'Content-Type': 'application/json'
                }
                data = {
                    "model": "openai/gpt-oss-120b",
                    "messages": [{"role": "user", "content": message}],
                    "max_tokens": 150
                }
                response = requests.post(
                    'https://api.together.xyz/v1/chat/completions',
                    headers=headers,
                    json=data
                )
                if response.status_code == 200:
                    return response.json()['choices'][0]['message']['content']
                return None
        except Exception as e:
            print(f"Together.ai error: {e}")
            return None
    
    def test_grok(self, message="Hello! Test response please."):
        """Test Grok/xAI API"""
        try:
            headers = {
                'Authorization': f'Bearer {os.getenv("XAI_API_KEY")}',
                'Content-Type': 'application/json'
            }
            data = {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": message}
                ],
                "model": "grok-4-latest",
                "stream": False,
                "temperature": 0.3
            }
            response = requests.post(
                'https://api.x.ai/v1/chat/completions',
                headers=headers,
                json=data
            )
            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content']
            return None
        except Exception as e:
            print(f"Grok error: {e}")
            return None
    
    def test_deepseek(self, message="Hello! Test response please."):
        """Test DeepSeek API"""
        try:
            headers = {
                'Authorization': f'Bearer {os.getenv("DEEPSEEK_API_KEY")}',
                'Content-Type': 'application/json'
            }
            data = {
                "model": "deepseek-coder",
                "messages": [{"role": "user", "content": message}],
                "stream": False
            }
            response = requests.post(
                'https://api.deepseek.com/v1/chat/completions',
                headers=headers,
                json=data
            )
            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content']
            return None
        except Exception as e:
            print(f"DeepSeek error: {e}")
            return None
    
    def test_openai(self, message="Hello! Test response please."):
        """Test OpenAI API"""
        try:
            headers = {
                'Authorization': f'Bearer {os.getenv("OPENAI_API_KEY")}',
                'Content-Type': 'application/json'
            }
            data = {
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": message}],
                "max_tokens": 150
            }
            response = requests.post(
                'https://api.openai.com/v1/chat/completions',
                headers=headers,
                json=data
            )
            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content']
            return None
        except Exception as e:
            print(f"OpenAI error: {e}")
            return None
    
    def test_all_providers(self):
        """Test all providers and return working ones"""
        print("🧪 Testing All AI Providers...")
        print("=" * 50)
        
        working_providers = []
        
        for provider in self.providers:
            print(f"\n🔧 Testing {provider['name']} ({provider['cost']})...")
            
            if not provider['api_key']:
                print(f"❌ No API key for {provider['name']}")
                continue
                
            response = provider['test_func']("Say 'Hello from AI Playground!'")
            
            if response:
                print(f"✅ {provider['name']} WORKING!")
                print(f"   Response: {response[:100]}...")
                working_providers.append(provider)
            else:
                print(f"❌ {provider['name']} failed")
        
        return working_providers
    
    def get_best_provider(self):
        """Get the cheapest working provider"""
        working = self.test_all_providers()
        if working:
            best = working[0]  # First in list = cheapest
            print(f"\n🎯 Best Provider: {best['name']} ({best['cost']})")
            return best
        return None
    
    def chat(self, message, provider_name=None):
        """Smart chat that uses the best available provider"""
        if provider_name:
            # Use specific provider
            provider = next((p for p in self.providers if p['name'] == provider_name), None)
            if provider:
                return provider['test_func'](message)
        
        # Use best available provider
        for provider in self.providers:
            if provider['api_key']:
                response = provider['test_func'](message)
                if response:
                    print(f"✅ Used {provider['name']}")
                    return response
        
        return "❌ No working providers available"

def main():
    """Test all providers and show the best one"""
    print("🎮 AI Playground - Multi-Provider Test")
    print("=" * 50)
    
    manager = MultiAPIManager()
    
    print(f"📡 Found {len(manager.providers)} providers with API keys:")
    for p in manager.providers:
        print(f"   • {p['name']} ({p['cost']})")
    
    # Test all and find the best
    best = manager.get_best_provider()
    
    if best:
        print(f"\n🚀 Ready to build! Using {best['name']} as primary.")
        
        # Interactive test
        test_message = "Hello! I'm building an AI playground. Give me one tip for success."
        print(f"\n🤖 Testing with: '{test_message}'")
        response = manager.chat(test_message)
        print(f"AI: {response}")
        
    else:
        print("\n❌ No working providers. Check your API keys!")

if __name__ == "__main__":
    main()