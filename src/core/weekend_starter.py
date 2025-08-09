import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import requests

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

load_dotenv()

def test_setup():
    """Quick test to make sure everything works"""
    print("🎮 AI Playground - Weekend Test")
    print(f"📁 Project root: {project_root}")
    print(f"🐍 Python: {sys.version}")
    
    # Test API key
    hf_token = os.getenv('HF_TOKEN')
    if hf_token:
        print("✅ HuggingFace token found")
        # Test API call
        try:
            headers = {"Authorization": f"Bearer {hf_token}"}
            response = requests.get("https://huggingface.co/api/whoami", headers=headers)
            if response.status_code == 200:
                print("✅ HuggingFace API working")
            else:
                print("❌ HuggingFace API error")
        except Exception as e:
            print(f"❌ API test failed: {e}")
    else:
        print("❌ No HF token - add to .env file")
    
    return True

def test_huggingface_api():
    """Test free Hugging Face inference API"""
    API_URL = "https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium"
    headers = {"Authorization": f"Bearer {os.getenv('HF_TOKEN')}"}
    
    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()
    
    try:
        output = query({
            "inputs": "Hello, I'm building my own AI system!",
        })
        print("🤖 AI Response:", output)
        return output
    except Exception as e:
        print(f"❌ API call failed: {e}")
        return None

if __name__ == "__main__":
    test_setup()
    if os.getenv('HF_TOKEN'):
        test_huggingface_api()
    else:
        print("💡 Add your HuggingFace token to .env to test API calls")
