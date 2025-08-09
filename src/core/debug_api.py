# src/core/debug_api.py
import os
import requests
import json
import time
from dotenv import load_dotenv

load_dotenv()

def test_auth():
    """Test if your token works"""
    print("🔑 Testing HuggingFace Authentication...")
    
    hf_token = os.getenv('HF_TOKEN')
    if not hf_token:
        print("❌ No HF_TOKEN found in .env file")
        return False
    
    headers = {"Authorization": f"Bearer {hf_token}"}
    
    try:
        response = requests.get("https://huggingface.co/api/whoami", headers=headers)
        print(f"Auth Status Code: {response.status_code}")
        
        if response.status_code == 200:
            user_info = response.json()
            print(f"✅ Authenticated as: {user_info.get('name', 'Unknown')}")
            return True
        else:
            print(f"❌ Auth failed: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Auth error: {e}")
        return False

def test_simple_model():
    """Test with a simple, fast model"""
    print("\n🤖 Testing Simple Model (GPT-2)...")
    
    hf_token = os.getenv('HF_TOKEN')
    headers = {"Authorization": f"Bearer {hf_token}"}
    
    # Use GPT-2 - it's fast and reliable
    API_URL = "https://api-inference.huggingface.co/models/gpt2"
    
    payload = {
        "inputs": "Hello world",
        "parameters": {
            "max_length": 50,
            "temperature": 0.7
        }
    }
    
    try:
        print(f"📡 Calling: {API_URL}")
        response = requests.post(API_URL, headers=headers, json=payload)
        print(f"Status Code: {response.status_code}")
        print(f"Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Success!")
            print(f"Response: {result}")
            return True
        else:
            print(f"❌ Error Response: {response.text}")
            if response.status_code == 503:
                print("💡 Model is loading - this is normal! Try again in 30 seconds.")
            return False
            
    except json.JSONDecodeError as e:
        print(f"❌ JSON decode error: {e}")
        print(f"Raw response: {response.text}")
        return False
    except Exception as e:
        print(f"❌ Request error: {e}")
        return False

def test_model_status():
    """Check if model is loaded"""
    print("\n📊 Checking Model Status...")
    
    hf_token = os.getenv('HF_TOKEN')
    headers = {"Authorization": f"Bearer {hf_token}"}
    
    # Check multiple models
    models_to_test = [
        "gpt2",
        "microsoft/DialoGPT-small",
        "facebook/blenderbot-400M-distill"
    ]
    
    for model in models_to_test:
        API_URL = f"https://api-inference.huggingface.co/models/{model}"
        try:
            response = requests.post(
                API_URL, 
                headers=headers, 
                json={"inputs": "test"}
            )
            
            if response.status_code == 200:
                print(f"✅ {model} - Ready")
            elif response.status_code == 503:
                print(f"⏳ {model} - Loading (try again later)")
            else:
                print(f"❌ {model} - Error {response.status_code}")
                
        except Exception as e:
            print(f"❌ {model} - Exception: {e}")

def wait_for_model():
    """Wait for model to load with progress"""
    print("\n⏳ Waiting for model to load...")
    
    for i in range(6):  # Wait up to 60 seconds
        print(f"Attempt {i+1}/6...")
        if test_simple_model():
            print("🎉 Model is ready!")
            return True
        
        if i < 5:  # Don't wait after last attempt
            print("⏳ Waiting 10 seconds...")
            time.sleep(10)
    
    print("❌ Model didn't load in time. Try again later.")
    return False

if __name__ == "__main__":
    print("🔧 HuggingFace API Diagnostic Tool")
    print("=" * 50)
    
    # Step 1: Test auth
    if not test_auth():
        print("\n💡 Fix your token first!")
        exit(1)
    
    # Step 2: Test simple model
    if test_simple_model():
        print("\n🎉 Everything works! Your API is ready.")
    else:
        print("\n⏳ Model might be loading...")
        # Step 3: Wait for model if needed
        wait_for_model()
    
    # Step 4: Check status of multiple models
    test_model_status()
    
    print("\n✅ Diagnostic complete!")