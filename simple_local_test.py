# simple_local_test.py
"""
Simplest possible local AI test
This proves it works with minimal code
"""

from llama_cpp import Llama
import requests
from pathlib import Path
import time

print("🚀 EdgeMind - Minimal Local AI Test\n")

# Create models directory
Path("models").mkdir(exist_ok=True)

# Download TinyLlama if needed (smallest model - 670MB)
model_path = Path("models/tinyllama.gguf")
if not model_path.exists():
    print("📥 Downloading TinyLlama (670MB)...")
    print("   This is a one-time download\n")
    
    url = "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    
    response = requests.get(url, stream=True)
    total = int(response.headers.get('content-length', 0))
    
    with open(model_path, 'wb') as f:
        downloaded = 0
        for chunk in response.iter_content(chunk_size=1024*1024):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                percent = (downloaded / total) * 100 if total > 0 else 0
                print(f"   Progress: {percent:.1f}%", end='\r')
    
    print("\n✅ Download complete!\n")

# Load model
print("🧠 Loading model (takes 10-30 seconds)...")
llm = Llama(
    model_path=str(model_path),
    n_ctx=2048,
    n_threads=4,
    verbose=False
)
print("✅ Model loaded!\n")

# Test it!
print("="*50)
print("TEST: Local AI Running on YOUR Computer")
print("="*50)

prompts = [
    "What is 2+2?",
    "Write a haiku about computers",
    "Name 3 colors"
]

for prompt in prompts:
    print(f"\n👤 Question: {prompt}")
    print("🤖 EdgeMind: ", end="", flush=True)
    
    start = time.time()
    response = llm(
        f"User: {prompt}\nAssistant:",
        max_tokens=50,
        temperature=0.7,
        stop=["User:", "\n\n"],
        echo=False
    )
    elapsed = time.time() - start
    
    text = response['choices'][0]['text'].strip()
    tokens = response['usage']['completion_tokens']
    
    print(text)
    print(f"   ⚡ Speed: {tokens/elapsed:.1f} tokens/sec | Time: {elapsed:.2f}s | Cost: $0.00")

print("\n" + "="*50)
print("✅ SUCCESS! This ran 100% locally!")
print("🚀 No internet, no API, no monthly fees!")
print("="*50)