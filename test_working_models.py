# Save as test_working_models.py
import subprocess
import time

def test_model(model_name, prompt):
    """Test a model with timing"""
    print(f"\n{'='*50}")
    print(f"Testing: {model_name}")
    print('='*50)
    
    start = time.time()
    cmd = f'ollama run {model_name} "{prompt}"'
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        print(f"Response: {result.stdout[:200]}...")
        print(f"Time: {time.time()-start:.2f}s")
        return True
    except subprocess.TimeoutExpired:
        print("❌ Too slow (>30s)")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

# Test all your models
models = [
    "phi3:mini",
    "llama3.2:3b", 
    "deepseek-r1:7b-qwen-distill-q4_k_m",
    "deepseek-r1:14b"  # Might be slow
]

prompt = "Write a Python function to reverse a string"

print("🧪 TESTING MODELS ON YOUR LAPTOP")
results = {}

for model in models:
    results[model] = test_model(model, prompt)

print("\n📊 RESULTS SUMMARY:")
for model, success in results.items():
    status = "✅ Works" if success else "❌ Failed"
    print(f"{model}: {status}")