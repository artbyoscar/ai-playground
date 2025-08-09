# src/core/local_llm_engine_fixed.py
"""
WORKING Local AI using llama-cpp-python
This actually runs models on YOUR hardware - no BS
"""

import os
import time
import psutil
import requests
from pathlib import Path
from typing import Optional, Dict, Any
import json
from llama_cpp import Llama
import numpy as np

class LocalLLMEngine:
    """
    REAL local inference - no API calls, runs on YOUR CPU/GPU
    """
    
    # Small models that actually work on consumer hardware
    MODELS = {
        "tinyllama": {
            "name": "TinyLlama-1.1B-Chat",
            "url": "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
            "size_gb": 0.67,
            "ram_gb": 2,
            "context": 2048
        },
        "phi-2": {
            "name": "Phi-2",
            "url": "https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q4_K_M.gguf",
            "size_gb": 1.6,
            "ram_gb": 3,
            "context": 2048
        },
        "mistral": {
            "name": "Mistral-7B-Instruct",
            "url": "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
            "size_gb": 4.1,
            "ram_gb": 6,
            "context": 32768
        },
        "llama2-7b": {
            "name": "Llama-2-7B-Chat",
            "url": "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf",
            "size_gb": 3.8,
            "ram_gb": 6,
            "context": 4096
        }
    }
    
    def __init__(self, model_name: str = "tinyllama"):
        self.model_name = model_name
        self.model_info = self.MODELS[model_name]
        self.models_dir = Path("./models")
        self.models_dir.mkdir(exist_ok=True)
        self.model_path = self.models_dir / f"{model_name}.gguf"
        self.llm = None
        
        print(f"🚀 EdgeMind Local AI Engine")
        print(f"📊 Model: {self.model_info['name']}")
        print(f"💾 Size: {self.model_info['size_gb']}GB")
        print(f"🧠 RAM Required: {self.model_info['ram_gb']}GB")
    
    def check_system(self) -> bool:
        """Check if we can run this model"""
        available_ram = psutil.virtual_memory().available / (1024**3)
        total_ram = psutil.virtual_memory().total / (1024**3)
        
        print(f"\n💻 System Check:")
        print(f"   Total RAM: {total_ram:.1f}GB")
        print(f"   Available: {available_ram:.1f}GB")
        print(f"   Required: {self.model_info['ram_gb']}GB")
        
        if available_ram < self.model_info['ram_gb']:
            print(f"⚠️ Low RAM! Trying anyway...")
            return False
        
        print(f"✅ System ready!")
        return True
    
    def download_model(self) -> bool:
        """Download model from HuggingFace"""
        if self.model_path.exists():
            print(f"✅ Model already downloaded: {self.model_path}")
            return True
        
        print(f"\n📥 Downloading {self.model_info['name']}...")
        print(f"   From: HuggingFace")
        print(f"   Size: {self.model_info['size_gb']}GB")
        print(f"   This is a one-time download...")
        
        try:
            response = requests.get(self.model_info['url'], stream=True)
            total_size = int(response.headers.get('content-length', 0))
            
            with open(self.model_path, 'wb') as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=1024*1024):  # 1MB chunks
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        # Progress bar
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            mb_downloaded = downloaded / (1024*1024)
                            mb_total = total_size / (1024*1024)
                            print(f"   Progress: {percent:.1f}% ({mb_downloaded:.0f}/{mb_total:.0f} MB)", end='\r')
            
            print(f"\n✅ Model downloaded successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Download failed: {e}")
            if self.model_path.exists():
                self.model_path.unlink()  # Remove partial download
            return False
    
    def load_model(self) -> bool:
        """Load model into memory"""
        if not self.model_path.exists():
            print("❌ Model file not found. Download first!")
            return False
        
        print(f"\n🧠 Loading model into memory...")
        print(f"   This takes 10-30 seconds...")
        
        try:
            # Load with llama-cpp-python
            self.llm = Llama(
                model_path=str(self.model_path),
                n_ctx=self.model_info['context'],  # Context window
                n_threads=psutil.cpu_count(logical=False),  # Physical cores
                n_gpu_layers=0,  # CPU only for now
                verbose=False
            )
            
            print(f"✅ Model loaded! Ready for inference!")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return False
    
    def generate(self, prompt: str, max_tokens: int = 256) -> Dict[str, Any]:
        """
        Generate text LOCALLY - this is the magic!
        NO API CALLS - runs on YOUR hardware
        """
        if not self.llm:
            return {"error": "Model not loaded"}
        
        print(f"\n🔮 Generating response locally...")
        start_time = time.time()
        
        try:
            # Generate with streaming
            response = self.llm(
                prompt,
                max_tokens=max_tokens,
                temperature=0.7,
                top_p=0.95,
                repeat_penalty=1.1,
                stop=["User:", "\n\n"]
            )
            
            inference_time = time.time() - start_time
            text = response['choices'][0]['text']
            tokens_generated = response['usage']['completion_tokens']
            tokens_per_second = tokens_generated / inference_time if inference_time > 0 else 0
            
            return {
                "response": text,
                "tokens": tokens_generated,
                "time": inference_time,
                "tokens_per_second": tokens_per_second,
                "model": self.model_name,
                "cost": 0.0,  # IT'S FREE!
                "local": True
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def chat(self, message: str) -> str:
        """Simple chat interface"""
        # Format as chat prompt
        if "llama" in self.model_name:
            prompt = f"<s>[INST] {message} [/INST]"
        elif "mistral" in self.model_name:
            prompt = f"<s>[INST] {message} [/INST]"
        else:
            prompt = f"User: {message}\nAssistant:"
        
        result = self.generate(prompt)
        
        if "error" in result:
            return f"Error: {result['error']}"
        
        return result["response"]
    
    def benchmark(self) -> Dict[str, float]:
        """Compare local vs cloud performance"""
        print("\n📊 Running benchmark...")
        
        test_prompts = [
            "What is 2+2?",
            "Write a haiku about AI",
            "Explain quantum computing in one sentence"
        ]
        
        times = []
        tokens_per_sec = []
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\nTest {i}/3: {prompt[:30]}...")
            result = self.generate(prompt, max_tokens=50)
            
            if "error" not in result:
                times.append(result["time"])
                tokens_per_sec.append(result["tokens_per_second"])
                print(f"   ⚡ {result['tokens_per_second']:.1f} tokens/sec")
        
        if times:
            avg_time = np.mean(times)
            avg_tps = np.mean(tokens_per_sec)
            
            # Calculate savings vs GPT-4
            queries_per_month = 1000
            gpt4_cost = queries_per_month * 0.03  # ~$0.03 per query
            
            print(f"\n🏆 Benchmark Results:")
            print(f"   Average response time: {avg_time:.2f}s")
            print(f"   Average speed: {avg_tps:.1f} tokens/sec")
            print(f"   Monthly cost (1000 queries):")
            print(f"      GPT-4: ${gpt4_cost:.2f}")
            print(f"      EdgeMind: $0.00")
            print(f"   Savings: ${gpt4_cost:.2f}/month!")
            
            return {
                "avg_time": avg_time,
                "avg_tokens_per_sec": avg_tps,
                "monthly_savings": gpt4_cost
            }
        
        return {}


def quick_test():
    """Quick test to show it works"""
    print("="*60)
    print("🚀 EdgeMind Local AI - Quick Test")
    print("="*60)
    
    # Use TinyLlama for quick test (smallest model)
    engine = LocalLLMEngine("tinyllama")
    
    # Check system
    engine.check_system()
    
    # Download model (one-time)
    if not engine.model_path.exists():
        if not engine.download_model():
            print("Failed to download. Check internet connection.")
            return
    
    # Load model
    if not engine.load_model():
        print("Failed to load model.")
        return
    
    # Test generation
    print("\n" + "="*60)
    print("💬 TEST CONVERSATION")
    print("="*60)
    
    test_messages = [
        "What is EdgeMind?",
        "Write a Python function to reverse a string",
        "What's the capital of France?"
    ]
    
    for msg in test_messages:
        print(f"\n👤 User: {msg}")
        print(f"🤖 EdgeMind: ", end="", flush=True)
        
        response = engine.chat(msg)
        print(response)
        
        # Show stats
        result = engine.generate(msg, max_tokens=50)
        if "tokens_per_second" in result:
            print(f"\n   ⚡ Speed: {result['tokens_per_second']:.1f} tokens/sec")
            print(f"   💰 Cost: $0.00 (it's local!)")
    
    # Benchmark
    engine.benchmark()
    
    print("\n" + "="*60)
    print("✅ EdgeMind is running 100% locally!")
    print("🚀 No API, no cloud, no monthly fees!")
    print("="*60)


if __name__ == "__main__":
    quick_test()