# src/core/local_llm_engine.py
"""
REAL Local AI - No API calls, runs on YOUR hardware
"""

import os
import json
import time
import psutil
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, List
import requests
from dataclasses import dataclass
import numpy as np

@dataclass
class ModelConfig:
    """Configuration for local models"""
    name: str
    size_gb: float
    ram_required_gb: int
    quantization: str  # Q4_K_M, Q5_K_M, Q8_0
    context_length: int
    url: str

class LocalLLMEngine:
    """
    This is the REAL deal - runs models locally using llama.cpp
    No API calls, no cloud, just raw compute on YOUR machine
    """
    
    # Models that ACTUALLY work on consumer hardware
    MODELS = {
        "mistral-7b-instruct": ModelConfig(
            name="Mistral-7B-Instruct-v0.2-GGUF",
            size_gb=4.1,
            ram_required_gb=6,
            quantization="Q4_K_M",
            context_length=32768,
            url="https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
        ),
        "llama2-7b": ModelConfig(
            name="Llama-2-7B-Chat-GGUF",
            size_gb=3.8,
            ram_required_gb=6,
            quantization="Q4_K_M",
            context_length=4096,
            url="https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf"
        ),
        "phi-2": ModelConfig(
            name="Phi-2-GGUF",
            size_gb=1.6,
            ram_required_gb=3,
            quantization="Q4_K_M",
            context_length=2048,
            url="https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q4_K_M.gguf"
        ),
        "tinyllama": ModelConfig(
            name="TinyLlama-1.1B-Chat-GGUF",
            size_gb=0.7,
            ram_required_gb=2,
            quantization="Q4_K_M",
            context_length=2048,
            url="https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
        )
    }
    
    def __init__(self, model_name: str = "tinyllama", models_dir: str = "./models"):
        self.model_name = model_name
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        self.model_config = self.MODELS[model_name]
        self.model_path = self.models_dir / f"{model_name}.gguf"
        self.llama_cpp_path = self.models_dir / "llama.cpp"
        
        # Performance metrics
        self.inference_times = []
        self.tokens_per_second = []
        
        print(f"🚀 Initializing LOCAL LLM Engine with {model_name}")
        print(f"📊 RAM Required: {self.model_config.ram_required_gb}GB")
        print(f"💾 Model Size: {self.model_config.size_gb}GB")
        
    def check_system_requirements(self) -> bool:
        """Check if system can run the model"""
        ram_gb = psutil.virtual_memory().total / (1024**3)
        available_gb = psutil.virtual_memory().available / (1024**3)
        
        print(f"💻 System RAM: {ram_gb:.1f}GB (Available: {available_gb:.1f}GB)")
        
        if available_gb < self.model_config.ram_required_gb:
            print(f"⚠️ WARNING: Need {self.model_config.ram_required_gb}GB RAM, only {available_gb:.1f}GB available")
            return False
        
        return True
    
    def install_llama_cpp(self):
        """Install llama.cpp for local inference"""
        print("📦 Installing llama.cpp for local inference...")
        
        # For Windows
        if os.name == 'nt':
            # Download pre-built Windows binary
            url = "https://github.com/ggerganov/llama.cpp/releases/latest/download/llama-b1696-bin-win-avx2-x64.zip"
            
            import zipfile
            import urllib.request
            
            zip_path = self.models_dir / "llama.cpp.zip"
            
            print("📥 Downloading llama.cpp Windows binary...")
            urllib.request.urlretrieve(url, zip_path)
            
            print("📦 Extracting...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.llama_cpp_path)
            
            print("✅ llama.cpp installed!")
            
        else:
            # Build from source for Linux/Mac
            if not self.llama_cpp_path.exists():
                print("🔨 Building llama.cpp from source...")
                subprocess.run([
                    "git", "clone", 
                    "https://github.com/ggerganov/llama.cpp.git",
                    str(self.llama_cpp_path)
                ])
                
                # Build
                subprocess.run(["make", "-j"], cwd=self.llama_cpp_path)
                print("✅ llama.cpp built!")
    
    def download_model(self) -> bool:
        """Download the actual model file"""
        if self.model_path.exists():
            print(f"✅ Model already downloaded: {self.model_path}")
            return True
        
        print(f"📥 Downloading {self.model_config.name}...")
        print(f"   Size: {self.model_config.size_gb}GB")
        print(f"   This will take a few minutes...")
        
        try:
            response = requests.get(self.model_config.url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            
            with open(self.model_path, 'wb') as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    # Progress
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"   Progress: {percent:.1f}%", end='\r')
            
            print(f"\n✅ Model downloaded: {self.model_path}")
            return True
            
        except Exception as e:
            print(f"❌ Download failed: {e}")
            return False
    
    def run_inference(self, prompt: str, max_tokens: int = 512) -> Dict[str, Any]:
        """
        Run ACTUAL local inference - no API calls!
        This is where the magic happens
        """
        start_time = time.time()
        
        # Prepare the command for llama.cpp
        if os.name == 'nt':
            exe_path = self.llama_cpp_path / "main.exe"
        else:
            exe_path = self.llama_cpp_path / "main"
        
        cmd = [
            str(exe_path),
            "-m", str(self.model_path),
            "-p", prompt,
            "-n", str(max_tokens),
            "-t", "4",  # threads
            "--temp", "0.7",
            "--repeat-penalty", "1.1",
            "--no-display-prompt"
        ]
        
        print(f"🧠 Running LOCAL inference (no API!)...")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            
            # Calculate tokens per second
            output_text = result.stdout
            approx_tokens = len(output_text.split())
            tps = approx_tokens / inference_time
            self.tokens_per_second.append(tps)
            
            return {
                "response": output_text,
                "inference_time": inference_time,
                "tokens_per_second": tps,
                "model": self.model_name,
                "local": True,
                "api_used": False,
                "cost": 0.0
            }
            
        except Exception as e:
            print(f"❌ Inference failed: {e}")
            return {
                "response": f"Local inference failed: {e}",
                "error": str(e)
            }
    
    def benchmark(self) -> Dict[str, float]:
        """Benchmark local vs API performance"""
        test_prompts = [
            "What is 2+2?",
            "Explain quantum computing in one sentence.",
            "Write a haiku about coding."
        ]
        
        results = {
            "avg_inference_time": 0,
            "avg_tokens_per_second": 0,
            "total_cost_saved": 0,
            "api_equivalent_cost": 0
        }
        
        print("\n📊 Running benchmark...")
        
        for prompt in test_prompts:
            result = self.run_inference(prompt, max_tokens=50)
            
            # Calculate what this would cost on OpenAI
            api_cost = (50 / 1000) * 0.002  # $0.002 per 1K tokens
            results["api_equivalent_cost"] += api_cost
        
        if self.inference_times:
            results["avg_inference_time"] = np.mean(self.inference_times)
            results["avg_tokens_per_second"] = np.mean(self.tokens_per_second)
            results["total_cost_saved"] = results["api_equivalent_cost"]
        
        print(f"\n🏆 Benchmark Results:")
        print(f"   Avg Inference Time: {results['avg_inference_time']:.2f}s")
        print(f"   Tokens/Second: {results['avg_tokens_per_second']:.1f}")
        print(f"   Cost Saved: ${results['total_cost_saved']:.4f}")
        
        return results
    
    def setup(self):
        """Complete setup process"""
        print("\n🚀 Setting up LOCAL AI (no cloud!)...\n")
        
        # Check system
        if not self.check_system_requirements():
            print("⚠️ System requirements not met, but continuing anyway...")
        
        # Install llama.cpp
        self.install_llama_cpp()
        
        # Download model
        if not self.download_model():
            print("❌ Failed to download model")
            return False
        
        print("\n✅ Local AI is READY! No more API calls needed!")
        return True


class EdgeFormerCompressor:
    """
    This is where we implement ACTUAL model compression
    Not just talk about it - actually DO it
    """
    
    @staticmethod
    def quantize_model(model_path: Path, target_bits: int = 4) -> Path:
        """
        Compress model using quantization
        This is REAL compression, not theoretical
        """
        print(f"🗜️ Compressing model to {target_bits}-bit...")
        
        output_path = model_path.parent / f"{model_path.stem}_Q{target_bits}.gguf"
        
        # Use llama.cpp's quantize tool
        cmd = [
            "./models/llama.cpp/quantize",
            str(model_path),
            str(output_path),
            f"Q{target_bits}_K_M"
        ]
        
        try:
            subprocess.run(cmd, check=True)
            
            # Compare sizes
            original_size = model_path.stat().st_size / (1024**3)
            compressed_size = output_path.stat().st_size / (1024**3)
            compression_ratio = original_size / compressed_size
            
            print(f"✅ Compression complete!")
            print(f"   Original: {original_size:.2f}GB")
            print(f"   Compressed: {compressed_size:.2f}GB")
            print(f"   Ratio: {compression_ratio:.1f}x")
            
            return output_path
            
        except Exception as e:
            print(f"❌ Compression failed: {e}")
            return model_path


# Test it RIGHT NOW
if __name__ == "__main__":
    print("=" * 60)
    print("🚀 EdgeMind LOCAL AI Engine - NO CLOUD, NO API, JUST LOCAL")
    print("=" * 60)
    
    # Start with TinyLlama - it's small and fast
    engine = LocalLLMEngine(model_name="tinyllama")
    
    # Set it up
    if engine.setup():
        # Test inference
        response = engine.run_inference(
            "You are a helpful AI assistant. User: What makes EdgeMind different from ChatGPT? Assistant:",
            max_tokens=100
        )
        
        print("\n📝 Response:")
        print(response["response"])
        
        # Run benchmark
        engine.benchmark()
        
        print("\n🎯 This is REAL local AI - no API calls, no cloud, just YOUR hardware!")