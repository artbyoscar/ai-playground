"""
EdgeMind Core Integration v0.3.0
Revolutionary Local AI System with Multi-Model Support
"""

import os
import sys
import json
import time
import psutil
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass
from enum import Enum

# Your existing imports
try:
    from ..agents.safe_computer_control import SafeComputerControl
    from ..core.smart_rag import SmartRAG
    from ..models.gpt_oss_integration import GPTOSSIntegration
except ImportError:
    # Fallback for direct execution
    sys.path.append(str(Path(__file__).parent.parent))
    from agents.safe_computer_control import SafeComputerControl
    from core.smart_rag import SmartRAG
    from models.gpt_oss_integration import GPTOSSIntegration

# Try to import optional dependencies
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False


class ModelType(Enum):
    """Available model types with their characteristics"""
    TINYLLAMA = "tinyllama"  # 670MB, 34 tok/s
    PHI2 = "phi-2"  # 1.6GB, 25 tok/s
    MISTRAL = "mistral-7b"  # 4.1GB, 12 tok/s
    CODELLAMA = "codellama-7b"  # 4.1GB, 10 tok/s
    GPT_OSS = "gpt-oss"  # OpenAI open models
    TOGETHER = "together-api"  # Together API fallback


@dataclass
class ModelConfig:
    """Configuration for each model"""
    name: str
    size_mb: int
    tokens_per_sec: int
    context_length: int
    quantization: str
    path: Optional[str] = None
    url: Optional[str] = None


class EdgeMind:
    """
    Core EdgeMind Integration - Unifying all AI capabilities
    
    Features:
    - Multi-model routing with automatic fallback
    - Local-first with cloud backup
    - Intelligent memory management
    - Performance monitoring
    - Safety controls
    """
    
    # Model configurations
    MODEL_CONFIGS = {
        ModelType.TINYLLAMA: ModelConfig(
            name="TinyLlama-1.1B",
            size_mb=670,
            tokens_per_sec=34,
            context_length=2048,
            quantization="Q4_K_M",
            url="https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
        ),
        ModelType.PHI2: ModelConfig(
            name="Phi-2",
            size_mb=1600,
            tokens_per_sec=25,
            context_length=2048,
            quantization="Q4_K_M",
            url="https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q4_K_M.gguf"
        ),
        ModelType.MISTRAL: ModelConfig(
            name="Mistral-7B",
            size_mb=4100,
            tokens_per_sec=12,
            context_length=8192,
            quantization="Q4_K_M",
            url="https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
        ),
    }
    
    def __init__(
        self,
        model_type: ModelType = ModelType.TINYLLAMA,
        models_dir: str = "./models",
        enable_rag: bool = True,
        enable_safety: bool = True,
        enable_monitoring: bool = True,
        verbose: bool = True
    ):
        """
        Initialize EdgeMind with specified configuration
        
        Args:
            model_type: Which model to use
            models_dir: Directory for model storage
            enable_rag: Enable RAG system
            enable_safety: Enable safety controls
            enable_monitoring: Enable performance monitoring
            verbose: Print status messages
        """
        self.model_type = model_type
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.verbose = verbose
        
        # Components
        self.model = None
        self.rag_system = None
        self.safety_system = None
        self.gpt_oss = None
        
        # Performance tracking
        self.metrics = {
            "total_tokens": 0,
            "total_time": 0,
            "avg_tokens_per_sec": 0,
            "memory_usage_mb": 0,
            "model_loads": 0
        }
        
        # Initialize components
        self._log("🧠 Initializing EdgeMind v0.3.0...")
        
        if enable_rag:
            self._initialize_rag()
        
        if enable_safety:
            self._initialize_safety()
        
        if enable_monitoring:
            self._start_monitoring()
        
        # Load model
        self.load_model(model_type)
        
        self._log("✅ EdgeMind ready! Running 100% locally.")
    
    def _log(self, message: str):
        """Log message if verbose"""
        if self.verbose:
            print(f"[EdgeMind] {message}")
    
    def _initialize_rag(self):
        """Initialize RAG system"""
        try:
            self.rag_system = SmartRAG()
            self._log("📚 RAG system initialized")
        except Exception as e:
            self._log(f"⚠️ RAG initialization failed: {e}")
    
    def _initialize_safety(self):
        """Initialize safety controls"""
        try:
            self.safety_system = SafeComputerControl()
            self._log("🔒 Safety controls activated")
        except Exception as e:
            self._log(f"⚠️ Safety initialization failed: {e}")
    
    def _start_monitoring(self):
        """Start performance monitoring"""
        self.start_time = time.time()
        self.initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        self._log("📊 Performance monitoring enabled")
    
    def load_model(self, model_type: ModelType):
        """
        Load specified model with automatic download if needed
        
        Args:
            model_type: Type of model to load
        """
        config = self.MODEL_CONFIGS.get(model_type)
        if not config:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model_path = self.models_dir / f"{config.name}.gguf"
        
        # Download if needed
        if not model_path.exists() and config.url:
            self._download_model(config.url, model_path)
        
        # Load model based on availability
        if LLAMA_CPP_AVAILABLE and model_path.exists():
            self._load_llama_cpp(model_path, config)
        elif model_type == ModelType.GPT_OSS:
            self._load_gpt_oss()
        elif model_type == ModelType.TOGETHER:
            self._load_together_api()
        else:
            self._log("⚠️ No suitable backend found, using mock mode")
            self.model = self._create_mock_model()
        
        self.metrics["model_loads"] += 1
    
    def _download_model(self, url: str, path: Path):
        """Download model from URL"""
        import requests
        from tqdm import tqdm
        
        self._log(f"📥 Downloading model to {path}...")
        
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(path, 'wb') as file:
            with tqdm(total=total_size, unit='iB', unit_scale=True) as pbar:
                for data in response.iter_content(chunk_size=8192):
                    pbar.update(len(data))
                    file.write(data)
        
        self._log(f"✅ Model downloaded: {path}")
    
    def _load_llama_cpp(self, model_path: Path, config: ModelConfig):
        """Load model using llama-cpp-python"""
        self._log(f"🔧 Loading {config.name} with llama.cpp...")
        
        self.model = Llama(
            model_path=str(model_path),
            n_ctx=config.context_length,
            n_batch=512,
            n_threads=psutil.cpu_count(logical=False),
            verbose=False
        )
        
        self._log(f"✅ Model loaded: {config.tokens_per_sec} tokens/sec expected")
    
    def _load_gpt_oss(self):
        """Load GPT-OSS integration"""
        self.gpt_oss = GPTOSSIntegration()
        self.model = self.gpt_oss
        self._log("✅ GPT-OSS models ready")
    
    def _load_together_api(self):
        """Load Together API as fallback"""
        # This would integrate with your existing Together setup
        self._log("🌐 Using Together API fallback")
        self.model = None  # Implement Together integration
    
    def _create_mock_model(self):
        """Create mock model for testing"""
        class MockModel:
            def __call__(self, prompt, **kwargs):
                return {"choices": [{"text": "Mock response for: " + prompt[:50]}]}
        
        return MockModel()
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        stream: bool = False,
        use_rag: bool = False,
        safety_check: bool = True
    ) -> Union[str, Dict[str, Any]]:
        """
        Generate response with EdgeMind
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stream: Stream response
            use_rag: Use RAG for context
            safety_check: Apply safety controls
            
        Returns:
            Generated text or full response dict
        """
        start_time = time.time()
        
        # RAG enhancement
        if use_rag and self.rag_system:
            context = self.rag_system.get_context(prompt)
            if context:
                prompt = f"Context: {context}\n\nQuery: {prompt}"
                self._log("📚 RAG context added")
        
        # Safety check
        if safety_check and self.safety_system:
            if not self.safety_system.is_safe(prompt):
                return "⚠️ Request blocked by safety system"
        
        # Generate response
        if isinstance(self.model, Llama):
            response = self.model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=stream
            )
            text = response["choices"][0]["text"]
        else:
            # Fallback to mock or API
            response = self.model(prompt, max_tokens=max_tokens)
            text = response["choices"][0]["text"]
        
        # Update metrics
        elapsed = time.time() - start_time
        tokens = len(text.split())
        self.metrics["total_tokens"] += tokens
        self.metrics["total_time"] += elapsed
        self.metrics["avg_tokens_per_sec"] = self.metrics["total_tokens"] / max(self.metrics["total_time"], 1)
        
        if self.verbose:
            self._log(f"⚡ Generated {tokens} tokens in {elapsed:.2f}s ({tokens/elapsed:.1f} tok/s)")
        
        return text if not stream else response
    
    def benchmark(self, prompts: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run benchmark on model
        
        Args:
            prompts: List of prompts to test
            
        Returns:
            Benchmark results
        """
        if not prompts:
            prompts = [
                "What is the capital of France?",
                "Explain quantum computing in simple terms.",
                "Write a Python function to sort a list.",
                "What are the benefits of exercise?",
                "How does photosynthesis work?"
            ]
        
        results = []
        self._log("🏃 Running benchmark...")
        
        for prompt in prompts:
            start = time.time()
            response = self.generate(prompt, max_tokens=100, safety_check=False)
            elapsed = time.time() - start
            
            results.append({
                "prompt": prompt[:50] + "...",
                "response_length": len(response),
                "time": elapsed,
                "tokens_per_sec": len(response.split()) / elapsed
            })
        
        # Calculate averages
        avg_time = sum(r["time"] for r in results) / len(results)
        avg_tokens = sum(r["tokens_per_sec"] for r in results) / len(results)
        
        # Memory usage
        current_memory = psutil.Process().memory_info().rss / 1024 / 1024
        memory_used = current_memory - self.initial_memory
        
        benchmark_results = {
            "model": self.model_type.value,
            "prompts_tested": len(prompts),
            "avg_response_time": avg_time,
            "avg_tokens_per_sec": avg_tokens,
            "memory_usage_mb": memory_used,
            "detailed_results": results
        }
        
        self._log(f"""
✅ Benchmark Complete!
📊 Model: {self.model_type.value}
⚡ Speed: {avg_tokens:.1f} tokens/sec
💾 Memory: {memory_used:.1f} MB
⏱️ Avg Response: {avg_time:.2f}s
        """)
        
        return benchmark_results
    
    def chat(self):
        """Interactive chat mode"""
        self._log("""
💬 Chat Mode - EdgeMind v0.3.0
Commands:
  /rag <query>  - Search with RAG
  /safe <query> - Check safety
  /bench       - Run benchmark
  /switch <model> - Switch model
  /quit        - Exit
        """)
        
        while True:
            try:
                user_input = input("\n👤 You: ").strip()
                
                if user_input.lower() == '/quit':
                    break
                elif user_input.lower() == '/bench':
                    self.benchmark()
                elif user_input.startswith('/switch'):
                    model_name = user_input.split(' ', 1)[1] if ' ' in user_input else 'tinyllama'
                    self.load_model(ModelType(model_name))
                elif user_input.startswith('/rag'):
                    query = user_input.split(' ', 1)[1] if ' ' in user_input else ''
                    response = self.generate(query, use_rag=True)
                    print(f"\n🤖 EdgeMind: {response}")
                elif user_input.startswith('/safe'):
                    query = user_input.split(' ', 1)[1] if ' ' in user_input else ''
                    is_safe = self.safety_system.is_safe(query) if self.safety_system else True
                    print(f"\n🔒 Safety Check: {'✅ SAFE' if is_safe else '⚠️ UNSAFE'}")
                else:
                    response = self.generate(user_input)
                    print(f"\n🤖 EdgeMind: {response}")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                self._log(f"❌ Error: {e}")
        
        self._log("👋 Goodbye!")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        current_memory = psutil.Process().memory_info().rss / 1024 / 1024
        self.metrics["memory_usage_mb"] = current_memory - self.initial_memory
        return self.metrics


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="EdgeMind - Local AI System")
    parser.add_argument("--model", default="tinyllama", help="Model to use")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark")
    parser.add_argument("--chat", action="store_true", help="Start chat mode")
    parser.add_argument("--prompt", type=str, help="Single prompt to run")
    
    args = parser.parse_args()
    
    # Initialize EdgeMind
    edgemind = EdgeMind(
        model_type=ModelType(args.model),
        enable_rag=True,
        enable_safety=True,
        enable_monitoring=True
    )
    
    if args.benchmark:
        edgemind.benchmark()
    elif args.chat:
        edgemind.chat()
    elif args.prompt:
        response = edgemind.generate(args.prompt)
        print(f"Response: {response}")
    else:
        # Demo mode
        print("\n🎯 EdgeMind Demo - Running locally with zero cloud dependency!\n")
        
        demo_prompts = [
            "What is artificial intelligence?",
            "Write a haiku about local AI",
            "Explain the benefits of edge computing"
        ]
        
        for prompt in demo_prompts:
            print(f"📝 Prompt: {prompt}")
            response = edgemind.generate(prompt, max_tokens=100)
            print(f"🤖 Response: {response}\n")
        
        print("\n📊 Session Metrics:")
        metrics = edgemind.get_metrics()
        for key, value in metrics.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()