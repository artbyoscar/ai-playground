"""
EdgeMind Core Integration v0.4.0
Now with Ollama integration for models that actually work on laptops!
"""

import os
import sys
import json
import time
import psutil
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass
from enum import Enum

# Fix imports for direct execution
sys.path.insert(0, str(Path(__file__).parent.parent))

# Try to import optional components
try:
    from agents.safe_computer_control import SafeComputerControl
    SAFETY_AVAILABLE = True
except ImportError:
    SAFETY_AVAILABLE = False
    class SafeComputerControl:
        def is_safe(self, prompt): return True

try:
    from core.smart_rag import SmartRAG
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    class SmartRAG:
        def get_context(self, prompt): return None

# Mock GPT-OSS for now
class GPTOSSIntegration:
    def __init__(self):
        self.name = "GPT-OSS"
    def generate(self, prompt, **kwargs):
        return "GPT-OSS not implemented. Use Ollama models instead."

# Try Ollama Python library (optional)
try:
    import ollama
    OLLAMA_PYTHON = True
except ImportError:
    OLLAMA_PYTHON = False
    print("📦 Tip: Install ollama-python for better integration: pip install ollama")


class ModelType(Enum):
    """Available models that actually work on laptops"""
    PHI3_MINI = "phi3:mini"  # 2.2GB, fastest
    LLAMA32_3B = "llama3.2:3b"  # 2.0GB, balanced
    DEEPSEEK_7B = "deepseek-r1:7b-qwen-distill-q4_k_m"  # 4.7GB, best for code
    DEEPSEEK_14B = "deepseek-r1:14b"  # 9.0GB, powerful but slow
    # Removed models that don't work on laptops:
    # MIXTRAL = "mixtral:8x7b"  # 26GB - NEEDS 32GB RAM!


@dataclass
class ModelConfig:
    """Configuration for each model"""
    name: str
    size_gb: float
    expected_speed: str  # tokens/sec on laptop
    context_length: int
    best_for: str
    max_ram_needed: int  # GB
    temperature: float = 0.7  # Default temp for this model


class EdgeMind:
    """
    EdgeMind v0.4.0 - Realistic local AI for consumer laptops
    
    Major changes:
    - Ollama integration (no more GGUF downloads)
    - Only includes models that work on 8-16GB RAM laptops
    - Proper safety integration
    - Model routing based on query type
    - Conversation memory
    """
    
    # Realistic model configs for laptops
    MODEL_CONFIGS = {
        ModelType.PHI3_MINI: ModelConfig(
            name="Microsoft Phi-3 Mini",
            size_gb=2.2,
            expected_speed="15-20 tok/s",
            context_length=4096,
            best_for="Quick responses, general chat",
            max_ram_needed=4,
            temperature=0.7
        ),
        ModelType.LLAMA32_3B: ModelConfig(
            name="Meta Llama 3.2 3B",
            size_gb=2.0,
            expected_speed="12-15 tok/s",
            context_length=8192,
            best_for="Balanced performance, safety",
            max_ram_needed=4,
            temperature=0.7
        ),
        ModelType.DEEPSEEK_7B: ModelConfig(
            name="DeepSeek R1 7B Distilled",
            size_gb=4.7,
            expected_speed="8-10 tok/s",
            context_length=32768,
            best_for="Coding, technical tasks",
            max_ram_needed=8,
            temperature=0.3  # Lower for code
        ),
        ModelType.DEEPSEEK_14B: ModelConfig(
            name="DeepSeek R1 14B",
            size_gb=9.0,
            expected_speed="3-5 tok/s",
            context_length=32768,
            best_for="Complex reasoning (slow)",
            max_ram_needed=16,
            temperature=0.5
        ),
    }
    
    def __init__(
        self,
        default_model: ModelType = ModelType.LLAMA32_3B,
        enable_safety: bool = True,
        enable_rag: bool = False,
        enable_routing: bool = True,
        verbose: bool = True,
        check_ollama: bool = True
    ):
        """
        Initialize EdgeMind with Ollama backend
        
        Args:
            default_model: Default model to use
            enable_safety: Enable safety checks
            enable_rag: Enable RAG (if available)
            enable_routing: Route queries to best model
            verbose: Print status messages
            check_ollama: Check if Ollama is running
        """
        self.default_model = default_model
        self.current_model = default_model
        self.enable_routing = enable_routing
        self.verbose = verbose
        
        # Components
        self.safety_system = None
        self.rag_system = None
        
        # Conversation memory
        self.conversation_history = []
        self.max_history = 10
        
        # Performance tracking
        self.metrics = {
            "queries": 0,
            "total_time": 0,
            "avg_response_time": 0,
            "models_used": {}
        }
        
        # Initialize
        self._log("🧠 EdgeMind v0.4.0 - Realistic Local AI")
        self._log("=" * 50)
        
        # Check system resources
        self._check_system_resources()
        
        # Check Ollama
        if check_ollama:
            self._check_ollama()
        
        # Initialize safety
        if enable_safety and SAFETY_AVAILABLE:
            self._initialize_safety()
        elif enable_safety:
            self._log("⚠️ Safety module not found - basic keyword filtering only")
        
        # Initialize RAG
        if enable_rag and RAG_AVAILABLE:
            self._initialize_rag()
        
        self._log("✅ EdgeMind ready with Ollama backend!")
    
    def _log(self, message: str):
        """Log message if verbose"""
        if self.verbose:
            print(f"[EdgeMind] {message}")
    
    def _check_system_resources(self):
        """Check if system can run models"""
        ram_gb = psutil.virtual_memory().total / (1024**3)
        available_gb = psutil.virtual_memory().available / (1024**3)
        
        self._log(f"💻 System: {ram_gb:.1f}GB RAM ({available_gb:.1f}GB available)")
        
        # Warn about models that won't work
        for model_type, config in self.MODEL_CONFIGS.items():
            if config.max_ram_needed > available_gb:
                self._log(f"⚠️ {model_type.value} needs {config.max_ram_needed}GB RAM")
    
    def _check_ollama(self):
        """Check if Ollama is installed and running"""
        try:
            # Check if Ollama is running
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                self._log("✅ Ollama is running")
                
                # Parse available models
                lines = result.stdout.strip().split('\n')[1:]  # Skip header
                available_models = []
                
                for line in lines:
                    if line.strip():
                        model_name = line.split()[0]
                        # Check if it's one of our configured models
                        for model_type in ModelType:
                            if model_type.value in model_name:
                                available_models.append(model_type.value)
                
                if available_models:
                    self._log(f"📦 Available models: {', '.join(available_models)}")
                else:
                    self._log("⚠️ No compatible models found. Run: ollama pull llama3.2:3b")
            else:
                self._log("❌ Ollama error. Is it installed?")
                
        except FileNotFoundError:
            self._log("❌ Ollama not found. Install from: https://ollama.com")
        except subprocess.TimeoutExpired:
            self._log("⚠️ Ollama check timed out")
        except Exception as e:
            self._log(f"⚠️ Ollama check failed: {e}")
    
    def _initialize_safety(self):
        """Initialize safety system"""
        try:
            self.safety_system = SafeComputerControl()
            self._log("🔒 Safety system activated")
        except Exception as e:
            self._log(f"⚠️ Safety initialization failed: {e}")
    
    def _initialize_rag(self):
        """Initialize RAG system"""
        try:
            self.rag_system = SmartRAG()
            self._log("📚 RAG system initialized")
        except Exception as e:
            self._log(f"⚠️ RAG initialization failed: {e}")
    
    def _route_to_model(self, prompt: str) -> ModelType:
        """
        Route query to the best model based on content
        
        Args:
            prompt: User query
            
        Returns:
            Best model for this query
        """
        if not self.enable_routing:
            return self.current_model
        
        prompt_lower = prompt.lower()
        
        # Coding queries -> DeepSeek
        code_keywords = ['code', 'function', 'python', 'javascript', 'program', 
                        'debug', 'error', 'syntax', 'algorithm', 'implement']
        if any(keyword in prompt_lower for keyword in code_keywords):
            # Use 7B for most code, 14B only for complex
            if 'complex' in prompt_lower or 'algorithm' in prompt_lower:
                return ModelType.DEEPSEEK_14B
            return ModelType.DEEPSEEK_7B
        
        # Quick queries -> Phi-3
        quick_keywords = ['what is', 'when', 'where', 'who', 'define', 'meaning',
                         'simple', 'quick', 'briefly']
        if any(keyword in prompt_lower for keyword in quick_keywords):
            return ModelType.PHI3_MINI
        
        # Complex reasoning -> DeepSeek 14B
        complex_keywords = ['analyze', 'compare', 'evaluate', 'explain why',
                           'deep dive', 'comprehensive', 'detailed']
        if any(keyword in prompt_lower for keyword in complex_keywords):
            return ModelType.DEEPSEEK_14B
        
        # Default -> Llama 3.2 (balanced)
        return ModelType.LLAMA32_3B
    
    def _safety_check(self, prompt: str) -> tuple[bool, str]:
        """
        Check if prompt is safe
        
        Returns:
            (is_safe, message)
        """
        # First use SafeComputerControl if available
        if self.safety_system:
            try:
                if not self.safety_system.is_safe(prompt):
                    return False, "Request blocked by safety system"
            except Exception as e:
                self._log(f"⚠️ Safety check error: {e}")
        
        # Fallback to keyword filtering
        dangerous_keywords = [
            'bomb', 'explosive', 'weapon', 'terrorism', 'kill', 'murder',
            'suicide', 'meth', 'drugs', 'poison', 'hack passwords', 'malware'
        ]
        
        prompt_lower = prompt.lower()
        for keyword in dangerous_keywords:
            if keyword in prompt_lower:
                return False, f"Cannot provide information about: {keyword}"
        
        return True, "Safe"
    
    def generate(
        self,
        prompt: str,
        model: Optional[ModelType] = None,
        max_tokens: int = 256,
        temperature: Optional[float] = None,
        use_history: bool = True,
        safety_check: bool = True,
        stream: bool = False
    ) -> str:
        """
        Generate response using Ollama
        
        Args:
            prompt: User query
            model: Override model selection
            max_tokens: Max tokens to generate
            temperature: Override temperature
            use_history: Include conversation history
            safety_check: Check safety
            stream: Stream response
            
        Returns:
            Generated response
        """
        start_time = time.time()
        
        # Safety check
        if safety_check:
            is_safe, safety_msg = self._safety_check(prompt)
            if not is_safe:
                self._log(f"🔒 {safety_msg}")
                return f"I cannot provide that information. {safety_msg}"
        
        # Route to best model
        if model is None:
            model = self._route_to_model(prompt)
            if model != self.current_model:
                self._log(f"📍 Routing to {model.value}")
        
        # Get model config
        config = self.MODEL_CONFIGS[model]
        
        # Use model's default temperature if not specified
        if temperature is None:
            temperature = config.temperature
        
        # Add conversation history if enabled
        if use_history and self.conversation_history:
            context = "\n".join([
                f"{msg['role']}: {msg['content']}" 
                for msg in self.conversation_history[-self.max_history:]
            ])
            full_prompt = f"Previous conversation:\n{context}\n\nUser: {prompt}\nAssistant:"
        else:
            full_prompt = prompt
        
        # Generate with Ollama
        try:
            if OLLAMA_PYTHON:
                # Use Python library if available
                response = ollama.generate(
                    model=model.value,
                    prompt=full_prompt,
                    options={
                        'temperature': temperature,
                        'num_predict': max_tokens,
                        'top_k': 40,
                        'top_p': 0.95,
                        'repeat_penalty': 1.1
                    },
                    stream=stream
                )
                
                if stream:
                    # Return generator for streaming
                    return response
                else:
                    result = response['response']
            else:
                # Use subprocess
                cmd = [
                    "ollama", "run", model.value,
                    "--verbose",
                    full_prompt
                ]
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                
                if result.returncode != 0:
                    raise Exception(f"Ollama error: {result.stderr}")
                
                result = result.stdout.strip()
            
            # Clean up DeepSeek's thinking tags if present
            if '<think>' in result:
                # Remove thinking process from output
                import re
                result = re.sub(r'<think>.*?</think>', '', result, flags=re.DOTALL)
                result = result.strip()
            
            # Update conversation history
            if use_history:
                self.conversation_history.append({"role": "user", "content": prompt})
                self.conversation_history.append({"role": "assistant", "content": result})
                
                # Trim history
                if len(self.conversation_history) > self.max_history * 2:
                    self.conversation_history = self.conversation_history[-(self.max_history * 2):]
            
            # Update metrics
            elapsed = time.time() - start_time
            self.metrics["queries"] += 1
            self.metrics["total_time"] += elapsed
            self.metrics["avg_response_time"] = self.metrics["total_time"] / self.metrics["queries"]
            
            # Track model usage
            model_name = model.value
            if model_name not in self.metrics["models_used"]:
                self.metrics["models_used"][model_name] = 0
            self.metrics["models_used"][model_name] += 1
            
            # Log performance
            tokens_estimate = len(result.split())
            speed = tokens_estimate / elapsed if elapsed > 0 else 0
            self._log(f"⚡ {tokens_estimate} tokens in {elapsed:.1f}s ({speed:.1f} tok/s)")
            
            return result
            
        except subprocess.TimeoutExpired:
            return "Response timed out. Try a smaller model or shorter prompt."
        except Exception as e:
            self._log(f"❌ Generation error: {e}")
            return f"Error generating response: {e}"
    
    def chat(self):
        """Interactive chat mode with model switching"""
        print("""
💬 EdgeMind Chat v0.4.0 - Realistic Local AI
============================================
Models available:
  📘 phi3:mini - Fast responses (2.2GB)
  📗 llama3.2:3b - Balanced (2.0GB) [DEFAULT]
  📙 deepseek-r1:7b - Best for code (4.7GB)
  📕 deepseek-r1:14b - Complex tasks (9.0GB, slow)

Commands:
  /model <name>  - Switch model
  /route on/off  - Toggle smart routing
  /safety on/off - Toggle safety checks
  /clear        - Clear conversation
  /metrics      - Show performance stats
  /help         - Show this help
  /quit         - Exit

Smart routing is {'ON' if self.enable_routing else 'OFF'}
Safety checks are {'ON' if self.safety_system else 'OFF'}
        """)
        
        while True:
            try:
                user_input = input("\n👤 You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.lower() == '/quit':
                    break
                
                elif user_input.lower() == '/clear':
                    self.conversation_history = []
                    print("🗑️ Conversation cleared")
                
                elif user_input.lower() == '/metrics':
                    print("\n📊 Performance Metrics:")
                    print(f"  Queries: {self.metrics['queries']}")
                    print(f"  Avg Response Time: {self.metrics['avg_response_time']:.1f}s")
                    print("  Models Used:")
                    for model, count in self.metrics['models_used'].items():
                        print(f"    {model}: {count} times")
                
                elif user_input.lower() == '/help':
                    self.chat()  # Show help again
                    return
                
                elif user_input.startswith('/model'):
                    parts = user_input.split(maxsplit=1)
                    if len(parts) > 1:
                        model_name = parts[1]
                        # Find matching model
                        for model_type in ModelType:
                            if model_name in model_type.value:
                                self.current_model = model_type
                                print(f"✅ Switched to {model_type.value}")
                                break
                        else:
                            print(f"❌ Unknown model: {model_name}")
                    else:
                        print("Available models:")
                        for mt in ModelType:
                            print(f"  - {mt.value}")
                
                elif user_input.startswith('/route'):
                    if 'off' in user_input.lower():
                        self.enable_routing = False
                        print("🔀 Routing disabled")
                    else:
                        self.enable_routing = True
                        print("🔀 Smart routing enabled")
                
                elif user_input.startswith('/safety'):
                    if 'off' in user_input.lower():
                        self.safety_system = None
                        print("⚠️ Safety checks disabled")
                    else:
                        if SAFETY_AVAILABLE:
                            self._initialize_safety()
                        print("🔒 Safety checks enabled")
                
                else:
                    # Generate response
                    response = self.generate(
                        user_input,
                        use_history=True,
                        safety_check=(self.safety_system is not None)
                    )
                    
                    print(f"\n🤖 EdgeMind: {response}")
                    
                    # Show which model was used
                    if self.enable_routing:
                        used_model = self._route_to_model(user_input)
                        if used_model != self.default_model:
                            print(f"    [Used: {used_model.value}]")
            
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
        
        print("\n📊 Session Summary:")
        print(f"  Total queries: {self.metrics['queries']}")
        if self.metrics['queries'] > 0:
            print(f"  Avg response time: {self.metrics['avg_response_time']:.1f}s")
    
    def benchmark(self, models: Optional[List[ModelType]] = None):
        """
        Benchmark available models
        
        Args:
            models: List of models to test (None = all available)
        """
        if models is None:
            models = [ModelType.PHI3_MINI, ModelType.LLAMA32_3B, ModelType.DEEPSEEK_7B]
        
        test_prompts = [
            "What is 2+2?",
            "Write a Python hello world",
            "Explain quantum computing in one sentence"
        ]
        
        print("\n🏃 Running Benchmark...")
        print("=" * 60)
        
        results = {}
        
        for model in models:
            print(f"\nTesting {model.value}...")
            model_results = []
            
            for prompt in test_prompts:
                try:
                    start = time.time()
                    response = self.generate(
                        prompt,
                        model=model,
                        max_tokens=50,
                        use_history=False,
                        safety_check=False
                    )
                    elapsed = time.time() - start
                    
                    tokens = len(response.split())
                    speed = tokens / elapsed if elapsed > 0 else 0
                    
                    model_results.append({
                        'prompt': prompt[:30] + '...',
                        'time': elapsed,
                        'tokens': tokens,
                        'speed': speed
                    })
                    
                    print(f"  ✅ {elapsed:.1f}s ({speed:.1f} tok/s)")
                    
                except Exception as e:
                    print(f"  ❌ Failed: {e}")
                    model_results.append({
                        'prompt': prompt[:30] + '...',
                        'time': 0,
                        'tokens': 0,
                        'speed': 0
                    })
            
            # Calculate averages
            avg_time = sum(r['time'] for r in model_results) / len(model_results)
            avg_speed = sum(r['speed'] for r in model_results) / len(model_results)
            
            results[model.value] = {
                'avg_time': avg_time,
                'avg_speed': avg_speed,
                'config': self.MODEL_CONFIGS[model]
            }
        
        # Show summary
        print("\n" + "=" * 60)
        print("📊 Benchmark Results:")
        print("=" * 60)
        
        for model_name, result in results.items():
            config = result['config']
            print(f"\n{model_name}:")
            print(f"  Size: {config.size_gb}GB")
            print(f"  Avg Time: {result['avg_time']:.1f}s")
            print(f"  Avg Speed: {result['avg_speed']:.1f} tok/s")
            print(f"  Expected: {config.expected_speed}")
            print(f"  Best For: {config.best_for}")
        
        return results


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="EdgeMind v0.4.0 - Realistic Local AI")
    parser.add_argument("--chat", action="store_true", help="Start chat mode")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark")
    parser.add_argument("--prompt", type=str, help="Single prompt")
    parser.add_argument("--model", type=str, help="Model to use")
    parser.add_argument("--no-safety", action="store_true", help="Disable safety")
    parser.add_argument("--no-routing", action="store_true", help="Disable routing")
    
    args = parser.parse_args()
    
    # Header
    print("\n" + "=" * 60)
    print("🧠 EdgeMind v0.4.0 - Realistic Local AI for Laptops")
    print("=" * 60)
    
    # Initialize
    edgemind = EdgeMind(
        enable_safety=not args.no_safety,
        enable_routing=not args.no_routing
    )
    
    if args.benchmark:
        edgemind.benchmark()
    elif args.chat:
        edgemind.chat()
    elif args.prompt:
        # Override model if specified
        model = None
        if args.model:
            for mt in ModelType:
                if args.model in mt.value:
                    model = mt
                    break
        
        response = edgemind.generate(args.prompt, model=model)
        print(f"\n📝 Prompt: {args.prompt}")
        print(f"🤖 Response: {response}")
    else:
        # Demo mode
        print("\n🎯 Quick Demo:")
        demo_prompts = [
            "What is artificial intelligence?",
            "Write a Python function to reverse a string",
            "Is it safe to mix bleach and ammonia?"  # Safety test
        ]
        
        for prompt in demo_prompts:
            print(f"\n📝 {prompt}")
            response = edgemind.generate(prompt, max_tokens=100)
            print(f"🤖 {response[:200]}...")
        
        print("\n💡 Tips:")
        print("  Run with --chat for interactive mode")
        print("  Run with --benchmark to test all models")
        print("  Run with --prompt 'your question' for single query")


if __name__ == "__main__":
    main()