# run_real_edgemind.py
"""
This is the REAL EdgeMind - Not a wrapper, but actual local AI with computer control
Run this to see what EdgeMind is supposed to be
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core.local_llm_engine import LocalLLMEngine, EdgeFormerCompressor
from src.agents.autonomous_agent import AutonomousAgent, EdgeComputeOptimizer
import time
import psutil

def main():
    print("="*80)
    print("🚀 EDGEMIND v0.3.0 - THE REAL DEAL")
    print("Local AI + Computer Control + Edge Optimization")
    print("="*80)
    
    # 1. Profile the device
    print("\n📊 DEVICE PROFILE:")
    profile = EdgeComputeOptimizer.profile_device()
    for key, value in profile.items():
        print(f"   {key}: {value}")
    
    # 2. Check what we can run
    ram_gb = psutil.virtual_memory().available / (1024**3)
    
    if ram_gb < 2:
        model = "tinyllama"
        print(f"\n⚠️ Low RAM ({ram_gb:.1f}GB) - Using TinyLlama")
    elif ram_gb < 6:
        model = "phi-2"
        print(f"\n✅ Medium RAM ({ram_gb:.1f}GB) - Using Phi-2")
    else:
        model = "mistral-7b-instruct"
        print(f"\n🚀 Good RAM ({ram_gb:.1f}GB) - Using Mistral 7B")
    
    # 3. Initialize LOCAL AI (no API!)
    print("\n🧠 INITIALIZING LOCAL AI ENGINE...")
    llm = LocalLLMEngine(model_name=model)
    
    # Download and setup
    if not llm.model_path.exists():
        print("\n📥 First-time setup - downloading model...")
        print("   This is a one-time download")
        if not llm.setup():
            print("❌ Setup failed. Try a smaller model.")
            return
    
    # 4. Initialize Autonomous Agent
    print("\n🤖 INITIALIZING AUTONOMOUS AGENT...")
    agent = AutonomousAgent(safety_mode=True)
    
    # 5. Demonstrate the REAL EdgeMind
    print("\n" + "🔥"*20)
    print("DEMONSTRATION: What EdgeMind Can ACTUALLY Do")
    print("🔥"*20)
    
    # Test 1: Local inference speed
    print("\n📏 TEST 1: Local Inference Speed")
    print("-"*40)
    
    start = time.time()
    result = llm.run_inference(
        "Write a Python function to calculate fibonacci numbers",
        max_tokens=100
    )
    elapsed = time.time() - start
    
    print(f"⏱️ Inference Time: {elapsed:.2f}s")
    print(f"💨 Tokens/Second: {result.get('tokens_per_second', 0):.1f}")
    print(f"💰 Cost: $0.00 (IT'S LOCAL!)")
    print(f"\n📝 Response:\n{result['response'][:200]}...")
    
    # Test 2: System control
    print("\n🎮 TEST 2: Computer Control Capabilities")
    print("-"*40)
    
    system_info = agent.get_system_info()
    print(f"🖥️ CPU Usage: {system_info['cpu_percent']}%")
    print(f"🧠 RAM Available: {system_info['ram_available_gb']:.1f}GB")
    print(f"💾 Disk Usage: {system_info['disk_usage_percent']}%")
    
    # Test 3: Combined - AI making decisions about system
    print("\n🎯 TEST 3: AI-Driven System Decisions")
    print("-"*40)
    
    prompt = f"""System Status:
    - CPU: {system_info['cpu_percent']}%
    - RAM: {system_info['ram_available_gb']:.1f}GB available
    - Disk: {system_info['disk_usage_percent']}% used
    
    Should we run a heavy computation task now? Answer yes or no with reasoning."""
    
    decision = llm.run_inference(prompt, max_tokens=50)
    print(f"🤖 AI Decision: {decision['response']}")
    
    # Show the difference
    print("\n" + "="*60)
    print("📊 COMPARISON: EdgeMind vs ChatGPT")
    print("="*60)
    
    comparison = """
    | Feature              | EdgeMind (Local) | ChatGPT (Cloud) |
    |---------------------|------------------|-----------------|
    | Monthly Cost        | $0               | $20-200         |
    | Privacy            | 100% Local       | Data to OpenAI  |
    | Internet Required   | No               | Yes             |
    | Computer Control    | Yes              | No              |
    | Response Time       | ~2s              | ~1s + network   |
    | Rate Limits         | None             | Yes             |
    | Customizable        | Yes              | No              |
    | Works Offline       | Yes              | No              |
    """
    print(comparison)
    
    print("\n✅ THIS is EdgeMind - Local AI that can control your computer!")
    print("🚀 No API calls, no cloud, no monthly fees - just YOUR hardware!")
    
    # Interactive demo
    print("\n" + "="*60)
    print("💬 INTERACTIVE DEMO - Chat with LOCAL AI")
    print("="*60)
    print("Type 'quit' to exit, 'action' for computer control demo")
    
    while True:
        user_input = input("\nYou: ")
        
        if user_input.lower() == 'quit':
            break
        
        elif user_input.lower() == 'action':
            print("\n🤖 Computer Control Actions Available:")
            print("1. Take screenshot")
            print("2. Open notepad")
            print("3. System report")
            print("4. Search web")
            
            choice = input("Choose action (1-4): ")
            
            if choice == '1':
                agent.take_screenshot()
                print("✅ Screenshot taken!")
            elif choice == '2':
                agent.open_application("notepad.exe")
            elif choice == '3':
                info = agent.get_system_info()
                for k, v in info.items():
                    if v: print(f"  {k}: {v}")
            elif choice == '4':
                query = input("Search for: ")
                agent.search_web(query)
        
        else:
            # Local AI response
            print("\n🤖 EdgeMind (LOCAL): ", end="", flush=True)
            response = llm.run_inference(
                f"User: {user_input}\nAssistant:",
                max_tokens=150
            )
            print(response['response'])
            print(f"\n  ⚡ {response.get('tokens_per_second', 0):.1f} tokens/sec | 💰 Cost: $0.00")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 EdgeMind shutting down...")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 Try installing requirements:")
        print("   pip install llama-cpp-python pyautogui psutil torch")