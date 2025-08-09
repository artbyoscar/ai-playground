# edgemind_demo.py
"""
EdgeMind Demo - Shows REAL local AI working
Run this to see the difference
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core.local_llm_engine_fixed import LocalLLMEngine
import time
import psutil

def main():
    print("="*70)
    print("🧠 EDGEMIND - LOCAL AI DEMO")
    print("="*70)
    print("\nThis runs 100% on YOUR computer. No internet needed after download.\n")
    
    # Check available RAM
    available_gb = psutil.virtual_memory().available / (1024**3)
    
    # Choose model based on RAM
    if available_gb < 3:
        model = "tinyllama"
        print(f"💻 Using TinyLlama (670MB) - perfect for {available_gb:.1f}GB RAM")
    elif available_gb < 6:
        model = "phi-2"
        print(f"💻 Using Phi-2 (1.6GB) - good for {available_gb:.1f}GB RAM")
    else:
        model = "mistral"
        print(f"💻 Using Mistral 7B (4.1GB) - best quality for {available_gb:.1f}GB RAM")
    
    # Initialize
    print("\n🚀 Initializing EdgeMind...")
    engine = LocalLLMEngine(model)
    
    # Download if needed (one-time)
    if not engine.model_path.exists():
        print("\n📥 First-time setup - downloading model...")
        print("   (This only happens once)")
        if not engine.download_model():
            print("❌ Download failed. Check connection.")
            return
    
    # Load model
    print("\n🧠 Loading AI model...")
    if not engine.load_model():
        print("❌ Failed to load. Try smaller model.")
        return
    
    print("\n" + "="*70)
    print("✅ EDGEMIND IS READY - 100% LOCAL AI")
    print("="*70)
    print("\nType 'quit' to exit")
    print("Type 'benchmark' to test speed")
    print("Type 'compare' to see vs ChatGPT")
    print("\n")
    
    while True:
        user_input = input("👤 You: ").strip()
        
        if user_input.lower() == 'quit':
            break
        
        elif user_input.lower() == 'benchmark':
            engine.benchmark()
            continue
        
        elif user_input.lower() == 'compare':
            print("\n📊 EdgeMind vs ChatGPT:")
            print("┌─────────────────┬──────────────┬──────────────┐")
            print("│ Feature         │ EdgeMind     │ ChatGPT      │")
            print("├─────────────────┼──────────────┼──────────────┤")
            print("│ Monthly Cost    │ $0           │ $20+         │")
            print("│ Privacy         │ 100% Local   │ Cloud        │")
            print("│ Internet Needed │ No           │ Yes          │")
            print("│ Rate Limits     │ None         │ Yes          │")
            print("│ Customizable    │ Yes          │ No           │")
            print("│ Speed           │ 5-20 tok/s   │ 30-50 tok/s  │")
            print("└─────────────────┴──────────────┴──────────────┘")
            continue
        
        # Generate response
        print(f"\n🤖 EdgeMind: ", end="", flush=True)
        
        start = time.time()
        response = engine.chat(user_input)
        elapsed = time.time() - start
        
        print(response)
        
        # Show stats
        result = engine.generate(user_input, max_tokens=100)
        if "tokens_per_second" in result:
            print(f"\n   ⚡ {result['tokens_per_second']:.1f} tokens/sec")
            print(f"   ⏱️  {elapsed:.2f} seconds")
            print(f"   💰 $0.00 (it's local!)")
    
    print("\n👋 Thanks for trying EdgeMind!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")