# gpt_oss_integration.py
"""
Integration with OpenAI's GPT-OSS open models
This is REVOLUTIONARY - OpenAI's models running locally!
"""

import subprocess
import os
from pathlib import Path

class GPTOSSEngine:
    """
    Run OpenAI's open models locally
    These are OFFICIAL OpenAI models with Apache 2.0 license!
    """
    
    def setup_ollama(self):
        """Setup GPT-OSS with Ollama (easiest method)"""
        print("🚀 Setting up OpenAI GPT-OSS models...")
        
        commands = [
            # Install Ollama if not installed
            "winget install Ollama.Ollama",
            
            # Pull GPT-OSS 20B (fits in 16GB RAM!)
            "ollama pull gpt-oss:20b",
            
            # Run it
            "ollama run gpt-oss:20b"
        ]
        
        for cmd in commands:
            print(f"Running: {cmd}")
            subprocess.run(cmd, shell=True)
    
    def setup_transformers(self):
        """Use with Transformers (what we already have!)"""
        
        # Install requirements
        subprocess.run("pip install transformers accelerate", shell=True)
        
        # Use GPT-OSS with our existing setup!
        from transformers import pipeline
        import torch
        
        # This is an OFFICIAL OpenAI model!
        model_id = "openai/gpt-oss-20b"
        
        pipe = pipeline(
            "text-generation",
            model=model_id,
            torch_dtype="auto",
            device_map="auto",
        )
        
        # Test it
        messages = [
            {"role": "user", "content": "What makes you different from GPT-4?"},
        ]
        
        outputs = pipe(messages, max_new_tokens=256)
        print(outputs[0]["generated_text"][-1])
        
        return pipe
    
    def download_weights(self):
        """Download the actual weights"""
        print("📥 Downloading OpenAI GPT-OSS weights...")
        
        # Use HuggingFace CLI
        commands = [
            # GPT-OSS 20B (21B parameters, fits in 16GB!)
            "huggingface-cli download openai/gpt-oss-20b --include 'original/*' --local-dir gpt-oss-20b/",
            
            # If you have 80GB GPU, get the big one
            # "huggingface-cli download openai/gpt-oss-120b --include 'original/*' --local-dir gpt-oss-120b/"
        ]
        
        for cmd in commands:
            subprocess.run(cmd, shell=True)


# Quick test
if __name__ == "__main__":
    print("="*60)
    print("🔥 OpenAI GPT-OSS Integration")
    print("="*60)
    print("\nOpenAI just released OPEN models!")
    print("- Apache 2.0 license (fully open)")
    print("- Native 4-bit quantization") 
    print("- Runs on consumer hardware")
    print("- Full reasoning capabilities")
    print("\nThis changes EVERYTHING for EdgeMind!")
    
    engine = GPTOSSEngine()
    
    print("\nChoose setup method:")
    print("1. Ollama (easiest)")
    print("2. Transformers (integrate with our code)")
    print("3. Download raw weights")
    
    choice = input("\nChoice (1-3): ")
    
    if choice == "1":
        engine.setup_ollama()
    elif choice == "2":
        engine.setup_transformers()
    elif choice == "3":
        engine.download_weights()