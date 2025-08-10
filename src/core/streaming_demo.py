"""
Streaming responses for better user experience
Shows text as it's generated instead of waiting
"""

from src.core.edgemind import EdgeMind
import sys

def stream_demo():
    em = EdgeMind(verbose=False)
    
    prompt = "Explain how a computer works in 3 paragraphs"
    
    print("🤖 EdgeMind: ", end="", flush=True)
    
    # Stream the response
    response_generator = em.generate(
        prompt, 
        stream=True,
        max_tokens=200
    )
    
    # Print each token as it arrives
    for chunk in response_generator:
        if 'response' in chunk:
            print(chunk['response'], end="", flush=True)
    
    print("\n\n✅ Streaming complete!")

if __name__ == "__main__":
    # First install: pip install ollama
    stream_demo()