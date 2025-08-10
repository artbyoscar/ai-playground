"""
AI-powered git commit message generator
"""
from src.core.edgemind import EdgeMind
import subprocess

def generate_commit_message():
    # Get git diff
    diff = subprocess.run(['git', 'diff', '--staged'], 
                         capture_output=True, text=True).stdout
    
    em = EdgeMind(verbose=False)
    prompt = f"Generate a conventional commit message for:\n{diff[:500]}"
    
    message = em.generate(prompt, max_tokens=50)
    print(f"Suggested: {message}")
    
    if input("Use this? (y/n): ").lower() == 'y':
        subprocess.run(['git', 'commit', '-m', message])