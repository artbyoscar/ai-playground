"""
Benchmark and optimize EdgeMind performance
"""
import time
from src.core.edgemind import EdgeMind, ModelType

def benchmark_all_models():
    em = EdgeMind(verbose=False)
    
    prompts = [
        "What is Python?",
        "Write a function",
        "Explain AI"
    ]
    
    for model in [ModelType.PHI3_MINI, ModelType.LLAMA32_3B]:
        total_time = 0
        for prompt in prompts:
            start = time.time()
            em.generate(prompt, model=model)
            total_time += time.time() - start
        
        print(f"{model.value}: {total_time/len(prompts):.2f}s avg")