# test_performance.py
from src.core.chain_of_thought_engine import ChainOfThoughtEngine
from src.core.working_ai_playground import AIPlayground
import time

ai = AIPlayground()
cot = ChainOfThoughtEngine(ai)

# Benchmark different strategies
strategies = ['zero_shot', 'chain_of_drafts', 'reflexion']
prompt = "How can we make AI inference 10x faster on edge devices?"

for strategy in strategies:
    start = time.time()
    result = cot.think(prompt, strategy=strategy, stream=False)
    elapsed = time.time() - start
    
    print(f"\n{strategy}:")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Confidence: {result['confidence']:.2f}")
    print(f"  Thoughts: {result['num_thoughts']}")