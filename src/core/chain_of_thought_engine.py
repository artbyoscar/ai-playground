# src/core/chain_of_thought_engine.py
"""
Chain of Thought/Draft Reasoning Engine
Optimized for local compute with intelligent caching and streaming
"""
import json
import time
import hashlib
from typing import List, Dict, Any, Optional, Generator
from dataclasses import dataclass
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor

@dataclass
class ThoughtNode:
    """Represents a single thought in the reasoning chain"""
    id: str
    content: str
    confidence: float
    reasoning_type: str  # 'analysis', 'synthesis', 'evaluation', 'revision'
    parent_id: Optional[str] = None
    children: List[str] = None
    metadata: Dict[str, Any] = None
    cached: bool = False
    
    def __post_init__(self):
        if self.children is None:
            self.children = []
        if self.metadata is None:
            self.metadata = {}

class ChainOfThoughtEngine:
    """
    Advanced reasoning engine optimized for local compute
    Uses caching, streaming, and parallel processing for efficiency
    """
    
    def __init__(self, ai_playground=None, cache_dir="data/thought_cache"):
        self.ai = ai_playground
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Performance optimizations
        self.max_parallel_thoughts = 3
        self.thought_cache = {}
        self.executor = ThreadPoolExecutor(max_workers=self.max_parallel_thoughts)
        
        # Reasoning strategies
        self.strategies = {
            'zero_shot': self._zero_shot_cot,
            'few_shot': self._few_shot_cot,
            'self_consistency': self._self_consistency_cot,
            'tree_of_thoughts': self._tree_of_thoughts,
            'chain_of_drafts': self._chain_of_drafts,
            'reflexion': self._reflexion_cot
        }
        
        # Compression patterns for local compute
        self.compression_templates = {
            'analysis': "Analyze: {input}\nKey points (brief):",
            'synthesis': "Combine: {inputs}\nSynthesis:",
            'evaluation': "Evaluate: {input}\nScore and reason:",
            'revision': "Revise: {input}\nImprovement:"
        }
    
    def think(self, 
              prompt: str, 
              strategy: str = 'zero_shot',
              max_thoughts: int = 5,
              stream: bool = True) -> Generator[ThoughtNode, None, Dict]:
        """
        Main thinking interface with streaming support
        
        Args:
            prompt: The problem to solve
            strategy: Reasoning strategy to use
            max_thoughts: Maximum thoughts to generate
            stream: Whether to stream thoughts as they're generated
            
        Yields:
            ThoughtNode objects as they're generated
            
        Returns:
            Final solution dictionary
        """
        # Check cache first
        cache_key = self._get_cache_key(prompt, strategy)
        if cache_key in self.thought_cache:
            print("💾 Using cached reasoning chain")
            cached_result = self.thought_cache[cache_key]
            if stream:
                for thought in cached_result['thoughts']:
                    yield thought
            return cached_result['solution']
        
        # Generate new reasoning chain
        print(f"🧠 Thinking with strategy: {strategy}")
        strategy_func = self.strategies.get(strategy, self._zero_shot_cot)
        
        thoughts = []
        thought_generator = strategy_func(prompt, max_thoughts)
        
        for thought in thought_generator:
            thoughts.append(thought)
            if stream:
                yield thought
        
        # Generate final solution
        solution = self._synthesize_solution(thoughts, prompt)
        
        # Cache the result
        result = {
            'thoughts': thoughts,
            'solution': solution
        }
        self.thought_cache[cache_key] = result
        self._save_to_disk_cache(cache_key, result)
        
        return solution
    
    def _zero_shot_cot(self, prompt: str, max_thoughts: int) -> Generator[ThoughtNode, None, None]:
        """Zero-shot chain of thought reasoning"""
        
        # Initial problem decomposition
        decomposition_prompt = f"""
        Break down this problem into logical steps (be concise):
        {prompt}
        
        Steps (numbered):
        """
        
        if self.ai:
            steps = self._compressed_inference(decomposition_prompt)
        else:
            steps = "1. Understand the problem\n2. Identify key components\n3. Develop solution\n4. Verify"
        
        # Process each step
        step_lines = [s.strip() for s in steps.split('\n') if s.strip()][:max_thoughts]
        
        for i, step in enumerate(step_lines):
            thought_id = f"thought_{i}_{hash(step)}"
            
            # Generate reasoning for this step
            reasoning_prompt = f"""
            Step: {step}
            Context: {prompt}
            
            Brief reasoning:
            """
            
            if self.ai:
                reasoning = self._compressed_inference(reasoning_prompt)
            else:
                reasoning = f"Processing: {step}"
            
            thought = ThoughtNode(
                id=thought_id,
                content=reasoning,
                confidence=0.8 - (i * 0.1),  # Decrease confidence as we go deeper
                reasoning_type='analysis'
            )
            
            yield thought
    
    def _few_shot_cot(self, prompt: str, max_thoughts: int) -> Generator[ThoughtNode, None, None]:
        """Few-shot learning with examples"""
        
        # Load relevant examples from cache
        examples = self._get_relevant_examples(prompt, limit=3)
        
        example_text = "\n".join([
            f"Example {i+1}:\nProblem: {ex['problem']}\nSolution: {ex['solution']}"
            for i, ex in enumerate(examples)
        ])
        
        enhanced_prompt = f"""
        {example_text}
        
        Now solve:
        {prompt}
        
        Step-by-step (brief):
        """
        
        # Generate thoughts with examples
        for thought in self._zero_shot_cot(enhanced_prompt, max_thoughts):
            thought.metadata['strategy'] = 'few_shot'
            yield thought
    
    def _self_consistency_cot(self, prompt: str, max_thoughts: int) -> Generator[ThoughtNode, None, None]:
        """Generate multiple reasoning paths and vote"""
        
        paths = []
        num_paths = min(3, self.max_parallel_thoughts)
        
        # Generate multiple reasoning paths in parallel
        futures = []
        for i in range(num_paths):
            future = self.executor.submit(self._generate_reasoning_path, prompt, i)
            futures.append(future)
        
        # Collect all paths
        for future in futures:
            path = future.result()
            paths.append(path)
            
            # Yield thoughts from this path
            for thought in path:
                thought.metadata['path_id'] = len(paths) - 1
                yield thought
        
        # Vote on best path
        best_path = self._vote_on_paths(paths)
        
        # Yield consensus thought
        consensus = ThoughtNode(
            id=f"consensus_{hash(prompt)}",
            content=f"Consensus from {num_paths} reasoning paths",
            confidence=0.9,
            reasoning_type='synthesis',
            metadata={'selected_path': best_path}
        )
        yield consensus
    
    def _tree_of_thoughts(self, prompt: str, max_thoughts: int) -> Generator[ThoughtNode, None, None]:
        """Tree-based exploration of solution space"""
        
        root = ThoughtNode(
            id="root",
            content=prompt,
            confidence=1.0,
            reasoning_type='analysis'
        )
        yield root
        
        # BFS exploration
        queue = [root]
        thought_count = 1
        
        while queue and thought_count < max_thoughts:
            current = queue.pop(0)
            
            # Generate children thoughts
            children = self._generate_child_thoughts(current, max_children=2)
            
            for child in children:
                if thought_count >= max_thoughts:
                    break
                    
                current.children.append(child.id)
                child.parent_id = current.id
                
                yield child
                queue.append(child)
                thought_count += 1
    
    def _chain_of_drafts(self, prompt: str, max_thoughts: int) -> Generator[ThoughtNode, None, None]:
        """Iterative refinement through drafts"""
        
        current_draft = f"Initial solution to: {prompt}"
        
        for i in range(min(max_thoughts, 5)):
            # Generate draft
            draft_prompt = f"""
            Problem: {prompt}
            Current draft: {current_draft}
            
            Improve this draft (concise):
            """
            
            if self.ai:
                improved = self._compressed_inference(draft_prompt)
            else:
                improved = f"Draft {i+1}: Refined solution"
            
            thought = ThoughtNode(
                id=f"draft_{i}",
                content=improved,
                confidence=0.6 + (i * 0.08),  # Increase confidence with iterations
                reasoning_type='revision',
                metadata={'draft_number': i+1}
            )
            
            current_draft = improved
            yield thought
    
    def _reflexion_cot(self, prompt: str, max_thoughts: int) -> Generator[ThoughtNode, None, None]:
        """Self-reflection and correction"""
        
        # Initial attempt
        initial = ThoughtNode(
            id="initial",
            content=f"Initial approach to: {prompt}",
            confidence=0.7,
            reasoning_type='analysis'
        )
        yield initial
        
        # Reflection loop
        for i in range(min(max_thoughts - 1, 3)):
            # Reflect on previous thought
            reflection_prompt = f"""
            Problem: {prompt}
            Previous approach: {initial.content}
            
            What could be wrong or improved? (brief):
            """
            
            if self.ai:
                reflection = self._compressed_inference(reflection_prompt)
            else:
                reflection = f"Reflection {i+1}: Identifying improvements"
            
            reflect_thought = ThoughtNode(
                id=f"reflection_{i}",
                content=reflection,
                confidence=0.75,
                reasoning_type='evaluation',
                parent_id=initial.id
            )
            yield reflect_thought
            
            # Generate improved solution
            improvement_prompt = f"""
            Based on reflection: {reflection}
            Improved approach (brief):
            """
            
            if self.ai:
                improved = self._compressed_inference(improvement_prompt)
            else:
                improved = f"Improved solution {i+1}"
            
            improve_thought = ThoughtNode(
                id=f"improvement_{i}",
                content=improved,
                confidence=0.8 + (i * 0.05),
                reasoning_type='revision',
                parent_id=reflect_thought.id
            )
            yield improve_thought
            
            initial = improve_thought
    
    def _compressed_inference(self, prompt: str, max_tokens: int = 150) -> str:
        """
        Optimized inference for local compute
        Uses compression and caching
        """
        # Check inference cache
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()[:8]
        cache_file = self.cache_dir / f"inference_{prompt_hash}.txt"
        
        if cache_file.exists():
            return cache_file.read_text()
        
        # Compress prompt for efficiency
        compressed_prompt = self._compress_prompt(prompt)
        
        # Run inference
        if self.ai:
            # Use the most efficient assistant for reasoning
            response = self.ai.code_assistant(compressed_prompt)
            # Truncate for efficiency
            response = response[:max_tokens]
        else:
            response = "Simulated response"
        
        # Cache the response
        cache_file.write_text(response)
        
        return response
    
    def _compress_prompt(self, prompt: str) -> str:
        """Compress prompt to reduce tokens"""
        # Remove redundant whitespace
        compressed = ' '.join(prompt.split())
        
        # Use abbreviations
        abbreviations = {
            'problem': 'prob',
            'solution': 'sol',
            'reasoning': 'reason',
            'analysis': 'analyze',
            'synthesis': 'synth'
        }
        
        for full, abbrev in abbreviations.items():
            compressed = compressed.replace(full, abbrev)
        
        return compressed
    
    def _generate_reasoning_path(self, prompt: str, path_id: int) -> List[ThoughtNode]:
        """Generate a single reasoning path"""
        thoughts = []
        
        # Add randomization for diversity
        import random
        num_steps = random.randint(3, 5)
        
        for i in range(num_steps):
            thought = ThoughtNode(
                id=f"path_{path_id}_thought_{i}",
                content=f"Path {path_id} step {i+1}",
                confidence=random.uniform(0.6, 0.9),
                reasoning_type=random.choice(['analysis', 'synthesis', 'evaluation'])
            )
            thoughts.append(thought)
        
        return thoughts
    
    def _generate_child_thoughts(self, parent: ThoughtNode, max_children: int = 2) -> List[ThoughtNode]:
        """Generate child thoughts for tree exploration"""
        children = []
        
        for i in range(max_children):
            child_prompt = f"""
            Parent thought: {parent.content}
            Generate option {i+1} (brief):
            """
            
            if self.ai:
                content = self._compressed_inference(child_prompt, max_tokens=100)
            else:
                content = f"Option {i+1} from {parent.id}"
            
            child = ThoughtNode(
                id=f"{parent.id}_child_{i}",
                content=content,
                confidence=parent.confidence * 0.9,
                reasoning_type='analysis'
            )
            children.append(child)
        
        return children
    
    def _vote_on_paths(self, paths: List[List[ThoughtNode]]) -> int:
        """Vote on the best reasoning path"""
        scores = []
        
        for i, path in enumerate(paths):
            # Score based on average confidence and coherence
            avg_confidence = sum(t.confidence for t in path) / len(path)
            coherence = 1.0 - (len(path) * 0.1)  # Prefer shorter paths
            score = avg_confidence * coherence
            scores.append(score)
        
        return scores.index(max(scores))
    
    def _synthesize_solution(self, thoughts: List[ThoughtNode], prompt: str) -> Dict[str, Any]:
        """Synthesize final solution from thought chain"""
        
        # Group thoughts by type
        by_type = {}
        for thought in thoughts:
            if thought.reasoning_type not in by_type:
                by_type[thought.reasoning_type] = []
            by_type[thought.reasoning_type].append(thought)
        
        # Build solution
        solution = {
            'prompt': prompt,
            'num_thoughts': len(thoughts),
            'confidence': sum(t.confidence for t in thoughts) / len(thoughts) if thoughts else 0,
            'reasoning_chain': [t.content for t in thoughts],
            'thought_types': {k: len(v) for k, v in by_type.items()},
            'final_answer': thoughts[-1].content if thoughts else "No solution found"
        }
        
        return solution
    
    def _get_relevant_examples(self, prompt: str, limit: int = 3) -> List[Dict]:
        """Retrieve relevant examples from cache"""
        # Simple keyword matching for now
        # In production, use vector similarity
        examples = []
        
        # Default examples
        default_examples = [
            {
                'problem': 'Calculate 15% of 200',
                'solution': '15% × 200 = 0.15 × 200 = 30'
            },
            {
                'problem': 'Sort list [3,1,4,1,5]',
                'solution': 'Compare adjacent elements, swap if needed: [1,1,3,4,5]'
            }
        ]
        
        return default_examples[:limit]
    
    def _get_cache_key(self, prompt: str, strategy: str) -> str:
        """Generate cache key for prompt+strategy"""
        combined = f"{prompt}_{strategy}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def _save_to_disk_cache(self, key: str, result: Dict):
        """Save result to disk cache"""
        cache_file = self.cache_dir / f"{key}.json"
        
        # Convert ThoughtNodes to dicts for serialization
        serializable = {
            'thoughts': [
                {
                    'id': t.id,
                    'content': t.content,
                    'confidence': t.confidence,
                    'reasoning_type': t.reasoning_type,
                    'parent_id': t.parent_id,
                    'children': t.children,
                    'metadata': t.metadata
                }
                for t in result['thoughts']
            ],
            'solution': result['solution']
        }
        
        with open(cache_file, 'w') as f:
            json.dump(serializable, f, indent=2)
    
    async def think_async(self, prompt: str, strategy: str = 'zero_shot') -> Dict:
        """Async version for better performance"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.think,
            prompt,
            strategy,
            5,
            False
        )
    
    def benchmark_strategies(self, prompt: str) -> Dict[str, Dict]:
        """Benchmark all strategies on a prompt"""
        results = {}
        
        for strategy_name in self.strategies:
            start_time = time.time()
            
            thoughts = list(self.think(prompt, strategy=strategy_name, stream=False))
            
            elapsed = time.time() - start_time
            
            results[strategy_name] = {
                'time': elapsed,
                'num_thoughts': len(thoughts),
                'confidence': sum(t.confidence for t in thoughts) / len(thoughts) if thoughts else 0
            }
        
        return results


def integrate_chain_of_thought():
    """Quick integration with existing AI Playground"""
    
    # Import existing AI
    from src.core.working_ai_playground import AIPlayground
    
    # Create enhanced AI with reasoning
    ai = AIPlayground()
    cot_engine = ChainOfThoughtEngine(ai)
    
    # Test different strategies
    test_prompt = "How can we optimize AI inference on edge devices with limited RAM?"
    
    print("🧠 CHAIN OF THOUGHT REASONING ENGINE")
    print("=" * 50)
    
    # Stream thoughts as they're generated
    print("\n📊 Reasoning with Zero-Shot CoT:")
    for thought in cot_engine.think(test_prompt, strategy='zero_shot'):
        print(f"  💭 [{thought.reasoning_type}] {thought.content[:100]}...")
        print(f"     Confidence: {thought.confidence:.2f}")
    
    # Benchmark all strategies
    print("\n⚡ Benchmarking all strategies:")
    results = cot_engine.benchmark_strategies(test_prompt)
    
    for strategy, metrics in results.items():
        print(f"\n{strategy}:")
        print(f"  Time: {metrics['time']:.2f}s")
        print(f"  Thoughts: {metrics['num_thoughts']}")
        print(f"  Confidence: {metrics['confidence']:.2f}")
    
    return cot_engine


if __name__ == "__main__":
    # Test the engine
    engine = integrate_chain_of_thought()
    
    # Example: Use for code generation with reasoning
    code_prompt = "Write a Python function to compress images for edge devices"
    
    print("\n🎯 Advanced Code Generation with Reasoning:")
    solution = engine.think(code_prompt, strategy='chain_of_drafts', stream=True)
    
    print(f"\n✅ Final solution confidence: {solution['confidence']:.2f}")