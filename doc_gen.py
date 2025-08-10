"""
Auto-generate documentation for Python files
"""
from src.core.edgemind import EdgeMind, ModelType

def generate_docs(filepath):
    with open(filepath, 'r') as f:
        code = f.read()
    
    em = EdgeMind(verbose=False)
    prompt = f"Generate comprehensive documentation for:\n{code[:1000]}"
    
    docs = em.generate(prompt, model=ModelType.DEEPSEEK_7B)
    
    with open(f"{filepath}.md", 'w') as f:
        f.write(docs)