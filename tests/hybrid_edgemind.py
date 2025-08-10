# Save as hybrid_edgemind.py
import ollama

class HybridEdgeMind:
    """Use right model for right task"""
    
    def __init__(self):
        self.models = {
            'fast': 'phi3:mini',           # Quick responses
            'balanced': 'llama3.2:3b',     # General purpose  
            'code': 'deepseek-r1:7b-qwen-distill-q4_k_m',  # Programming
            'complex': 'deepseek-r1:14b'   # Heavy reasoning
        }
    
    def route_query(self, query, model_type='balanced'):
        """Route to appropriate model"""
        # Detect query type
        if any(word in query.lower() for word in ['code', 'function', 'python', 'javascript']):
            model_type = 'code'
        elif any(word in query.lower() for word in ['quick', 'simple', 'what is']):
            model_type = 'fast'
        elif any(word in query.lower() for word in ['analyze', 'complex', 'detailed']):
            model_type = 'complex'
        
        model = self.models.get(model_type, self.models['balanced'])
        
        print(f"📍 Using {model} for this query")
        
        try:
            response = ollama.generate(model=model, prompt=query)
            return response['response']
        except Exception as e:
            print(f"❌ Error with {model}: {e}")
            # Fallback to fastest model
            return ollama.generate(model=self.models['fast'], prompt=query)['response']

# Test it
if __name__ == "__main__":
    ai = HybridEdgeMind()
    
    tests = [
        "What is 2+2?",  # Should use fast
        "Write a Python web scraper",  # Should use code
        "Analyze the economic impact of AI",  # Should use complex
    ]
    
    for test in tests:
        print(f"\nQuery: {test}")
        response = ai.route_query(test)
        print(f"Response: {response[:100]}...")