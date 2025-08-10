"""
Web search integration for EdgeMind
Enhances responses with current information
"""

import requests
from typing import List, Dict
import os

class WebSearcher:
    """Simple web search using DuckDuckGo (no API key needed)"""
    
    def search(self, query: str, max_results: int = 3) -> str:
        """
        Search DuckDuckGo and return formatted context
        """
        try:
            # DuckDuckGo instant answer API (free, no key)
            url = "https://api.duckduckgo.com/"
            params = {
                'q': query,
                'format': 'json',
                'no_html': 1,
                'skip_disambig': 1
            }
            
            response = requests.get(url, params=params, timeout=5)
            data = response.json()
            
            # Extract useful information
            context_parts = []
            
            # Abstract (summary)
            if data.get('Abstract'):
                context_parts.append(f"Summary: {data['Abstract']}")
            
            # Answer (direct answer if available)
            if data.get('Answer'):
                context_parts.append(f"Direct Answer: {data['Answer']}")
            
            # Definition
            if data.get('Definition'):
                context_parts.append(f"Definition: {data['Definition']}")
            
            # Related topics
            if data.get('RelatedTopics'):
                for i, topic in enumerate(data['RelatedTopics'][:2]):
                    if isinstance(topic, dict) and 'Text' in topic:
                        context_parts.append(f"Related {i+1}: {topic['Text']}")
            
            if context_parts:
                return "\n".join(context_parts)
            else:
                return f"No specific information found for: {query}"
                
        except Exception as e:
            return f"Search failed: {str(e)}"

def search_enhanced_query(query: str):
    """
    Enhance EdgeMind responses with web search
    """
    from src.core.edgemind import EdgeMind
    
    # Search for context
    searcher = WebSearcher()
    context = searcher.search(query)
    
    print(f"🔍 Web Context Found:\n{context}\n")
    print("-" * 50)
    
    # Generate response with context
    em = EdgeMind(verbose=False)
    
    enhanced_prompt = f"""Based on this context from web search:
{context}

Please answer this question: {query}

Provide an accurate, up-to-date response."""
    
    response = em.generate(enhanced_prompt, max_tokens=200)
    print(f"🤖 EdgeMind (with web search):\n{response}")

if __name__ == "__main__":
    # Test with current events
    search_enhanced_query("Who is the current president of the United States 2025?")