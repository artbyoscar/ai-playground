# src/core/smart_rag.py
"""
SmartRAG - Make your local AI smarter with knowledge
This gives your AI access to documents, Wikipedia, and more
"""

import chromadb
from chromadb.utils import embedding_functions
import wikipediaapi
import requests
from pathlib import Path
from typing import List, Dict, Any
import json

class SmartRAG:
    """
    RAG = Retrieval Augmented Generation
    Makes small models MUCH smarter by giving them knowledge
    """
    
    def __init__(self, llm_engine):
        self.llm = llm_engine
        
        # Initialize vector database for knowledge
        self.client = chromadb.Client()
        
        # Create collection with embeddings
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"  # Small, fast embedding model
        )
        
        self.collection = self.client.create_collection(
            name="edgemind_knowledge",
            embedding_function=self.embedding_fn,
            metadata={"description": "EdgeMind's knowledge base"}
        )
        
        self.doc_count = 0
        
        print("🧠 SmartRAG initialized - Your AI just got smarter!")
    
    def add_knowledge(self, text: str, source: str, metadata: Dict = None):
        """Add information to knowledge base"""
        self.doc_count += 1
        doc_id = f"doc_{self.doc_count}"
        
        meta = {"source": source}
        if metadata:
            meta.update(metadata)
        
        self.collection.add(
            documents=[text],
            metadatas=[meta],
            ids=[doc_id]
        )
        
        print(f"✅ Added knowledge from {source}")
        return doc_id
    
    def add_wikipedia(self, topic: str, max_paragraphs: int = 5):
        """Add Wikipedia knowledge about a topic"""
        try:
            page = wikipedia.page(topic)
            content = "\n".join(page.content.split("\n")[:max_paragraphs])
            
            self.add_knowledge(
                text=content,
                source=f"Wikipedia: {topic}",
                metadata={
                    "url": page.url,
                    "title": page.title
                }
            )
            
            print(f"📚 Added Wikipedia knowledge about {topic}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to add Wikipedia: {e}")
            return False
    
    def add_document(self, file_path: str):
        """Add a document to knowledge base"""
        path = Path(file_path)
        
        if not path.exists():
            print(f"❌ File not found: {file_path}")
            return False
        
        # Read file based on extension
        if path.suffix == ".txt":
            content = path.read_text()
        elif path.suffix == ".json":
            with open(path, 'r') as f:
                data = json.load(f)
                content = json.dumps(data, indent=2)
        elif path.suffix == ".py":
            content = path.read_text()
        else:
            print(f"❌ Unsupported file type: {path.suffix}")
            return False
        
        self.add_knowledge(
            text=content,
            source=f"File: {path.name}",
            metadata={"path": str(path)}
        )
        
        return True
    
    def search_knowledge(self, query: str, n_results: int = 3) -> List[Dict]:
        """Search the knowledge base"""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        if not results['documents'][0]:
            return []
        
        formatted_results = []
        for i, doc in enumerate(results['documents'][0]):
            formatted_results.append({
                "text": doc,
                "source": results['metadatas'][0][i]['source'],
                "distance": results['distances'][0][i] if 'distances' in results else None
            })
        
        return formatted_results
    
    def smart_answer(self, question: str, use_knowledge: bool = True) -> str:
        """
        Answer using knowledge + LLM
        This makes the AI MUCH smarter!
        """
        
        if use_knowledge and self.collection.count() > 0:
            # Search knowledge base
            results = self.search_knowledge(question, n_results=3)
            
            if results:
                # Build context from search results
                context_parts = []
                for r in results:
                    context_parts.append(f"From {r['source']}:\n{r['text'][:500]}")
                
                context = "\n\n".join(context_parts)
                
                # Create enhanced prompt
                prompt = f"""You are a helpful AI assistant with access to a knowledge base.

Context from knowledge base:
{context}

User Question: {question}

Instructions: Answer the question using the provided context. If the context doesn't contain relevant information, use your general knowledge.

Answer:"""
            else:
                # No relevant knowledge found
                prompt = f"Question: {question}\nAnswer:"
        else:
            # Direct answer without knowledge
            prompt = f"Question: {question}\nAnswer:"
        
        # Generate response
        result = self.llm.generate(prompt, max_tokens=256)
        
        if "error" in result:
            return f"Error: {result['error']}"
        
        return result["response"]
    
    def train_on_conversations(self, conversations: List[Dict]):
        """Learn from good conversations"""
        for conv in conversations:
            q = conv.get("question", "")
            a = conv.get("answer", "")
            
            if q and a:
                # Store Q&A as knowledge
                self.add_knowledge(
                    text=f"Q: {q}\nA: {a}",
                    source="Training Conversation",
                    metadata={"type": "qa_pair"}
                )
    
    def get_stats(self) -> Dict:
        """Get knowledge base statistics"""
        return {
            "total_documents": self.collection.count(),
            "sources": len(set([m['source'] for m in self.collection.get()['metadatas']])),
            "embedding_model": "all-MiniLM-L6-v2",
            "vector_dimensions": 384
        }


# Demo function
def demo_smart_rag():
    """Demo showing how RAG makes AI smarter"""
    from src.core.local_llm_engine_fixed import LocalLLMEngine
    
    print("="*60)
    print("🧠 SmartRAG Demo - Making AI Smarter with Knowledge")
    print("="*60)
    
    # Initialize LLM
    print("\n1️⃣ Loading local AI...")
    llm = LocalLLMEngine("tinyllama")
    llm.load_model()
    
    # Initialize RAG
    print("\n2️⃣ Initializing SmartRAG...")
    rag = SmartRAG(llm)
    
    # Add knowledge
    print("\n3️⃣ Adding knowledge...")
    
    # Add custom knowledge
    rag.add_knowledge(
        text="EdgeMind is an open-source AI platform that runs 100% locally. It was created in August 2025 by Oscar Nuñez. It achieves 30-40 tokens/sec on CPU.",
        source="EdgeMind Documentation"
    )
    
    # Add Wikipedia knowledge
    rag.add_wikipedia("Artificial Intelligence")
    
    # Test questions
    print("\n4️⃣ Testing Q&A...")
    
    questions = [
        "What is EdgeMind?",
        "Who created EdgeMind?",
        "What is artificial intelligence?"
    ]
    
    for q in questions:
        print(f"\n❓ Question: {q}")
        
        # Answer WITHOUT knowledge
        print("🤖 Without RAG: ", end="")
        basic_answer = llm.chat(q)
        print(basic_answer[:100] + "...")
        
        # Answer WITH knowledge
        print("🧠 With RAG: ", end="")
        smart_answer = rag.smart_answer(q)
        print(smart_answer[:200] + "...")
    
    # Show stats
    print("\n5️⃣ Knowledge Base Stats:")
    stats = rag.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print("\n" + "="*60)
    print("✅ RAG makes your AI much smarter!")
    print("="*60)


if __name__ == "__main__":
    demo_smart_rag()