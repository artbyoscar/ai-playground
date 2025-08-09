# src/core/working_ai_playground.py
import os
import requests
from dotenv import load_dotenv
import json
from datetime import datetime

load_dotenv()

class AIPlayground:
    """Your working AI playground using Together.ai + Mixtral"""
    
    def __init__(self):
        self.api_key = os.getenv('TOGETHER_API_KEY')
        self.model = "mistralai/Mixtral-8x7B-Instruct-v0.1"
        self.url = "https://api.together.xyz/v1/chat/completions"
        self.conversation_log = []
        
    def chat(self, message, system_prompt="You are a helpful AI assistant.", max_tokens=500):
        """Send a message to the AI and get a response"""
        
        if not self.api_key:
            return "❌ No Together.ai API key found. Check your .env file."
        
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message}
            ],
            "max_tokens": max_tokens,
            "temperature": 0.7
        }
        
        try:
            print(f"🤖 Sending to {self.model}...")
            response = requests.post(self.url, headers=headers, json=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                ai_response = result['choices'][0]['message']['content']
                
                # Log the conversation
                self.conversation_log.append({
                    "timestamp": datetime.now().isoformat(),
                    "user": message,
                    "ai": ai_response,
                    "model": self.model
                })
                
                print("✅ Response received!")
                return ai_response
            else:
                return f"❌ API Error {response.status_code}: {response.text}"
                
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    def code_assistant(self, code_request):
        """Specialized coding assistant"""
        system_prompt = """You are an expert programmer. Provide clear, working code with explanations. 
        Focus on Python, JavaScript, and web development. Always include comments and best practices."""
        
        return self.chat(code_request, system_prompt)
    
    def content_writer(self, content_request):
        """Specialized content writing assistant"""
        system_prompt = """You are a professional content writer. Create engaging, well-structured content 
        for marketing, social media, blogs, and business communications. Be creative but professional."""
        
        return self.chat(content_request, system_prompt)
    
    def business_advisor(self, business_question):
        """Specialized business advisor"""
        system_prompt = """You are a business advisor with expertise in startups, marketing, and strategy. 
        Provide actionable, practical advice for entrepreneurs and small businesses."""
        
        return self.chat(business_question, system_prompt)
    
    def save_conversation(self, filename=None):
        """Save conversation log to file"""
        if not filename:
            filename = f"data/logs/conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.conversation_log, f, indent=2, ensure_ascii=False)
        
        return f"💾 Conversation saved to {filename}"
    
    def get_stats(self):
        """Get usage statistics"""
        total_conversations = len(self.conversation_log)
        total_user_chars = sum(len(entry['user']) for entry in self.conversation_log)
        total_ai_chars = sum(len(entry['ai']) for entry in self.conversation_log)
        
        return {
            "total_conversations": total_conversations,
            "user_characters": total_user_chars,
            "ai_characters": total_ai_chars,
            "estimated_tokens": (total_user_chars + total_ai_chars) // 4,  # Rough estimate
            "model_used": self.model
        }

def interactive_demo():
    """Interactive demo of the AI playground"""
    print("🎮 AI Playground - Interactive Demo")
    print("=" * 50)
    print("Available commands:")
    print("  'chat' - General conversation")
    print("  'code' - Coding assistance") 
    print("  'content' - Content writing")
    print("  'business' - Business advice")
    print("  'stats' - Show usage stats")
    print("  'save' - Save conversation")
    print("  'quit' - Exit")
    print("=" * 50)
    
    ai = AIPlayground()
    
    while True:
        command = input("\n🎯 Command: ").strip().lower()
        
        if command == 'quit':
            print("👋 Goodbye!")
            break
        elif command == 'stats':
            stats = ai.get_stats()
            print(f"\n📊 Usage Stats:")
            for key, value in stats.items():
                print(f"   {key}: {value}")
        elif command == 'save':
            result = ai.save_conversation()
            print(f"\n{result}")
        elif command in ['chat', 'code', 'content', 'business']:
            message = input(f"\n💬 Your {command} request: ")
            if message.strip():
                print(f"\n🤖 AI Response:")
                print("-" * 30)
                
                if command == 'chat':
                    response = ai.chat(message)
                elif command == 'code':
                    response = ai.code_assistant(message)
                elif command == 'content':
                    response = ai.content_writer(message)
                elif command == 'business':
                    response = ai.business_advisor(message)
                
                print(response)
                print("-" * 30)
        else:
            print("❌ Unknown command. Try: chat, code, content, business, stats, save, quit")

def quick_test():
    """Quick test of all features"""
    print("🧪 Quick Test of AI Playground")
    print("=" * 50)
    
    ai = AIPlayground()
    
    tests = [
        ("General Chat", "Hello! I'm building an AI playground. Give me one encouraging tip."),
        ("Code Assistant", "Write a simple Python function that calculates compound interest."),
        ("Content Writer", "Write a catchy Instagram caption for a tech startup."),
        ("Business Advisor", "I want to monetize my AI skills as a solo developer. What's the best first step?")
    ]
    
    for test_name, test_message in tests:
        print(f"\n🔧 Testing {test_name}...")
        response = ai.chat(test_message)
        print(f"✅ Response: {response[:100]}...")
    
    # Show stats
    stats = ai.get_stats()
    print(f"\n📊 Test completed! Used ~{stats['estimated_tokens']} tokens")
    
    # Save the test conversation
    filename = ai.save_conversation()
    print(f"💾 {filename}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        interactive_demo()
    else:
        quick_test()