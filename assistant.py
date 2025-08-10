"""
EdgeMind Personal Assistant
A practical daily-use tool
"""

from src.core.edgemind import EdgeMind, ModelType
from datetime import datetime
import json
import requests

class BetterSearch:
    """Simple search utilities"""
    def get_weather(self, location="Renton, WA"):
        try:
            # Using wttr.in for simple weather
            url = f"https://wttr.in/{location}?format=3"
            response = requests.get(url, timeout=5)
            return response.text.strip()
        except:
            return "Weather unavailable"

class PersonalAssistant:
    def __init__(self):
        self.em = EdgeMind(verbose=False)
        self.searcher = BetterSearch()
        self.tasks = []
        self.load_tasks()
    
    def load_tasks(self):
        """Load saved tasks"""
        try:
            with open('tasks.json', 'r') as f:
                self.tasks = json.load(f)
        except:
            self.tasks = []
    
    def save_tasks(self):
        """Save tasks to file"""
        with open('tasks.json', 'w') as f:
            json.dump(self.tasks, f, indent=2)
    
    def morning_briefing(self):
        """Generate morning briefing"""
        print("\n" + "="*60)
        print(f"☀️ Good Morning! {datetime.now().strftime('%A, %B %d, %Y')}")
        print("="*60)
        
        # Weather
        weather = self.searcher.get_weather()
        print(f"\n📍 Weather: {weather}")
        
        # Tasks
        if self.tasks:
            print(f"\n📋 Today's Tasks ({len(self.tasks)}):")
            for i, task in enumerate(self.tasks[:5], 1):
                print(f"   {i}. {task}")
        else:
            print("\n📋 No tasks for today")
        
        # Daily tip
        tip = self.em.generate(
            "Give a brief productivity tip for programmers (one sentence)",
            model=ModelType.PHI3_MINI,
            max_tokens=30
        )
        print(f"\n💡 Daily Tip: {tip}")
        
        # Motivation
        quote = self.em.generate(
            "Generate a short motivational quote",
            model=ModelType.PHI3_MINI,
            max_tokens=30
        )
        print(f"\n✨ {quote}")
    
    def add_task(self, task):
        """Add a task"""
        self.tasks.append(task)
        self.save_tasks()
        print(f"✅ Added: {task}")
    
    def remove_task(self, index):
        """Remove a task by number"""
        if 0 < index <= len(self.tasks):
            removed = self.tasks.pop(index - 1)
            self.save_tasks()
            print(f"❌ Removed: {removed}")
        else:
            print("Invalid task number")
    
    def list_tasks(self):
        """List all tasks"""
        if self.tasks:
            print("\n📋 All Tasks:")
            for i, task in enumerate(self.tasks, 1):
                print(f"   {i}. {task}")
        else:
            print("No tasks")
    
    def code_helper(self, description):
        """Get coding help"""
        response = self.em.generate(
            f"Write Python code for: {description}",
            model=ModelType.DEEPSEEK_7B,
            max_tokens=200
        )
        print(f"\n💻 Code Helper:\n{response}")
    
    def chat(self):
        """Interactive assistant mode"""
        print("\n🤖 EdgeMind Assistant Commands:")
        print("  /brief - Morning briefing")
        print("  /task add <task> - Add task")
        print("  /task remove <n> - Remove task")
        print("  /task list - List tasks")
        print("  /code <description> - Get code help")
        print("  /quit - Exit")
        
        while True:
            user_input = input("\n> ").strip()
            
            if user_input == '/quit':
                print("👋 Goodbye!")
                break
            elif user_input == '/brief':
                self.morning_briefing()
            elif user_input.startswith('/task add '):
                task = user_input[10:]
                self.add_task(task)
            elif user_input.startswith('/task remove '):
                try:
                    index = int(user_input[13:])
                    self.remove_task(index)
                except ValueError:
                    print("Use: /task remove <number>")
            elif user_input == '/task list':
                self.list_tasks()
            elif user_input.startswith('/code '):
                self.code_helper(user_input[6:])
            else:
                # Regular chat
                response = self.em.generate(user_input, max_tokens=100)
                print(f"🤖 {response}")

if __name__ == "__main__":
    assistant = PersonalAssistant()
    assistant.morning_briefing()
    assistant.chat()