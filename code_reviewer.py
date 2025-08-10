"""
AI-powered code reviewer
Analyzes Python code for issues and improvements
"""

from src.core.edgemind import EdgeMind, ModelType
import ast
import os

class CodeReviewer:
    def __init__(self):
        self.em = EdgeMind(verbose=False)
    
    def analyze_file(self, filepath):
        """Analyze a Python file"""
        if not os.path.exists(filepath):
            print(f"❌ File not found: {filepath}")
            return
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                code = f.read()
        except UnicodeDecodeError:
            # Try with different encoding if UTF-8 fails
            try:
                with open(filepath, 'r', encoding='latin-1') as f:
                    code = f.read()
                print("⚠️ File encoding issue detected, using fallback encoding")
            except Exception as e:
                print(f"❌ Error reading file: {e}")
                return
        except Exception as e:
            print(f"❌ Error reading file: {e}")
            return
        
        print(f"\n📂 Analyzing: {filepath}")
        print("-" * 50)
        
        # Basic syntax check
        try:
            ast.parse(code)
            print("✅ Syntax valid")
        except SyntaxError as e:
            print(f"❌ Syntax error: {e}")
            return
        
        # Line count
        lines = code.split('\n')
        print(f"📏 Lines of code: {len(lines)}")
        
        # Count functions and classes
        tree = ast.parse(code)
        functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        print(f"🔧 Functions: {len(functions)}")
        print(f"📦 Classes: {len(classes)}")
        
        # AI review
        prompt = f"""Review this Python code for:
1. Potential bugs
2. Performance issues
3. Best practices
4. Security concerns
5. Code organization

Code (first 50 lines):
```python
{chr(10).join(lines[:50])}
```

Provide a brief review with specific suggestions. Be constructive and helpful."""
        
        review = self.em.generate(
            prompt,
            model=ModelType.DEEPSEEK_7B,
            max_tokens=300
        )
        
        print(f"\n📝 AI Code Review:\n{review}")
        
        # Check for common issues
        self.check_common_issues(code)
    
    def check_common_issues(self, code):
        """Check for common Python issues"""
        issues = []
        
        # Check for print statements (might want logging instead)
        if 'print(' in code and 'def main' not in code:
            issues.append("💡 Consider using logging instead of print statements")
        
        # Check for bare except
        if 'except:' in code:
            issues.append("⚠️ Bare 'except:' found - consider catching specific exceptions")
        
        # Check for TODO/FIXME
        if 'TODO' in code or 'FIXME' in code:
            issues.append("📌 TODO/FIXME comments found - don't forget to address them")
        
        # Check for hardcoded passwords
        if 'password =' in code.lower() or 'api_key =' in code.lower():
            issues.append("🔒 Possible hardcoded credentials - use environment variables")
        
        if issues:
            print("\n⚡ Quick Issues Found:")
            for issue in issues:
                print(f"  {issue}")
    
    def suggest_improvements(self, code_snippet):
        """Get improvement suggestions"""
        prompt = f"""Improve this Python code and explain the improvements:

Original code:
```python
{code_snippet}
```

Provide:
1. Improved version of the code
2. Explanation of changes made
3. Why these changes improve the code"""
        
        improved = self.em.generate(
            prompt,
            model=ModelType.DEEPSEEK_7B,
            max_tokens=400
        )
        
        return improved
    
    def compare_files(self, file1, file2):
        """Compare two Python files"""
        print(f"\n🔄 Comparing {file1} vs {file2}")
        
        if not os.path.exists(file1) or not os.path.exists(file2):
            print("❌ One or both files not found")
            return
        
        try:
            with open(file1, 'r', encoding='utf-8') as f:
                code1 = f.read()
            with open(file2, 'r', encoding='utf-8') as f:
                code2 = f.read()
        except Exception as e:
            print(f"❌ Error reading files: {e}")
            return
        
        prompt = f"""Compare these two Python implementations:

File 1 ({file1}):
```python
{code1[:500]}
```

File 2 ({file2}):
```python
{code2[:500]}
```

Which is better and why? Consider performance, readability, and best practices."""
        
        comparison = self.em.generate(
            prompt,
            model=ModelType.DEEPSEEK_7B,
            max_tokens=200
        )
        
        print(f"\n📊 Comparison Result:\n{comparison}")
    
    def interactive_review(self):
        """Interactive code review mode"""
        print("\n🔍 Code Reviewer - Interactive Mode")
        print("Commands:")
        print("  /file <path>         - Review a file")
        print("  /code               - Input code for review")
        print("  /compare <f1> <f2>  - Compare two files")
        print("  /help               - Show commands")
        print("  /quit               - Exit")
        
        while True:
            cmd = input("\n> ").strip()
            
            if cmd == '/quit':
                print("👋 Goodbye!")
                break
            
            elif cmd == '/help':
                print("Available commands:")
                print("  /file <path>         - Review a Python file")
                print("  /code               - Paste code for review")
                print("  /compare <f1> <f2>  - Compare two files")
                print("  /quit               - Exit reviewer")
            
            elif cmd.startswith('/file '):
                filepath = cmd[6:].strip()
                self.analyze_file(filepath)
            
            elif cmd.startswith('/compare '):
                parts = cmd[9:].strip().split()
                if len(parts) >= 2:
                    self.compare_files(parts[0], parts[1])
                else:
                    print("Usage: /compare file1.py file2.py")
            
            elif cmd == '/code':
                print("Paste your code (type 'END' on a new line when done):")
                lines = []
                while True:
                    try:
                        line = input()
                        if line == 'END':
                            break
                        lines.append(line)
                    except KeyboardInterrupt:
                        print("\n❌ Input cancelled")
                        break
                
                if lines:
                    code = '\n'.join(lines)
                    print("\n🔍 Analyzing your code...")
                    
                    # Check syntax first
                    try:
                        ast.parse(code)
                        print("✅ Syntax valid")
                        
                        # Get improvements
                        improved = self.suggest_improvements(code)
                        print(f"\n💡 Suggestions:\n{improved}")
                        
                        # Check common issues
                        self.check_common_issues(code)
                    except SyntaxError as e:
                        print(f"❌ Syntax error: {e}")
            
            else:
                if cmd:
                    print(f"Unknown command: {cmd}")
                    print("Type /help for available commands")

def main():
    """Main entry point"""
    print("\n" + "="*60)
    print("🔍 EdgeMind Code Reviewer v1.0")
    print("="*60)
    print("AI-powered Python code analysis and improvement")
    print("-"*60)
    
    reviewer = CodeReviewer()
    
    # Try to review demo.py if it exists
    if os.path.exists("demo.py"):
        print("\n📝 Reviewing demo.py as example...")
        reviewer.analyze_file("demo.py")
    else:
        print("\n💡 Tip: Use /file <path> to review any Python file")
    
    # Start interactive mode
    reviewer.interactive_review()

if __name__ == "__main__":
    main()