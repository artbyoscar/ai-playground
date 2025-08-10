"""
Project Structure Analyzer for EdgeMind AI Playground
Generates a complete map of your codebase with descriptions
"""

import os
import json
from pathlib import Path
from datetime import datetime

class ProjectMapper:
    def __init__(self, root_path="."):
        self.root_path = Path(root_path).resolve()
        self.structure = {}
        self.stats = {
            'total_files': 0,
            'total_dirs': 0,
            'python_files': 0,
            'total_lines': 0,
            'file_types': {}
        }
        
        # Directories to exclude
        self.exclude_dirs = {
            '__pycache__', '.git', 'venv', 'ai-env', 'env',
            'node_modules', '.vscode', '.idea', 'dist',
            'build', 'egg-info', '.pytest_cache', '.venv',
            'site-packages', 'Include', 'Lib', 'Scripts'
        }
        
        # File patterns to exclude
        self.exclude_files = {
            '.pyc', '.pyo', '.pyd', '.so', '.dll', '.dylib',
            '.egg', '.egg-info', '.dist-info', '.orig'
        }
        
        # Known file descriptions based on your project
        self.file_descriptions = {
            'edgemind.py': '🧠 Core AI engine v0.4.0 - Main orchestrator with Ollama integration',
            'assistant.py': '🤖 Personal AI assistant with tasks, weather, and daily briefings',
            'code_reviewer.py': '🔍 AI-powered code analysis and improvement suggestions',
            'web_ui.py': '🌐 Flask web interface for browser-based chat',
            'demo.py': '🎯 Feature demonstration and testing suite',
            'smart_rag.py': '📚 RAG (Retrieval-Augmented Generation) system',
            'streaming_demo.py': '⚡ Streaming response implementation',
            'safe_computer_control.py': '🔒 Safety system for blocking dangerous queries',
            'web_search.py': '🔎 Web search integration module',
            'better_search.py': '🌦️ Weather and news API integration',
            'voice_assistant.py': '🎤 Voice input/output capabilities',
            'gpt_oss_integration.py': '🔗 GPT-OSS model integration placeholder',
            'test_working_models.py': '✅ Model testing and benchmarking',
            'hybrid_edgemind.py': '🔄 Hybrid routing system tests',
            'swarm_demo.py': '🐝 Multi-agent swarm demonstration',
            'thinking_llm.py': '💭 Chain-of-thought reasoning implementation',
            'benchmark.py': '📊 Performance benchmarking tools',
            'requirements.txt': '📦 Python package dependencies',
            'README.md': '📖 Project documentation',
            'LICENSE': '⚖️ MIT License file',
            '.gitignore': '🚫 Git ignore patterns'
        }
    
    def should_exclude(self, path):
        """Check if path should be excluded"""
        path_obj = Path(path)
        
        # Check directory exclusions
        for part in path_obj.parts:
            if part in self.exclude_dirs:
                return True
                
        # Check file exclusions
        if path_obj.is_file():
            if any(str(path_obj).endswith(ext) for ext in self.exclude_files):
                return True
                
        return False
    
    def get_file_stats(self, filepath):
        """Get statistics for a file"""
        try:
            size = os.path.getsize(filepath)
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                lines = len(f.readlines())
            return size, lines
        except:
            return 0, 0
    
    def analyze_directory(self, directory, prefix="", is_last=True):
        """Recursively analyze directory structure"""
        items = []
        path = Path(directory)
        
        try:
            entries = sorted(path.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))
        except PermissionError:
            return items
        
        for i, entry in enumerate(entries):
            if self.should_exclude(entry):
                continue
                
            is_last_entry = (i == len(entries) - 1)
            
            if entry.is_dir():
                self.stats['total_dirs'] += 1
                items.append({
                    'name': entry.name,
                    'type': 'directory',
                    'path': str(entry.relative_to(self.root_path)),
                    'children': self.analyze_directory(entry, prefix, is_last_entry)
                })
            else:
                self.stats['total_files'] += 1
                size, lines = self.get_file_stats(entry)
                
                # Track file types
                ext = entry.suffix
                if ext:
                    self.stats['file_types'][ext] = self.stats['file_types'].get(ext, 0) + 1
                
                if ext == '.py':
                    self.stats['python_files'] += 1
                    self.stats['total_lines'] += lines
                
                # Get description if known
                description = self.file_descriptions.get(entry.name, '')
                
                items.append({
                    'name': entry.name,
                    'type': 'file',
                    'path': str(entry.relative_to(self.root_path)),
                    'size': size,
                    'lines': lines if ext == '.py' else 0,
                    'description': description
                })
        
        return items
    
    def generate_tree(self, items, prefix="", is_root=True):
        """Generate tree-like string representation"""
        tree_str = ""
        
        for i, item in enumerate(items):
            is_last = (i == len(items) - 1)
            
            # Determine the prefix characters
            if is_root:
                connector = ""
                extension = ""
            else:
                connector = "└── " if is_last else "├── "
                extension = "    " if is_last else "│   "
            
            # Add the item
            if item['type'] == 'directory':
                tree_str += f"{prefix}{connector}📁 {item['name']}/\n"
                if item.get('children'):
                    tree_str += self.generate_tree(
                        item['children'], 
                        prefix + extension, 
                        is_root=False
                    )
            else:
                # Show file with description
                desc = f" - {item['description']}" if item.get('description') else ""
                size_str = f" ({item['lines']} lines)" if item.get('lines') else ""
                tree_str += f"{prefix}{connector}📄 {item['name']}{size_str}{desc}\n"
        
        return tree_str
    
    def generate_markdown_report(self, structure):
        """Generate a detailed markdown report"""
        report = f"""# 🗂️ Project Structure Report
**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Project Path**: `{self.root_path}`

## 📊 Statistics
- **Total Directories**: {self.stats['total_dirs']}
- **Total Files**: {self.stats['total_files']}
- **Python Files**: {self.stats['python_files']}
- **Total Lines of Code**: {self.stats['total_lines']:,}

## 📈 File Types Distribution
"""
        # Add file type breakdown
        for ext, count in sorted(self.stats['file_types'].items(), key=lambda x: x[1], reverse=True):
            report += f"- `{ext}`: {count} files\n"
        
        report += "\n## 🌳 Directory Tree\n```\n"
        report += self.generate_tree(structure)
        report += "```\n\n"
        
        # Add detailed file listing
        report += "## 📝 Detailed File Descriptions\n\n"
        report += self.generate_detailed_listing(structure)
        
        return report
    
    def generate_detailed_listing(self, items, level=0):
        """Generate detailed file descriptions"""
        output = ""
        
        for item in items:
            if item['type'] == 'directory':
                indent = "  " * level
                output += f"{indent}### 📁 {item['name']}/\n"
                if item.get('children'):
                    output += self.generate_detailed_listing(item['children'], level + 1)
            elif item.get('description'):
                indent = "  " * level
                lines = f" ({item['lines']} lines)" if item.get('lines') else ""
                output += f"{indent}- **{item['name']}**{lines}: {item['description']}\n"
        
        return output
    
    def run(self, output_format='all'):
        """Run the analysis and generate output"""
        print(f"🔍 Analyzing project structure at: {self.root_path}\n")
        
        # Analyze the structure
        self.structure = self.analyze_directory(self.root_path)
        
        # Generate outputs
        outputs = {}
        
        # Console output (tree)
        if output_format in ['all', 'console', 'tree']:
            tree_output = self.generate_tree(self.structure)
            print("📂 Project Structure:\n")
            print(tree_output)
            outputs['tree'] = tree_output
        
        # JSON output
        if output_format in ['all', 'json']:
            json_file = 'project_structure.json'
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'root': str(self.root_path),
                    'generated': datetime.now().isoformat(),
                    'stats': self.stats,
                    'structure': self.structure
                }, f, indent=2)
            print(f"✅ JSON structure saved to: {json_file}")
            outputs['json'] = json_file
        
        # Markdown report
        if output_format in ['all', 'markdown', 'md']:
            md_file = 'project_structure.md'
            markdown = self.generate_markdown_report(self.structure)
            with open(md_file, 'w', encoding='utf-8') as f:
                f.write(markdown)
            print(f"✅ Markdown report saved to: {md_file}")
            outputs['markdown'] = md_file
        
        # Text file for printing
        if output_format in ['all', 'text', 'txt']:
            txt_file = 'project_structure.txt'
            with open(txt_file, 'w', encoding='utf-8') as f:
                f.write(f"PROJECT STRUCTURE - {self.root_path}\n")
                f.write(f"Generated: {datetime.now()}\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"STATISTICS:\n")
                f.write(f"  Directories: {self.stats['total_dirs']}\n")
                f.write(f"  Files: {self.stats['total_files']}\n")
                f.write(f"  Python Files: {self.stats['python_files']}\n")
                f.write(f"  Total Lines: {self.stats['total_lines']:,}\n\n")
                f.write("STRUCTURE:\n")
                f.write(self.generate_tree(self.structure))
            print(f"✅ Text file saved to: {txt_file}")
            outputs['text'] = txt_file
        
        # Print summary
        print(f"\n📊 Summary:")
        print(f"  - {self.stats['total_dirs']} directories")
        print(f"  - {self.stats['total_files']} files")
        print(f"  - {self.stats['python_files']} Python files")
        print(f"  - {self.stats['total_lines']:,} lines of Python code")
        
        return outputs

def main():
    """Main function with CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Map project structure')
    parser.add_argument('path', nargs='?', default='.', 
                      help='Project path to analyze (default: current directory)')
    parser.add_argument('-f', '--format', default='all',
                      choices=['all', 'tree', 'json', 'markdown', 'md', 'text', 'txt', 'console'],
                      help='Output format (default: all)')
    parser.add_argument('--no-descriptions', action='store_true',
                      help='Skip file descriptions')
    
    args = parser.parse_args()
    
    mapper = ProjectMapper(args.path)
    if args.no_descriptions:
        mapper.file_descriptions = {}
    
    mapper.run(output_format=args.format)

if __name__ == "__main__":
    main()