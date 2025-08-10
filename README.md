# 🧠 EdgeMind v0.4.0 - Working Local AI for Real Laptops

[![GitHub](https://img.shields.io/badge/GitHub-artbyoscar-blue)](https://github.com/artbyoscar/ai-playground)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Working-success)](https://github.com/artbyoscar/ai-playground/tree/main)

## ✅ Current Status (August 10, 2025)

**WORKING**: EdgeMind v0.4.0 is now fully functional with Ollama integration, safety systems, and smart routing. This README reflects what ACTUALLY works on consumer laptops.

### 🎯 What's Working Now
- **Ollama Integration** - All models run through Ollama backend
- **Safety System** - Properly blocks dangerous queries (confirmed working!)
- **Smart Routing** - Automatically selects best model for each query
- **Conversation Memory** - Maintains context across 10 exchanges
- **Multiple Models** - Phi-3, Llama 3.2, DeepSeek all tested and working
- **Practical Apps** - Personal assistant, code reviewer, web UI ready

### 📊 Real Performance (Lenovo Yoga, 16GB RAM)
| Model | Size | Actual Speed | Use Case |
|-------|------|--------------|----------|
| **phi3:mini** | 2.2GB | 5.8 tok/s | Quick responses |
| **llama3.2:3b** | 2.0GB | 7.9 tok/s | General chat |
| **deepseek-r1:7b** | 4.7GB | 4.7 tok/s | Coding tasks |
| **deepseek-r1:14b** | 9.0GB | 2-3 tok/s | Complex reasoning |

### ❌ What Won't Work on Laptops
- **Mixtral 8x7b** - Needs 32GB RAM (your laptop has 16GB)
- **Llama 3:70B** - Needs 40GB+ RAM
- **GPU Acceleration** - AMD integrated graphics not supported by Ollama
- **Real-time voice** - Possible but requires setup

---

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.10+
- 8-16GB RAM
- 20GB free disk space
- Windows/Mac/Linux

### 2. Installation
```bash
# Clone repository
git clone https://github.com/artbyoscar/ai-playground.git
cd ai-playground

# Create virtual environment
python -m venv ai-env

# Activate (Windows)
.\ai-env\Scripts\Activate.ps1
# Or Mac/Linux
source ai-env/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install ollama  # Optional but recommended
```

### 3. Install Ollama
```bash
# Windows: Download from https://ollama.com/download/windows
# Mac/Linux:
curl -fsSL https://ollama.com/install.sh | sh
```

### 4. Download Working Models
```bash
# Essential models (tested and working)
ollama pull phi3:mini          # 2.2GB - Fastest
ollama pull llama3.2:3b        # 2.0GB - Best overall
ollama pull deepseek-r1:7b-qwen-distill-q4_k_m  # 4.7GB - For coding

# Optional (slower but more capable)
ollama pull deepseek-r1:14b    # 9.0GB - Complex tasks
```

### 5. Test EdgeMind
```bash
# Run demo
python demo.py

# Start chat interface
python src/core/edgemind.py --chat

# Run benchmark
python src/core/edgemind.py --benchmark
```

---

## 💻 Practical Applications (Ready to Use!)

### 1. Personal Assistant
```bash
python assistant.py
```
Features:
- Morning briefings with weather
- Task management
- Code helper
- Daily tips and motivation

### 2. Code Reviewer
```bash
python code_reviewer.py
```
Features:
- Analyzes Python files for issues
- Suggests improvements
- Security checks
- Best practices recommendations

### 3. Web Interface
```bash
python web_ui.py
# Open http://localhost:5000
```
Features:
- Browser-based chat
- Share with others on network
- Clean, simple interface

### 4. Interactive Chat
```bash
python src/core/edgemind.py --chat
```
Commands:
- `/model <name>` - Switch models
- `/route on/off` - Toggle smart routing
- `/safety on/off` - Toggle safety checks
- `/metrics` - Show performance stats
- `/quit` - Exit

---

## 🔧 Configuration

### Model Routing (Automatic)
EdgeMind automatically selects the best model:
- **Coding queries** → DeepSeek 7B
- **Quick questions** → Phi-3 Mini
- **General chat** → Llama 3.2
- **Complex analysis** → DeepSeek 14B

### Safety System
Properly blocks:
- ✅ "How to make a bomb" → BLOCKED
- ✅ "Write malicious code" → REFUSED
- ✅ "Create ransomware" → BLOCKED
- ✅ Normal queries → ALLOWED

### Memory Management
- Maintains last 10 conversation exchanges
- Automatically trims older context
- Can be cleared with `/clear` command

---

## 📁 Project Structure
```
ai-playground/
├── src/
│   ├── core/
│   │   ├── edgemind.py         # v0.4.0 main engine ✅
│   │   ├── smart_rag.py        # RAG system (optional)
│   │   └── streaming_demo.py   # Streaming responses
│   ├── agents/
│   │   └── safe_computer_control.py  # Safety system ✅
│   ├── tools/
│   │   ├── web_search.py       # Web search integration
│   │   ├── better_search.py    # Weather and news
│   │   └── voice_assistant.py  # Voice I/O (optional)
│   └── models/
│       └── gpt_oss_integration.py  # Placeholder
├── tests/
│   ├── test_working_models.py  # Model benchmarks ✅
│   └── hybrid_edgemind.py      # Routing tests ✅
├── demo.py                      # Feature demonstration ✅
├── assistant.py                 # Personal assistant app
├── code_reviewer.py            # Code analysis tool
├── web_ui.py                   # Web interface
└── README.md                   # This file
```

---

## 🚀 Upcoming Features

### This Week (Immediate)
- [ ] **Streaming Responses** - Show text as it generates
- [ ] **Web Search Integration** - Current information access
- [ ] **Better Weather API** - More accurate local weather
- [ ] **VSCode Extension** - Direct IDE integration

### Next Month (September 2025)
- [ ] **RAG System** - Document search and retrieval
- [ ] **Voice Assistant** - Hands-free interaction
- [ ] **Fine-tuning Interface** - Customize models for your needs
- [ ] **Mobile Web App** - Responsive design for phones

### Q4 2025
- [ ] **Multi-user Support** - Family/team accounts
- [ ] **Plugin System** - Extensible architecture
- [ ] **Model Merging** - Combine strengths of different models
- [ ] **Advanced Safety** - Constitutional AI implementation

### 2026 Vision
- [ ] **BitNet Models** - When available (71x efficiency)
- [ ] **Distributed Inference** - Multiple devices as one
- [ ] **Custom Hardware** - Optimized for edge AI
- [ ] **Brain-Computer Interface** - Just kidding... or are we?

---

## 🛠️ Troubleshooting

### Common Issues & Solutions

#### "Model requires more memory than available"
```bash
# Solution: Use smaller model
ollama pull phi3:mini  # Instead of larger models
```

#### Slow response times
```bash
# Close other applications
# Use faster model:
/model phi3:mini  # In chat mode
```

#### Safety system blocking legitimate queries
```bash
# Temporarily disable in chat:
/safety off
# Re-enable after:
/safety on
```

#### Import errors
```python
# Create missing file:
# src/models/gpt_oss_integration.py
class GPTOSSIntegration:
    def __init__(self):
        self.name = "GPT-OSS"
```

---

## 📊 Benchmarks

### Real-World Performance (Your Laptop)
```
System: Lenovo Yoga, 16GB RAM, AMD Radeon 780M
OS: Windows 11
Python: 3.10+

Results from actual testing:
- phi3:mini: 5.8 tokens/sec average
- llama3.2:3b: 7.9 tokens/sec average  
- deepseek-r1:7b: 4.7 tokens/sec average
- Memory usage: 11.3GB with models loaded
```

### Comparison to Cloud Services
| Service | Cost | Speed | Privacy | Offline |
|---------|------|-------|---------|---------|
| EdgeMind | Free* | 5-8 tok/s | 100% | Yes |
| ChatGPT | $20/mo | 50+ tok/s | No | No |
| Claude | $20/mo | 40+ tok/s | No | No |
| Groq | Free tier | 100+ tok/s | No | No |

*After initial setup

---

## 💡 Tips for Best Performance

### Optimize Your Setup
1. **Close unnecessary apps** - Free up RAM
2. **Use SSD** - Faster model loading
3. **Disable Windows Defender scanning** - For ai-playground folder
4. **Use smaller models** - Phi-3 for most tasks
5. **Enable routing** - Let EdgeMind choose

### Model Selection Guide
- **General questions**: Llama 3.2 (balanced)
- **Code/technical**: DeepSeek 7B (specialized)
- **Quick lookups**: Phi-3 Mini (fastest)
- **Complex analysis**: DeepSeek 14B (if you can wait)
- **Safety-critical**: Always keep safety ON

### Power User Commands
```python
# Create shortcuts for common tasks
# Add to your PowerShell profile:
function ai { python C:\path\to\edgemind.py --chat }
function ai-help { python C:\path\to\assistant.py }
function ai-code { python C:\path\to\code_reviewer.py $args }
```

---

## 🤝 Contributing

### We Need Help With
1. **Documentation** - Improve guides and examples
2. **Testing** - On different hardware configurations  
3. **Features** - Web search, voice, plugins
4. **Models** - Fine-tuning for specific tasks
5. **UI/UX** - Better interfaces and experiences

### How to Contribute
```bash
# Fork the repo at github.com/artbyoscar/ai-playground
# Create feature branch
git checkout -b feature/amazing-feature

# Make changes
# Test thoroughly on YOUR hardware
# Commit with clear message
git commit -m "feat: add amazing feature"

# Push and create PR
git push origin feature/amazing-feature
```

---

## 📈 Roadmap Progress

### ✅ Completed (v0.4.0)
- [x] Ollama integration
- [x] Safety system working
- [x] Model routing
- [x] Conversation memory
- [x] Multiple model support
- [x] Practical applications
- [x] Web interface
- [x] Benchmarking suite

### 🔄 In Progress
- [ ] Streaming responses (code ready)
- [ ] Web search integration (basic version ready)
- [ ] Voice assistant (TTS working)

### 📋 Planned
- [ ] RAG implementation
- [ ] Fine-tuning interface
- [ ] Plugin system
- [ ] Mobile app

---

## 🐛 Known Limitations

### Hardware Constraints
- **No GPU acceleration** on AMD integrated graphics
- **CPU-only inference** limits speed to 5-10 tok/s
- **RAM limitations** prevent large models
- **Battery drain** during continuous use

### Software Limitations
- **Model quality** varies by size
- **No internet access** for models (by design)
- **Context window** limited to 8-32K tokens
- **No multimodal** support yet (text only)

---

## 📞 Support & Community

### Get Help
- **Issues**: [GitHub Issues](https://github.com/artbyoscar/ai-playground/issues)
- **Discussions**: [GitHub Discussions](https://github.com/artbyoscar/ai-playground/discussions)
- **Email**: art.by.oscar.n@gmail.com

### Resources
- [Ollama Documentation](https://ollama.com/docs)
- [Model Library](https://ollama.com/library)
- [Hardware Guide](docs/hardware.md) (coming soon)
- [API Reference](docs/api.md) (coming soon)

---

## 📄 License

MIT License - Free to use, modify, and distribute

---

## 🙏 Acknowledgments

- **Oscar Nuñez** - Creator and maintainer
- **Ollama** - For making local AI accessible
- **Microsoft** - For Phi-3 models
- **Meta** - For Llama models
- **DeepSeek** - For powerful open models
- **Community** - For testing and feedback

---

## 🎯 The Bottom Line

**EdgeMind v0.4.0** is a working, practical, local AI system that:
- ✅ Actually runs on normal laptops
- ✅ Provides useful features today
- ✅ Respects your privacy
- ✅ Works completely offline
- ✅ Costs nothing after setup

It's not AGI, it's not as fast as ChatGPT, but it's **yours**, it's **private**, and it **works**.

---

**Start here**: `python src/core/edgemind.py --chat`

**Repository**: [github.com/artbyoscar/ai-playground](https://github.com/artbyoscar/ai-playground)

*Created by Oscar Nuñez - Last Updated: August 10, 2025 - v0.4.0 Release*