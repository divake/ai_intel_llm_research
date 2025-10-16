# AI Intel Research - Complete Memory Guide & Command Reference

**Last Updated: October 15, 2025 - IMPORTANT UPDATE**

## ⚠️ IMPORTANT: October 2025 System Update

**What Changed:**
- System updated to kernel 6.14.0-29 and glibc 2.39
- Old IPEX-LLM binary (April 2025) became incompatible
- **Solution:** Switched to standard Ollama (v0.12.5)
- ✅ **Everything works perfectly** - same models, same features

**New Startup Commands:**
```bash
# Start everything (NEW - October 2025)
cd ~/divek_nus/AI-Intel-Research
./start-ollama-working.sh

# Stop everything (NEW)
./stop-ollama-working.sh
```

---

## 📋 Table of Contents
1. [Quick Start Commands](#quick-start-commands) ⭐ **START HERE**
2. [What We Built](#what-we-built)
3. [Model Management](#model-management)
4. [API Usage](#api-usage)
5. [Troubleshooting](#troubleshooting)
6. [Advanced Usage](#advanced-usage)

---

## 🚀 Quick Start Commands

### ⚡ Daily Startup (Use This!)
```bash
cd ~/divek_nus/AI-Intel-Research
./start-ollama-working.sh
```

### 🛑 Shutdown
```bash
cd ~/divek_nus/AI-Intel-Research
./stop-ollama-working.sh
```

### ✅ Check Status
```bash
# Test API
curl http://localhost:11434/api/tags

# List models
/usr/local/bin/ollama list

# Test generation
curl -s http://localhost:11434/api/generate -d '{"model":"tinyllama","prompt":"Hello","stream":false}'
```

### 🌐 Access Points
- **Web UI**: http://localhost:8080
- **API**: http://localhost:11434

---

## 🏗️ What We Built

### Overview
Complete AI research environment optimized for Intel Core Ultra 7 165H:
- **Standard Ollama**: Latest version (v0.12.5) - reliable and stable
- **Open WebUI**: ChatGPT-like web interface
- **6 Ready-to-Use Models**: All working perfectly
- **CPU Inference**: Stable and compatible with current system

### Available Models
1. **tinyllama:latest** (637 MB) - Fastest, great for testing
2. **deepseek-coder:1.3b** (776 MB) - Code generation
3. **llama3.2:3b** (2.0 GB) - Balanced performance
4. **phi3:mini** (2.2 GB) - Smart reasoning
5. **mistral:7b** (4.1 GB) - Advanced capabilities
6. **llava:7b** (4.7 GB) - Vision + text understanding

---

## 📦 Model Management

### List Available Models
```bash
/usr/local/bin/ollama list
```

### Pull New Models
```bash
# Small models (recommended)
ollama pull tinyllama              # 1.1B parameters
ollama pull phi3:mini              # 3.8B parameters
ollama pull gemma2:2b              # 2B parameters

# Medium models
ollama pull llama3.2:3b            # 3B parameters
ollama pull mistral:7b             # 7B parameters
ollama pull deepseek-coder:6.7b    # 6.7B code model

# Vision models
ollama pull llava:7b               # 7B vision model
ollama pull bakllava:7b            # Alternative vision

# Large models (need more RAM)
ollama pull llama3.1:8b            # 8B parameters
ollama pull mixtral:8x7b           # 47B MoE model
```

### Delete Models
```bash
ollama rm <model-name>
```

### Model Information
```bash
ollama show <model-name>
```

---

## 💬 Interactive Usage

### Chat with Model
```bash
ollama run tinyllama
# Type your messages, use /bye to exit
```

### One-shot Query
```bash
echo "What is 2+2?" | ollama run tinyllama
```

### With System Prompt
```bash
ollama run tinyllama "You are a helpful coding assistant. Explain Python decorators."
```

### Generate Code
```bash
echo "Write a Python function to calculate fibonacci numbers" | ollama run deepseek-coder:1.3b
```

---

## 🔌 API Usage

### Basic Generation
```bash
curl http://localhost:11434/api/generate -d '{
  "model": "tinyllama",
  "prompt": "Why is the sky blue?",
  "stream": false
}'
```

### Streaming Response
```bash
curl http://localhost:11434/api/generate -d '{
  "model": "tinyllama",
  "prompt": "Tell me a story",
  "stream": true
}'
```

### Chat Format
```bash
curl http://localhost:11434/api/chat -d '{
  "model": "tinyllama",
  "messages": [
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi there! How can I help you?"},
    {"role": "user", "content": "What is quantum computing?"}
  ]
}'
```

### Python Integration
```python
import requests

def query_ollama(prompt, model="tinyllama"):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False
        }
    )
    return response.json()["response"]

# Usage
result = query_ollama("What is the meaning of life?")
print(result)
```

---

## 🔧 Troubleshooting

### Problem: Port already in use
```bash
sudo lsof -ti:11434 | xargs sudo kill -9
sleep 2
./start-ollama-working.sh
```

### Problem: Ollama won't start
```bash
# Check if process is running
ps aux | grep ollama

# Kill all Ollama processes
pkill -f ollama
sleep 2

# Restart
./start-ollama-working.sh
```

### Problem: Web UI not accessible
```bash
# Check Docker status
docker ps -a | grep open-webui

# Restart container
docker restart open-webui

# View logs
docker logs open-webui
```

### Problem: Model not responding
```bash
# Check API
curl http://localhost:11434/api/tags

# Test simple query
curl -s http://localhost:11434/api/generate -d '{
  "model": "tinyllama",
  "prompt": "Hi",
  "stream": false
}'
```

---

## 🚀 Advanced Usage

### Custom System Prompts
```bash
# Create a modelfile
cat > Modelfile << EOF
FROM tinyllama
SYSTEM You are a pirate. Always speak like a pirate.
EOF

# Create custom model
ollama create pirate-llama -f Modelfile

# Use it
ollama run pirate-llama "Tell me about AI"
```

### JSON Output Mode
```bash
curl http://localhost:11434/api/generate -d '{
  "model": "tinyllama",
  "prompt": "List 3 fruits as JSON",
  "format": "json",
  "stream": false
}'
```

### Batch Processing
```bash
# Process multiple prompts from file
cat prompts.txt | while read prompt; do
  echo "=== $prompt ==="
  echo "$prompt" | ollama run tinyllama
  echo
done > responses.txt
```

---

## 📂 Directory Structure

```
/home/nus-ai/divek_nus/AI-Intel-Research/
├── start-ollama-working.sh     # ⭐ NEW: Use this to start
├── stop-ollama-working.sh      # ⭐ NEW: Use this to stop
├── QUICK_START.md              # ⭐ NEW: Quick reference
├── MEMORY_GUIDE.md             # This file
├── QUICK_REFERENCE.md          # Quick commands
├── README.md                   # Project documentation
├── frameworks/
│   ├── ollama-ipex-llm-2.3.0b20250429-ubuntu/  # Old (don't use)
│   └── ipex-llm/                               # Repository
├── models/                     # Model categorization
├── benchmarks/                 # Performance tests
│   ├── scripts/
│   └── results/
└── vlm_env_description/        # VLM demos
```

---

## 🔑 Key Locations

- **Ollama Binary**: `/usr/local/bin/ollama` (system-wide)
- **Models Storage**: `~/.ollama/models/`
- **Web UI**: http://localhost:8080
- **API**: http://localhost:11434
- **Startup Script**: `~/divek_nus/AI-Intel-Research/start-ollama-working.sh`

---

## 📝 Quick Reference Card

### Most Used Commands
```bash
# Start everything
cd ~/divek_nus/AI-Intel-Research && ./start-ollama-working.sh

# Quick chat
ollama run tinyllama

# Download new model
ollama pull llama3.2:3b

# List models
ollama list

# Check status
curl http://localhost:11434/api/tags

# Stop everything
cd ~/divek_nus/AI-Intel-Research && ./stop-ollama-working.sh
```

---

## 🔮 Future: Get GPU Acceleration Back

To restore IPEX-LLM GPU optimizations (84+ tokens/sec):

### Option 1: Build from Source
```bash
cd frameworks/ipex-llm
git pull
# Follow build instructions for your system
```

### Option 2: Wait for New Release
Check https://github.com/intel/ipex-llm/releases for post-October 2025 builds

### Option 3: Use Current Setup (Recommended)
Standard Ollama is stable, reliable, and works perfectly for development.

---

## 📊 What Happened (Technical Details)

**Timeline:**
- **April 2025**: IPEX-LLM 2.3.0b released
- **July 2025**: Benchmarks showed 84.4 tokens/sec (TinyLlama) ✅
- **October 2025**: System updated (kernel 6.14, glibc 2.39)
- **Result**: IPEX-LLM binary incompatible with new libraries
- **Fix**: Installed standard Ollama v0.12.5 ✅

**Current Status:**
- ✅ All 6 models working
- ✅ Text generation functional
- ✅ Web UI accessible
- ✅ API responding
- ⚠️ CPU-only (no GPU acceleration)

---

## 📞 Getting Help

1. Read `QUICK_START.md` for simple instructions
2. Check if Ollama is running: `curl http://localhost:11434/api/tags`
3. View this guide's troubleshooting section
4. Check system resources: `htop` and `free -h`
5. If stuck, restart: `./stop-ollama-working.sh && ./start-ollama-working.sh`

---

**Remember:** Use `./start-ollama-working.sh` - the old scripts are deleted!

---

Last Updated: October 15, 2025
Version: 2.0 (Updated for system compatibility)
Status: ✅ Fully Working with Standard Ollama
