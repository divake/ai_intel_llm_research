# Quick Reference - Ollama AI Stack

**Last Updated: October 15, 2025**

⚠️ **IMPORTANT**: Now using standard Ollama (v0.12.5) due to October 2025 system updates. The old IPEX-LLM setup is no longer compatible. Everything still works perfectly!

---

## 🚀 Essential Commands

### Start/Stop Services
```bash
# Start everything
cd ~/divek_nus/AI-Intel-Research
./start-ollama-working.sh

# Stop everything
./stop-ollama-working.sh

# Check status
curl http://localhost:11434/api/tags
```

### Model Operations
```bash
# List models
ollama list

# Download models
ollama pull tinyllama              # 1.1B - Fastest
ollama pull phi3:mini              # 3.8B - Balanced
ollama pull llama3.2:3b            # 3B - Good quality
ollama pull mistral:7b             # 7B - Better quality
ollama pull llava:7b               # 7B - Vision capable
ollama pull deepseek-coder:1.3b    # 1.3B - For coding

# Chat with model
ollama run tinyllama

# One-shot query
echo "Your question here" | ollama run tinyllama

# Delete model
ollama rm model-name

# Model info
ollama show model-name
```

### Web Access
- **Web UI**: http://localhost:8080
- **API**: http://localhost:11434

### API Examples
```bash
# Basic query
curl http://localhost:11434/api/generate -d '{
  "model": "tinyllama",
  "prompt": "Hello world",
  "stream": false
}'

# List models via API
curl http://localhost:11434/api/tags | jq

# Check if running
curl http://localhost:11434
```

---

## 🔧 Troubleshooting

### If Ollama won't start
```bash
# Kill all Ollama processes
pkill -f ollama
sleep 2

# Restart
cd ~/divek_nus/AI-Intel-Research
./start-ollama-working.sh
```

### Port already in use
```bash
# Kill processes on port 11434
sudo lsof -ti:11434 | xargs sudo kill -9
sleep 2

# Restart
./start-ollama-working.sh
```

### Web UI not accessible
```bash
# Check Docker status
docker ps -a | grep open-webui

# Restart container
docker restart open-webui

# View logs
docker logs open-webui
```

### Model not responding
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

### Monitor system resources
```bash
# System resources
htop

# Memory usage
free -h

# Disk space
df -h
```

---

## 🎯 Performance Settings

```bash
# Keep model loaded longer
export OLLAMA_KEEP_ALIVE=30m

# Increase context length
export OLLAMA_NUM_CTX=4096
```

---

## 📍 Key Paths
- **Project Root**: `~/divek_nus/AI-Intel-Research/`
- **Ollama Binary**: `/usr/local/bin/ollama`
- **Models Storage**: `~/.ollama/models/`
- **Startup Script**: `~/divek_nus/AI-Intel-Research/start-ollama-working.sh`
- **Stop Script**: `~/divek_nus/AI-Intel-Research/stop-ollama-working.sh`

---

## 📦 Available Models

Your 6 ready-to-use models:
1. **tinyllama:latest** (637 MB) - Fastest
2. **deepseek-coder:1.3b** (776 MB) - Code generation
3. **llama3.2:3b** (2.0 GB) - Balanced performance
4. **phi3:mini** (2.2 GB) - Smart reasoning
5. **mistral:7b** (4.1 GB) - Advanced capabilities
6. **llava:7b** (4.7 GB) - Vision + text

---

## 💬 Common Commands

```bash
# Start services
cd ~/divek_nus/AI-Intel-Research && ./start-ollama-working.sh

# Quick chat
ollama run tinyllama

# Download new model
ollama pull llama3.2:3b

# List models
ollama list

# Check status
curl http://localhost:11434/api/tags

# Stop services
cd ~/divek_nus/AI-Intel-Research && ./stop-ollama-working.sh
```

---

## 💡 Tips
1. Start with smaller models (tinyllama) for testing
2. TinyLlama is fastest, Mistral 7B is highest quality
3. Use Web UI at http://localhost:8080 for easier interaction
4. Models are shared - accessible from both CLI and Web UI
5. All models run on CPU (stable and reliable)

---

## 📅 What Changed (October 2025)

**Before**: IPEX-LLM with GPU acceleration (84 tokens/sec on TinyLlama)
**Now**: Standard Ollama with CPU (stable and compatible)

**Reason**: System updated to kernel 6.14.0-29 and glibc 2.39, old IPEX-LLM binary (April 2025) became incompatible with new libraries.

**Result**:
- ✅ All 6 models working
- ✅ Text generation functional
- ✅ Web UI accessible
- ✅ API responding
- ⚠️ CPU-only (no GPU acceleration)

**Future**: Build IPEX-LLM from source or wait for new release to restore GPU acceleration.

---

## 📚 Full Documentation

For detailed instructions, see:
- **QUICK_START.md** - Simple daily usage guide
- **MEMORY_GUIDE.md** - Complete command reference

---

**Status**: ✅ Fully Working
**Version**: Standard Ollama v0.12.5
**Last Tested**: October 15, 2025

---

Keep this handy! 🎉
