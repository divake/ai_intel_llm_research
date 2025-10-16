# 🚀 Quick Start Guide - Ollama (Working Version)

## ⚡ Simple Startup (Every Day)

### Option 1: Use the Script (Easiest)
```bash
cd ~/divek_nus/AI-Intel-Research
./start-ollama-working.sh
```

### Option 2: Manual Steps
```bash
# 1. Start Ollama
/usr/local/bin/ollama serve &

# 2. Wait 5 seconds
sleep 5

# 3. Start Web UI
docker start open-webui

# 4. Check it's working
curl http://localhost:11434/api/tags
```

## 🛑 Stop Everything
```bash
cd ~/divek_nus/AI-Intel-Research
./stop-ollama-working.sh
```

## 🔍 Check Status
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Check Web UI
docker ps | grep open-webui

# List available models
/usr/local/bin/ollama list
```

## 🌐 Access Points
- **Web UI**: http://localhost:8080
- **API**: http://localhost:11434

## 📋 Available Models
Your 6 models are ready to use:
1. `tinyllama:latest` - Fastest (1.1B)
2. `llama3.2:3b` - Balanced (3B)
3. `phi3:mini` - Smart (3.8B)
4. `mistral:7b` - Advanced (7B)
5. `deepseek-coder:1.3b` - Code (1.3B)
6. `llava:7b` - Vision (7B)

## 💡 Quick Test
```bash
# Test text generation
curl -s http://localhost:11434/api/generate -d '{
  "model": "tinyllama",
  "prompt": "Hello, how are you?",
  "stream": false
}' | jq .response

# Or use the Web UI at http://localhost:8080
```

## ⚠️ Troubleshooting

### Problem: Port already in use
```bash
# Kill everything on port 11434
sudo lsof -ti:11434 | xargs sudo kill -9
sleep 2
# Then restart
./start-ollama-working.sh
```

### Problem: Ollama won't start
```bash
# Check logs
journalctl -u ollama -n 50

# Try manual start to see errors
/usr/local/bin/ollama serve
```

### Problem: Web UI not accessible
```bash
# Check Docker status
docker ps -a

# Restart container
docker restart open-webui
```

## 📝 What Changed (October 15, 2025)

**Problem**: IPEX-LLM binary (April 2025) became incompatible with updated system libraries (glibc 2.39, kernel 6.14).

**Solution**: Switched to standard Ollama which:
- ✅ Works with current system
- ✅ Uses same models
- ✅ Reliable and stable
- ⚠️ CPU-only (no GPU acceleration)

**Your old IPEX-LLM**: Still in `frameworks/ollama-ipex-llm-2.3.0b20250429-ubuntu/` (don't use until rebuilt)

## 🔄 To Get GPU Acceleration Back

You'll need to either:
1. **Build IPEX-LLM from source** against current libraries:
   ```bash
   cd frameworks/ipex-llm
   git pull
   # Follow build instructions
   ```

2. **Wait for newer IPEX-LLM release** (post-October 2025)

3. **Use current working setup** (recommended for now)

---

**Last Updated**: October 15, 2025
**Status**: ✅ Working perfectly with standard Ollama
