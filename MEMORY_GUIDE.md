# AI Intel Research - Complete Memory Guide & Command Reference

## 📋 Table of Contents
1. [What We Built](#what-we-built)
2. [Quick Start Commands](#quick-start-commands)
3. [Directory Structure](#directory-structure)
4. [Essential Commands Reference](#essential-commands-reference)
5. [Adding New Models](#adding-new-models)
6. [Benchmarking & Testing](#benchmarking--testing)
7. [Hardware Configuration](#hardware-configuration)
8. [Troubleshooting Guide](#troubleshooting-guide)
9. [Future Enhancements](#future-enhancements)
10. [Advanced Usage](#advanced-usage)

---

## 🏗️ What We Built

### Overview
We created a complete AI research environment optimized for Intel hardware (Core Ultra 7 165H) with:
- **IPEX-LLM Ollama**: Intel-optimized LLM inference server
- **Open WebUI**: ChatGPT-like web interface
- **Benchmarking Suite**: Performance testing tools
- **Multi-hardware Support**: CPU, GPU, and NPU-ready infrastructure

### Key Achievements
1. ✅ Structured project layout for scalability
2. ✅ Intel Arc GPU acceleration working (~51 tokens/s on TinyLlama)
3. ✅ Web interface accessible at http://localhost:8080
4. ✅ Automated launch/stop scripts
5. ✅ Comprehensive benchmarking tools
6. ✅ Hardware detection for CPU/GPU/NPU

---

## 🚀 Quick Start Commands

### Start Everything
```bash
cd ~/divek_nus/AI-Intel-Research
./start-ai-stack.sh
```

### Stop Everything
```bash
cd ~/divek_nus/AI-Intel-Research
./stop-ai-stack.sh
```

### Check Status
```bash
cd ~/divek_nus/AI-Intel-Research
./check_status.sh
```

### Quick Test
```bash
cd ~/divek_nus/AI-Intel-Research/frameworks/ollama-ipex-llm-2.3.0b20250429-ubuntu
echo "Hello, how are you?" | ./ollama run tinyllama
```

---

## 📁 Directory Structure

```
/home/nus-ai/divek_nus/AI-Intel-Research/
├── frameworks/
│   ├── ollama-ipex-llm-2.3.0b20250429-ubuntu/      # Main Ollama binary
│   │   ├── ollama                                   # Executable
│   │   ├── start-ollama.sh                          # Original start script
│   │   └── lib*.so                                  # Intel libraries
│   └── ipex-llm/                                    # Cloned repository
├── models/
│   ├── llm/
│   │   ├── text/      # Text-only models
│   │   ├── multimodal/ # Text+Image models
│   │   └── code/      # Code models
│   └── vlm/
│       ├── image/     # Image understanding
│       └── video/     # Video understanding
├── benchmarks/
│   ├── scripts/
│   │   ├── benchmark_llm.py          # Python benchmark tool
│   │   └── test_hardware_modes.sh    # Hardware comparison
│   └── results/                       # JSON result files
├── start-ai-stack.sh      # Main launcher
├── stop-ai-stack.sh       # Stop all services
├── check_status.sh        # System status checker
├── README.md              # Project documentation
├── PROJECT_STRUCTURE.md   # Directory explanation
└── MEMORY_GUIDE.md        # This file
```

---

## 📚 Essential Commands Reference

### 1. Service Management

#### Start Ollama Server Only
```bash
cd ~/divek_nus/AI-Intel-Research/frameworks/ollama-ipex-llm-2.3.0b20250429-ubuntu

# With GPU acceleration
export OLLAMA_NUM_GPU=999
export ONEAPI_DEVICE_SELECTOR=level_zero:0
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
./start-ollama.sh

# CPU only mode
export OLLAMA_NUM_GPU=0
./start-ollama.sh
```

#### Start Web UI Only
```bash
# If Ollama is already running
docker start open-webui

# First time setup
docker run -d \
  --network="host" \
  --add-host=host.docker.internal:host-gateway \
  --name open-webui \
  --restart always \
  ghcr.io/open-webui/open-webui:main
```

#### Stop Services Manually
```bash
# Stop Ollama
pkill -f "ollama serve"

# Stop Web UI
docker stop open-webui
```

### 2. Model Management

#### List Available Models
```bash
cd ~/divek_nus/AI-Intel-Research/frameworks/ollama-ipex-llm-2.3.0b20250429-ubuntu
./ollama list
```

#### Pull New Models
```bash
# Small models (good for testing)
./ollama pull tinyllama              # 1.1B parameters
./ollama pull phi3:mini               # 3.8B parameters
./ollama pull gemma2:2b               # 2B parameters

# Medium models (better quality)
./ollama pull llama3.2:3b             # 3B parameters
./ollama pull mistral:7b-instruct     # 7B parameters
./ollama pull deepseek-coder:6.7b     # 6.7B code model

# Vision models
./ollama pull llava:7b                # 7B vision model
./ollama pull bakllava:7b             # Alternative vision model

# Large models (need more VRAM)
./ollama pull llama3.1:8b             # 8B parameters
./ollama pull mixtral:8x7b            # 47B MoE model
```

#### Delete Models
```bash
./ollama rm <model-name>
```

#### Model Information
```bash
./ollama show <model-name>
```

### 3. Interactive Usage

#### Chat with Model (Interactive)
```bash
./ollama run tinyllama
# Type your messages, use /bye to exit
```

#### One-shot Query
```bash
echo "What is 2+2?" | ./ollama run tinyllama
```

#### With System Prompt
```bash
./ollama run tinyllama "You are a helpful coding assistant. Explain Python decorators."
```

#### Generate Code
```bash
echo "Write a Python function to calculate fibonacci numbers" | ./ollama run deepseek-coder:1.3b
```

### 4. API Usage

#### Basic Generation
```bash
curl http://localhost:11434/api/generate -d '{
  "model": "tinyllama",
  "prompt": "Why is the sky blue?",
  "stream": false
}'
```

#### Streaming Response
```bash
curl http://localhost:11434/api/generate -d '{
  "model": "tinyllama",
  "prompt": "Tell me a story",
  "stream": true
}'
```

#### Chat Format
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

---

## 🆕 Adding New Models

### Step-by-Step Guide

1. **Check Available Storage**
```bash
df -h ~/.ollama/models
```

2. **Search for Models**
Visit: https://ollama.com/library

3. **Download Model**
```bash
cd ~/divek_nus/AI-Intel-Research/frameworks/ollama-ipex-llm-2.3.0b20250429-ubuntu
./ollama pull <model-name>:<tag>
```

4. **Test Model**
```bash
echo "Hello, introduce yourself" | ./ollama run <model-name>
```

5. **Benchmark Model**
```bash
cd ~/divek_nus/AI-Intel-Research/benchmarks/scripts
python3 benchmark_llm.py --models <model-name> --runs 5
```

### Model Recommendations by Use Case

#### For General Chat
```bash
./ollama pull llama3.2:3b        # Fast, good quality
./ollama pull mistral:7b         # Better quality, slower
./ollama pull phi3:medium        # Microsoft's model
```

#### For Coding
```bash
./ollama pull deepseek-coder:1.3b   # Fast code completion
./ollama pull codellama:7b          # Better code understanding
./ollama pull deepseek-coder:6.7b   # Best quality
```

#### For Vision Tasks
```bash
./ollama pull llava:7b              # General vision
./ollama pull bakllava:7b           # Alternative vision
```

#### For Math/Reasoning
```bash
./ollama pull deepseek-math:7b      # Math specialist
./ollama pull phi3:medium           # Good reasoning
```

---

## 🧪 Benchmarking & Testing

### Run Complete Benchmark Suite
```bash
cd ~/divek_nus/AI-Intel-Research/benchmarks/scripts

# Test all models
python3 benchmark_llm.py

# Test specific model
python3 benchmark_llm.py --models tinyllama:latest

# Custom prompts
python3 benchmark_llm.py --models tinyllama:latest --prompts "What is AI?" "Explain quantum physics"

# More iterations for accuracy
python3 benchmark_llm.py --models tinyllama:latest --runs 10
```

### Hardware Mode Comparison
```bash
cd ~/divek_nus/AI-Intel-Research/benchmarks/scripts
./test_hardware_modes.sh
```
This will test:
- CPU-only performance
- GPU-accelerated performance  
- Auto mode (CPU+GPU)

### View Benchmark Results
```bash
cd ~/divek_nus/AI-Intel-Research/benchmarks/results

# List all results
ls -la benchmark_*.json

# View latest result
cat $(ls -t benchmark_results_*.json | head -1) | jq

# Extract summary
cat $(ls -t benchmark_results_*.json | head -1) | jq '.benchmarks[0].average'
```

### Custom Performance Test
```bash
# Time a single query
time echo "Write a haiku" | ./ollama run tinyllama

# Measure tokens per second
curl -s -X POST http://localhost:11434/api/generate \
  -d '{"model":"tinyllama","prompt":"Tell me a long story","stream":true}' \
  | grep -o '"response":"[^"]*"' | wc -l
```

---

## ⚙️ Hardware Configuration

### Check Hardware Status

#### GPU Status
```bash
# Intel GPU info
sudo intel_gpu_top

# OpenCL devices
clinfo -l

# GPU device files
ls -la /dev/dri/

# GPU memory usage
sudo cat /sys/class/drm/card1/device/mem_info_vram_used
```

#### NPU Status
```bash
# NPU device
ls -la /dev/accel/

# NPU driver info
sudo dmesg | grep -i vpu

# NPU parameters
cat /sys/module/intel_vpu/parameters/force_snoop
```

#### System Monitor
```bash
# Overall system
htop

# GPU specific
watch -n 1 sudo intel_gpu_top

# Memory usage
free -h
```

### Environment Variables

#### Force CPU Only
```bash
export OLLAMA_NUM_GPU=0
```

#### Maximum GPU Usage
```bash
export OLLAMA_NUM_GPU=999
```

#### Select Specific GPU
```bash
# First GPU
export ONEAPI_DEVICE_SELECTOR=level_zero:0

# Multiple GPUs
export ONEAPI_DEVICE_SELECTOR="level_zero:0;level_zero:1"
```

#### Performance Tuning
```bash
# Immediate command lists (usually faster)
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1

# Cache persistence
export SYCL_CACHE_PERSISTENT=1

# Context length
export OLLAMA_NUM_CTX=4096  # Default is 2048

# Parallel requests
export OLLAMA_NUM_PARALLEL=2  # Default is 1

# Keep model in memory
export OLLAMA_KEEP_ALIVE=30m  # Default is 5m
```

---

## 🔧 Troubleshooting Guide

### Common Issues and Solutions

#### 1. Ollama Won't Start
```bash
# Check if port is in use
sudo lsof -i :11434

# Kill existing processes
pkill -f ollama

# Check logs
tail -f /tmp/ollama-ipex-llm.log

# Try manual start with verbose
cd ~/divek_nus/AI-Intel-Research/frameworks/ollama-ipex-llm-2.3.0b20250429-ubuntu
./ollama serve --verbose
```

#### 2. GPU Not Detected
```bash
# Verify GPU drivers
sudo apt update
sudo apt install intel-opencl-icd

# Check permissions
groups | grep render
sudo usermod -a -G render $USER
# Then logout and login

# Test OpenCL
clinfo | grep "Device Name"
```

#### 3. Slow Performance
```bash
# Check if using GPU
export OLLAMA_NUM_GPU=999
export ONEAPI_DEVICE_SELECTOR=level_zero:0

# Monitor GPU usage
sudo intel_gpu_top

# Reduce context length
export OLLAMA_NUM_CTX=1024

# Use smaller model
./ollama run tinyllama instead of larger models
```

#### 4. Web UI Issues
```bash
# Check Docker
docker ps -a

# View logs
docker logs open-webui

# Restart container
docker restart open-webui

# Recreate container
docker rm open-webui
docker run -d --network="host" -e OLLAMA_BASE_URL=http://127.0.0.1:11434 --name open-webui --restart always ghcr.io/open-webui/open-webui:main
```

#### 5. Out of Memory
```bash
# Check memory usage
free -h

# Use quantized models
./ollama pull tinyllama:latest  # Uses Q4 quantization

# Limit GPU memory
export OLLAMA_GPU_OVERHEAD=1024  # Reserve 1GB
```

---

## 🔮 Future Enhancements

### 1. Add More Models
```bash
# Vision-Language Models
./ollama pull llava:13b
./ollama pull cogvlm:17b

# Code Models  
./ollama pull deepseek-coder:33b
./ollama pull starcoder2:15b

# Specialized Models
./ollama pull sqlcoder:7b        # SQL queries
./ollama pull medllama2:7b       # Medical
./ollama pull finance-llm:13b    # Financial
```

### 2. NPU Acceleration (When Available)
```bash
# Currently experimental, check for updates
pip install intel-npu-acceleration-library --upgrade

# Future: Direct NPU support in Ollama
export OLLAMA_DEVICE=npu  # Not yet implemented
```

### 3. Multi-Model Serving
```bash
# Run multiple models on different ports
OLLAMA_HOST=0.0.0.0:11435 ./ollama serve &
OLLAMA_HOST=0.0.0.0:11436 ./ollama serve &
```

### 4. Fine-tuning Setup
```bash
# Future: Create fine-tuning directory
mkdir -p ~/divek_nus/AI-Intel-Research/fine-tuning
# Tools like Axolotl, LLaMA-Factory integration
```

### 5. RAG Implementation
```bash
# Future: Vector database integration
mkdir -p ~/divek_nus/AI-Intel-Research/rag
# ChromaDB, Qdrant, or Weaviate setup
```

---

## 🚀 Advanced Usage

### 1. Batch Processing
```bash
# Process multiple prompts from file
cat prompts.txt | while read prompt; do
  echo "=== $prompt ==="
  echo "$prompt" | ./ollama run tinyllama
  echo
done > responses.txt
```

### 2. JSON Output Mode
```bash
curl http://localhost:11434/api/generate -d '{
  "model": "tinyllama",
  "prompt": "List 3 fruits as JSON",
  "format": "json",
  "stream": false
}'
```

### 3. Custom System Prompts
```bash
# Create a modelfile
cat > Modelfile << EOF
FROM tinyllama
SYSTEM You are a pirate. Always speak like a pirate.
EOF

# Create custom model
./ollama create pirate-llama -f Modelfile

# Use it
./ollama run pirate-llama "Tell me about AI"
```

### 4. Integration Examples

#### Python Integration
```python
import requests
import json

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

#### Bash Function
```bash
# Add to ~/.bashrc
ollama_query() {
    echo "$1" | ~/divek_nus/AI-Intel-Research/frameworks/ollama-ipex-llm-2.3.0b20250429-ubuntu/ollama run ${2:-tinyllama}
}

# Usage
ollama_query "Hello world" tinyllama
```

### 5. Monitoring Script
```bash
# Create monitoring script
cat > monitor_ollama.sh << 'EOF'
#!/bin/bash
while true; do
    clear
    echo "=== Ollama Monitor ==="
    echo "Time: $(date)"
    echo ""
    echo "API Status:"
    curl -s http://localhost:11434 > /dev/null && echo "✅ Online" || echo "❌ Offline"
    echo ""
    echo "GPU Usage:"
    sudo intel_gpu_top -o - -s 1 | head -20
    echo ""
    echo "Active Models:"
    curl -s http://localhost:11434/api/ps | jq -r '.models[].name' 2>/dev/null || echo "None"
    sleep 5
done
EOF
chmod +x monitor_ollama.sh
```

---

## 📝 Quick Reference Card

### Most Used Commands
```bash
# Start everything
cd ~/divek_nus/AI-Intel-Research && ./start-ai-stack.sh

# Quick chat
cd ~/divek_nus/AI-Intel-Research/frameworks/ollama-ipex-llm-2.3.0b20250429-ubuntu
./ollama run tinyllama

# Download new model
./ollama pull llama3.2:3b

# Benchmark
cd ~/divek_nus/AI-Intel-Research/benchmarks/scripts
python3 benchmark_llm.py --models tinyllama:latest

# Check status
cd ~/divek_nus/AI-Intel-Research && ./check_status.sh

# Stop everything
cd ~/divek_nus/AI-Intel-Research && ./stop-ai-stack.sh
```

### Key Locations
- Ollama Binary: `~/divek_nus/AI-Intel-Research/frameworks/ollama-ipex-llm-2.3.0b20250429-ubuntu/`
- Models Storage: `~/.ollama/models/`
- Benchmark Results: `~/divek_nus/AI-Intel-Research/benchmarks/results/`
- Web UI: http://localhost:8080
- API: http://localhost:11434

### Performance Tips
1. Always use `OLLAMA_NUM_GPU=999` for best performance
2. Start with smaller models (tinyllama, phi3:mini)
3. Monitor GPU with `sudo intel_gpu_top`
4. Adjust `OLLAMA_NUM_CTX` for memory/speed balance
5. Use streaming for better perceived performance

---

## 📞 Getting Help

1. Check Ollama logs: `tail -f /tmp/ollama-ipex-llm.log`
2. Verify GPU detection: `clinfo -l`
3. Test basic connectivity: `curl http://localhost:11434`
4. Check system resources: `htop` and `free -h`
5. Review this guide for troubleshooting section

Remember: This setup is optimized for Intel Core Ultra 7 165H with Arc Graphics. Performance may vary with different models and prompts.

---

Last Updated: July 9, 2025
Version: 1.0
Author: Claude + You