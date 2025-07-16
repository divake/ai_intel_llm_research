# AI Intel Research - Quick Reference Card

## 🚀 Essential Commands

### Start/Stop Services
```bash
# Start everything
cd ~/divek_nus/AI-Intel-Research
./start-ai-stack.sh

# Stop everything  
./stop-ai-stack.sh

# Check status
./check_status.sh
```

### Model Operations
```bash
# Navigate to Ollama directory
cd ~/divek_nus/AI-Intel-Research/frameworks/ollama-ipex-llm-2.3.0b20250429-ubuntu

# List models
./ollama list

# Download models
./ollama pull tinyllama              # 1.1B - Fastest
./ollama pull phi3:mini               # 3.8B - Balanced
./ollama pull llama3.2:3b             # 3B - Good quality
./ollama pull mistral:7b              # 7B - Better quality
./ollama pull llava:7b                # 7B - Vision capable
./ollama pull deepseek-coder:1.3b     # 1.3B - For coding

# Chat with model
./ollama run tinyllama

# One-shot query
echo "Your question here" | ./ollama run tinyllama

# Delete model
./ollama rm model-name
```

### Benchmarking
```bash
# Run benchmark
cd ~/divek_nus/AI-Intel-Research/benchmarks/scripts
python3 benchmark_llm.py --models tinyllama:latest

# Test hardware modes
./test_hardware_modes.sh

# View results
ls -la ../results/
cat ../results/benchmark_results_*.json | jq
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

## 🔧 Troubleshooting

### If Ollama won't start
```bash
pkill -f ollama
cd ~/divek_nus/AI-Intel-Research/frameworks/ollama-ipex-llm-2.3.0b20250429-ubuntu
./ollama serve
```

### If GPU not working
```bash
# Check GPU
clinfo -l
ls -la /dev/dri/

# Force GPU mode
export OLLAMA_NUM_GPU=999
export ONEAPI_DEVICE_SELECTOR=level_zero:0
```

### Monitor performance
```bash
# GPU usage
sudo intel_gpu_top

# System resources
htop

# Ollama logs
tail -f /tmp/ollama-ipex-llm.log
```

## 🎯 Performance Settings

```bash
# Maximum GPU usage (default)
export OLLAMA_NUM_GPU=999

# CPU only
export OLLAMA_NUM_GPU=0

# Increase context length
export OLLAMA_NUM_CTX=4096

# Keep model loaded longer
export OLLAMA_KEEP_ALIVE=30m
```

## 📍 Key Paths
- **Project Root**: `~/divek_nus/AI-Intel-Research/`
- **Ollama Binary**: `~/divek_nus/AI-Intel-Research/frameworks/ollama-ipex-llm-2.3.0b20250429-ubuntu/`
- **Models**: `~/.ollama/models/`
- **Results**: `~/divek_nus/AI-Intel-Research/benchmarks/results/`

## 💡 Tips
1. Start with smaller models for testing
2. Always check GPU is being used with `sudo intel_gpu_top`
3. TinyLlama is fastest, Mistral 7B is highest quality
4. Use Web UI for easier interaction
5. Run benchmarks to compare models

---
Keep this handy! 🎉