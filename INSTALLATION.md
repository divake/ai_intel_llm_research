# Installation Guide

## Prerequisites
- Ubuntu 24.04 LTS
- Intel Core Ultra processor or Intel Arc GPU
- Docker installed
- Python 3.8+

## Quick Setup

1. **Download IPEX-LLM Ollama**
   ```bash
   cd frameworks
   wget https://github.com/ipex-llm/ipex-llm/releases/download/v2.3.0-nightly/ollama-ipex-llm-2.3.0b20250429-ubuntu.tgz
   tar -xzf ollama-ipex-llm-2.3.0b20250429-ubuntu.tgz
   ```

2. **Make Scripts Executable**
   ```bash
   chmod +x *.sh
   chmod +x benchmarks/scripts/*.sh
   chmod +x benchmarks/scripts/*.py
   ```

3. **Start Services**
   ```bash
   ./start-ai-stack.sh
   ```

4. **Access Web UI**
   - Open browser to http://localhost:8080

## Environment Setup

Ensure you have Intel GPU drivers:
```bash
sudo apt update
sudo apt install intel-opencl-icd intel-level-zero-gpu level-zero
```

## First Model

The setup will download TinyLlama automatically. For more models:
```bash
./add_model.sh
```