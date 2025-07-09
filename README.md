# AI Intel Research - LLM/VLM on Intel Hardware

## Overview

This project provides a comprehensive framework for running Large Language Models (LLMs) and Vision Language Models (VLMs) on Intel hardware, specifically optimized for:
- Intel Core Ultra 7 165H processor
- Intel Arc Graphics (integrated GPU)
- Intel AI Boost NPU (3rd Gen)

## Quick Start

### 1. Start the AI Stack
```bash
./start-ai-stack.sh
```

This will:
- Start IPEX-LLM Ollama server with Intel GPU acceleration
- Launch Open WebUI on http://localhost:3000
- Display available models

### 2. Access the Services

- **Web UI**: http://localhost:3000 (ChatGPT-like interface)
- **API**: http://localhost:11434 (Ollama API)

### 3. Stop Services
```bash
./stop-ai-stack.sh
```

## Project Structure

```
AI-Intel-Research/
├── models/              # Model storage
│   ├── llm/            # Language models
│   └── vlm/            # Vision-language models
├── frameworks/         # Inference frameworks
│   └── ipex-llm/      # Intel optimized Ollama
├── benchmarks/        # Performance testing
│   ├── scripts/       # Benchmark scripts
│   └── results/       # Test results
└── configs/           # Configuration files
```

## Available Scripts

### Launch Scripts
- `start-ai-stack.sh` - Start Ollama and Web UI
- `stop-ai-stack.sh` - Stop all services

### Benchmark Scripts
- `benchmarks/scripts/benchmark_llm.py` - Run performance benchmarks
- `benchmarks/scripts/test_hardware_modes.sh` - Test CPU/GPU/Auto modes

## Running Benchmarks

### Basic Benchmark
```bash
cd benchmarks/scripts
python3 benchmark_llm.py --models tinyllama:latest
```

### Hardware Mode Testing
```bash
cd benchmarks/scripts
./test_hardware_modes.sh
```

This will test performance on:
1. CPU-only mode
2. GPU-accelerated mode
3. Auto mode (CPU+GPU)

## Adding New Models

### Download Models
```bash
cd frameworks/ipex-llm/ollama-ipex-llm-*/
./ollama pull <model-name>
```

### Recommended Models
- **Small/Fast**: `tinyllama` (1.1B params)
- **Balanced**: `phi3:mini` (3.8B params)
- **Vision**: `llava:7b` (7B params + vision)
- **Code**: `deepseek-coder:1.3b` (1.3B params)

## Environment Variables

### GPU Optimization
```bash
export OLLAMA_NUM_GPU=999  # Use all GPU layers
export ONEAPI_DEVICE_SELECTOR=level_zero:0  # Select GPU
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1  # Performance
```

### NPU Usage (Experimental)
Currently, NPU support is through Intel NPU Acceleration Library only.
Direct Ollama NPU support is pending.

## Performance Tips

1. **GPU Acceleration**: Ensure `OLLAMA_NUM_GPU=999` for best performance
2. **Memory**: Models under 4B parameters work best with 8GB VRAM
3. **Context Length**: Adjust with `OLLAMA_NUM_CTX` (default: 2048)

## Troubleshooting

### Ollama not detecting GPU
```bash
# Check GPU availability
clinfo -l
ls -la /dev/dri/

# Restart with explicit GPU selection
export ONEAPI_DEVICE_SELECTOR=level_zero:0
```

### Web UI not accessible
```bash
# Check Docker status
docker ps
docker logs open-webui
```

### Performance issues
- Ensure Intel GPU drivers are up to date
- Check GPU utilization with `sudo intel_gpu_top`
- Monitor system resources with `htop`

## Future Enhancements

- [ ] Direct NPU inference support
- [ ] Multi-model serving
- [ ] Distributed inference
- [ ] Fine-tuning capabilities
- [ ] Integration with LangChain/LlamaIndex

## System Requirements

- Ubuntu 24.04 LTS
- Intel GPU drivers installed
- Docker (for Web UI)
- Python 3.8+
- 16GB+ RAM recommended

## License

This project uses various open-source components. Please refer to individual component licenses.