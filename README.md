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
- Launch Open WebUI on http://localhost:8080
- Display available models

### 2. Access the Services

- **Web UI**: http://localhost:8080 (ChatGPT-like interface)
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
├── vlm_env_description/ # VLM + RealSense camera demos
│   ├── realsense_vlm_demo.py      # Direct VLM scene analysis
│   ├── realsense_yolo_vlm_demo.py # YOLO + VLM pipeline
│   └── run_demo.sh               # Easy launcher
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
cd frameworks/ollama-ipex-llm-*/
./ollama pull <model-name>
```

### Recommended Models
- **Small/Fast**: `tinyllama` (1.1B params)
- **Balanced**: `phi3:mini` (3.8B params)
- **Vision**: `llava:7b` (7B params + vision)
- **Code**: `deepseek-coder:1.3b` (1.3B params)

### Vision Models for Camera Integration
- **LLaVA:7b**: Best for scene description (~15-17 tokens/s)
- **BakLLaVA:7b**: Alternative vision model
- **Qwen-VL**: Good for detailed object analysis
- **CogVLM**: Advanced vision-language understanding

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

## VLM + RealSense Camera Integration

### Real-time Environment Description
This project includes advanced demos that combine Vision Language Models with Intel RealSense D455 camera for real-time scene understanding.

#### Features
- **Real-time Scene Analysis**: Live camera feed processed by VLMs
- **Depth Integration**: Uses RealSense D455 depth information
- **Object Detection**: YOLO + VLM pipeline for detailed object descriptions
- **Intel GPU Acceleration**: Optimized for Intel Arc Graphics

#### Quick Start
```bash
cd vlm_env_description
./run_demo.sh
```

#### Available Demos
1. **Direct VLM Analysis** (`realsense_vlm_demo.py`)
   - Press SPACE to analyze current view
   - Press 'C' for continuous mode
   - Comprehensive scene descriptions

2. **YOLO + VLM Pipeline** (`realsense_yolo_vlm_demo.py`)
   - Object detection with YOLO
   - VLM describes each detected object
   - Depth information per object

#### Use Cases
- **Accessibility**: Environment description for visually impaired
- **Security**: Intelligent scene monitoring
- **Research**: Computer vision experimentation
- **Smart Home**: Context-aware automation

#### Performance
- **LLaVA:7b**: ~15-17 tokens/s on Intel Arc Graphics
- **Analysis Time**: 2-3 seconds per description
- **Camera**: 640x480 @ 30fps with depth

## Future Enhancements

- [ ] Direct NPU inference support
- [ ] Multi-model serving
- [ ] Distributed inference
- [ ] Fine-tuning capabilities
- [ ] Integration with LangChain/LlamaIndex
- [ ] Voice synthesis for VLM descriptions
- [ ] Web interface for remote camera access

## System Requirements

- Ubuntu 24.04 LTS
- Intel GPU drivers installed
- Docker (for Web UI)
- Python 3.8+
- 16GB+ RAM recommended

## License

This project uses various open-source components. Please refer to individual component licenses.