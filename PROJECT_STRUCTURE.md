# AI Intel Research Project Structure

## Directory Organization

```
AI-Intel-Research/
├── models/              # Model storage and management
│   ├── llm/            # Large Language Models
│   │   ├── text/       # Text-only models (Llama, Mistral, etc.)
│   │   ├── multimodal/ # Text+Image models (LLaVA, etc.)
│   │   └── code/       # Code-specific models (DeepSeek Coder, etc.)
│   ├── vlm/            # Vision Language Models
│   │   ├── image/      # Image understanding models
│   │   └── video/      # Video understanding models
│   └── specialized/    # Domain-specific models
│
├── frameworks/         # Different inference frameworks
│   ├── ipex-llm/      # Intel IPEX-LLM (Ollama)
│   ├── llamacpp/      # llama.cpp implementations
│   ├── vllm/          # vLLM for high-throughput
│   └── transformers/  # HuggingFace Transformers
│
├── benchmarks/        # Performance testing
│   ├── results/       # Benchmark results (JSON, CSV)
│   └── scripts/       # Benchmarking scripts
│
├── configs/           # Configuration files
├── utils/             # Utility scripts
└── docs/              # Documentation
```

## Framework Details

### IPEX-LLM (Current Implementation)
- Intel optimized Ollama fork
- Supports CPU, GPU, NPU acceleration
- Best for quick deployment

### Future Frameworks
- **llama.cpp**: Direct C++ implementation
- **vLLM**: Production-grade serving
- **Transformers**: Research and fine-tuning

## Model Categories

### LLM Text Models
- Llama 3.2, 3.3
- Mistral 7B
- Phi-3
- TinyLlama (testing)

### Multimodal Models
- LLaVA
- Qwen-VL
- CogVLM

### Code Models
- DeepSeek Coder
- CodeLlama
- StarCoder

## Hardware Targets
- Intel CPU (with AVX-512)
- Intel Arc GPU (via OpenCL/XPU)
- Intel NPU (via acceleration library)