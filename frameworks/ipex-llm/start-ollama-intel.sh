#!/bin/bash

# Start Ollama with Intel GPU acceleration

echo "Starting IPEX-LLM Ollama server with Intel hardware acceleration..."

# Navigate to Ollama directory
cd "$(dirname "$0")/ollama-ipex-llm-2.3.0b20250429-ubuntu"

# Set environment variables for Intel GPU
export OLLAMA_NUM_GPU=999  # Use all GPU layers
export no_proxy=localhost,127.0.0.1
export ZES_ENABLE_SYSMAN=1
export SYCL_CACHE_PERSISTENT=1

# Source Intel oneAPI if available
if [ -f "/opt/intel/oneapi/setvars.sh" ]; then
    echo "Sourcing Intel oneAPI environment..."
    source /opt/intel/oneapi/setvars.sh
fi

# Optional performance tuning
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
export ONEAPI_DEVICE_SELECTOR=level_zero:0  # Use first Intel GPU

# Start Ollama server
echo "Starting Ollama server..."
echo "Server will be available at http://localhost:11434"
echo "Press Ctrl+C to stop"

./start-ollama.sh