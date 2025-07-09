#!/bin/bash

# AI Intel Research Stack Launcher
# Starts IPEX-LLM Ollama and Open WebUI

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OLLAMA_DIR="$SCRIPT_DIR/frameworks/ipex-llm/ollama-ipex-llm-2.3.0b20250429-ubuntu"
OLLAMA_PID_FILE="/tmp/ollama-ipex-llm.pid"

echo "================================================"
echo "AI Intel Research Stack Launcher"
echo "================================================"

# Function to check if Ollama is running
check_ollama() {
    if curl -s http://localhost:11434 > /dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# Function to start Ollama
start_ollama() {
    echo "Starting IPEX-LLM Ollama server..."
    
    # Kill any existing Ollama processes
    if [ -f "$OLLAMA_PID_FILE" ]; then
        OLD_PID=$(cat "$OLLAMA_PID_FILE")
        kill $OLD_PID 2>/dev/null || true
    fi
    pkill -f "ollama serve" 2>/dev/null || true
    
    # Navigate to Ollama directory
    cd "$OLLAMA_DIR"
    
    # Set environment variables for Intel GPU
    export OLLAMA_NUM_GPU=999
    export no_proxy=localhost,127.0.0.1
    export ZES_ENABLE_SYSMAN=1
    export SYCL_CACHE_PERSISTENT=1
    export OLLAMA_KEEP_ALIVE=10m
    export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
    export ONEAPI_DEVICE_SELECTOR=level_zero:0
    
    # Source Intel oneAPI if available
    if [ -f "/opt/intel/oneapi/setvars.sh" ]; then
        echo "Sourcing Intel oneAPI environment..."
        source /opt/intel/oneapi/setvars.sh 2>/dev/null
    fi
    
    # Start Ollama in background
    nohup ./ollama serve > /tmp/ollama-ipex-llm.log 2>&1 &
    echo $! > "$OLLAMA_PID_FILE"
    
    # Wait for Ollama to be ready
    echo -n "Waiting for Ollama to start..."
    for i in {1..30}; do
        if check_ollama; then
            echo " Ready!"
            return 0
        fi
        echo -n "."
        sleep 1
    done
    echo " Failed to start!"
    return 1
}

# Function to start Open WebUI
start_webui() {
    echo "Starting Open WebUI..."
    
    # Check if container exists
    if docker ps -a | grep -q open-webui; then
        # Start existing container
        docker start open-webui > /dev/null 2>&1
    else
        # Create new container
        docker run -d \
            -p 3000:8080 \
            --add-host=host.docker.internal:host-gateway \
            --name open-webui \
            --restart always \
            ghcr.io/open-webui/open-webui:main > /dev/null 2>&1
    fi
    
    # Wait for WebUI to be ready
    echo -n "Waiting for WebUI to start..."
    for i in {1..30}; do
        if curl -s http://localhost:3000 > /dev/null 2>&1; then
            echo " Ready!"
            return 0
        fi
        echo -n "."
        sleep 1
    done
    echo " Failed to start!"
    return 1
}

# Main execution
main() {
    # Check if Ollama is already running
    if ! check_ollama; then
        start_ollama || exit 1
    else
        echo "Ollama is already running"
    fi
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        echo "Docker not found. Skipping WebUI."
    else
        start_webui || echo "WebUI failed to start, but Ollama is running"
    fi
    
    echo ""
    echo "================================================"
    echo "✅ AI Stack is running!"
    echo "================================================"
    echo "📌 Ollama API: http://localhost:11434"
    echo "📌 Web UI: http://localhost:3000"
    echo ""
    echo "Available models:"
    curl -s http://localhost:11434/api/tags | jq -r '.models[].name' 2>/dev/null || echo "Failed to list models"
    echo ""
    echo "To stop all services, run: $SCRIPT_DIR/stop-ai-stack.sh"
    echo "================================================"
}

# Trap to handle Ctrl+C
trap 'echo ""; echo "Use stop-ai-stack.sh to stop services"; exit 0' INT

# Run main function
main

# Keep script running if started in foreground
if [ -t 0 ]; then
    echo ""
    echo "Press Ctrl+C to exit (services will keep running)"
    while true; do
        sleep 1
    done
fi