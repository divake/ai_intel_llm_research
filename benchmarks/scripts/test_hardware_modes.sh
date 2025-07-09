#!/bin/bash

# Hardware-specific testing script for Intel CPU/GPU/NPU
# Tests LLM performance on different accelerators

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="$SCRIPT_DIR/../results"
OLLAMA_DIR="$SCRIPT_DIR/../../frameworks/ipex-llm/ollama-ipex-llm-2.3.0b20250429-ubuntu"

echo "================================================"
echo "Intel Hardware Acceleration Test Suite"
echo "================================================"

# Create results directory
mkdir -p "$RESULTS_DIR"

# Function to stop Ollama
stop_ollama() {
    echo "Stopping Ollama..."
    pkill -f "ollama serve" 2>/dev/null || true
    sleep 2
}

# Function to start Ollama with specific configuration
start_ollama_with_config() {
    local mode=$1
    local device_selector=$2
    
    echo "Starting Ollama in $mode mode..."
    
    cd "$OLLAMA_DIR"
    
    # Base environment variables
    export no_proxy=localhost,127.0.0.1
    export ZES_ENABLE_SYSMAN=1
    export SYCL_CACHE_PERSISTENT=1
    export OLLAMA_KEEP_ALIVE=10m
    
    # Mode-specific configuration
    case $mode in
        "CPU")
            export OLLAMA_NUM_GPU=0  # Force CPU only
            unset ONEAPI_DEVICE_SELECTOR
            ;;
        "GPU")
            export OLLAMA_NUM_GPU=999  # Use all GPU layers
            export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
            export ONEAPI_DEVICE_SELECTOR="$device_selector"
            ;;
        "AUTO")
            export OLLAMA_NUM_GPU=999  # Let system decide
            unset ONEAPI_DEVICE_SELECTOR
            ;;
    esac
    
    # Start Ollama in background
    nohup ./ollama serve > "/tmp/ollama-$mode.log" 2>&1 &
    local pid=$!
    
    # Wait for Ollama to be ready
    echo -n "Waiting for Ollama to start..."
    for i in {1..30}; do
        if curl -s http://localhost:11434 > /dev/null 2>&1; then
            echo " Ready!"
            return 0
        fi
        echo -n "."
        sleep 1
    done
    echo " Failed!"
    return 1
}

# Function to run benchmark
run_benchmark() {
    local mode=$1
    local output_file="$RESULTS_DIR/benchmark_${mode}_$(date +%Y%m%d_%H%M%S).json"
    
    echo ""
    echo "Running benchmark in $mode mode..."
    
    # Run Python benchmark script
    python3 "$SCRIPT_DIR/benchmark_llm.py" \
        --models tinyllama:latest \
        --runs 3 \
        --output "$output_file"
    
    # Extract and display key metrics
    if [ -f "$output_file" ]; then
        echo ""
        echo "Results for $mode mode:"
        python3 -c "
import json
with open('$output_file', 'r') as f:
    data = json.load(f)
    for bench in data['benchmarks']:
        model = bench['model']
        speeds = []
        for prompt in bench['prompts']:
            if 'average' in prompt:
                speeds.append(prompt['average']['tokens_per_second'])
        if speeds:
            avg_speed = sum(speeds) / len(speeds)
            print(f'  {model}: {avg_speed:.1f} tokens/s')
"
    fi
}

# Main test sequence
main() {
    # Check if Ollama is installed
    if [ ! -f "$OLLAMA_DIR/ollama" ]; then
        echo "Error: Ollama not found at $OLLAMA_DIR"
        exit 1
    fi
    
    # Test CPU mode
    echo ""
    echo "1. Testing CPU-only mode"
    echo "------------------------"
    stop_ollama
    if start_ollama_with_config "CPU" ""; then
        run_benchmark "CPU"
    fi
    
    # Test GPU mode
    echo ""
    echo "2. Testing GPU mode"
    echo "-------------------"
    stop_ollama
    if start_ollama_with_config "GPU" "level_zero:0"; then
        run_benchmark "GPU"
    fi
    
    # Test Auto mode (CPU+GPU)
    echo ""
    echo "3. Testing Auto mode (CPU+GPU)"
    echo "------------------------------"
    stop_ollama
    if start_ollama_with_config "AUTO" ""; then
        run_benchmark "AUTO"
    fi
    
    # Stop Ollama
    stop_ollama
    
    # Summary
    echo ""
    echo "================================================"
    echo "Test Complete!"
    echo "Results saved in: $RESULTS_DIR"
    echo "================================================"
    
    # Display comparison
    echo ""
    echo "Performance Comparison:"
    echo "----------------------"
    
    # Find and compare results
    for mode in CPU GPU AUTO; do
        latest=$(ls -t "$RESULTS_DIR"/benchmark_${mode}_*.json 2>/dev/null | head -1)
        if [ -f "$latest" ]; then
            speed=$(python3 -c "
import json
with open('$latest', 'r') as f:
    data = json.load(f)
    speeds = []
    for bench in data['benchmarks']:
        for prompt in bench['prompts']:
            if 'average' in prompt:
                speeds.append(prompt['average']['tokens_per_second'])
    if speeds:
        print(f'{sum(speeds)/len(speeds):.1f}')
    else:
        print('N/A')
")
            echo "$mode: $speed tokens/s"
        fi
    done
}

# Run main function
main