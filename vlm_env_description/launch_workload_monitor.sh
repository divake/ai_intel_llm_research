#!/bin/bash

# Intel Workload Intelligence Monitor Launcher
# This script launches the advanced VLM workload monitoring system

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "================================================"
echo "Intel Workload Intelligence Monitor"
echo "VLM + Real-time Hardware Monitoring"
echo "With Predictive Offloading Decisions"
echo "================================================"

# Check if Ollama is running
echo "Checking Ollama status..."
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "❌ Ollama is not running!"
    echo "Starting AI stack..."
    cd "$PROJECT_ROOT"
    ./start-ai-stack.sh
    
    # Wait for Ollama to be ready
    echo "Waiting for Ollama to be ready..."
    for i in {1..30}; do
        if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
            echo "✅ Ollama is ready!"
            break
        fi
        sleep 1
    done
else
    echo "✅ Ollama is running"
fi

# Check for vision models
echo "Checking for vision models..."
MODELS=$(curl -s http://localhost:11434/api/tags | jq -r '.models[].name' 2>/dev/null)
VLM_MODELS=$(echo "$MODELS" | grep -i llava | head -1)

if [ -z "$VLM_MODELS" ]; then
    echo "❌ No LLaVA models found!"
    echo "Installing LLaVA:7b model..."
    cd "$PROJECT_ROOT/frameworks/ollama-ipex-llm-"*
    ./ollama pull llava:7b
    
    if [ $? -eq 0 ]; then
        echo "✅ LLaVA:7b installed successfully"
    else
        echo "❌ Failed to install vision model"
        exit 1
    fi
else
    echo "✅ Found vision model: $VLM_MODELS"
fi

# Check RealSense camera
echo "Checking RealSense camera..."
if ! python3 -c "import pyrealsense2 as rs; ctx = rs.context(); print(f'Found {len(ctx.query_devices())} RealSense devices')" 2>/dev/null; then
    echo "❌ RealSense camera not detected"
    echo "Make sure your camera is connected and drivers are installed"
    exit 1
else
    echo "✅ RealSense camera detected"
fi

# Check dependencies
echo "Checking Python dependencies..."
python3 -c "import sklearn, cv2, numpy, requests" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Missing dependencies. Installing..."
    pip install scikit-learn opencv-python numpy requests
fi

echo ""
echo "================================================"
echo "🚀 LAUNCHING INTEL WORKLOAD INTELLIGENCE MONITOR"
echo "================================================"
echo ""
echo "This system will show:"
echo "📊 Real-time hardware monitoring (CPU/GPU/NPU/Memory/Temperature)"
echo "🤖 VLM scene analysis with Intel RealSense camera"
echo "🔮 Predictive offloading decisions with uncertainty quantification"
echo "📈 Conformal prediction for workload forecasting"
echo ""
echo "Controls:"
echo "  SPACE - Trigger scene analysis"
echo "  C     - Toggle continuous analysis mode"
echo "  Q     - Quit"
echo ""
echo "Research Focus:"
echo "  - Hardware workload distribution (CPU/GPU/NPU)"
echo "  - Predictive edge-to-cloud offloading"
echo "  - Uncertainty quantification in AI workloads"
echo "  - Real-time performance monitoring"
echo ""
echo "Press ENTER to continue..."
read

# Launch the monitor
cd "$SCRIPT_DIR"
python3 intel_workload_monitor.py