#!/bin/bash

# VLM Environment Description Demo Runner
# Quick launch script for the VLM demos

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "================================================"
echo "VLM Environment Description Demo"
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
VLM_MODELS=$(echo "$MODELS" | grep -E "(llava|bakllava|cogvlm|qwen-vl)" | head -5)

if [ -z "$VLM_MODELS" ]; then
    echo "❌ No vision models found!"
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
    echo "✅ Found vision models:"
    echo "$VLM_MODELS" | sed 's/^/    /'
fi

# Check RealSense camera
echo "Checking RealSense camera..."
if ! python3 -c "import pyrealsense2 as rs; ctx = rs.context(); print(f'Found {len(ctx.query_devices())} RealSense devices')" 2>/dev/null; then
    echo "❌ RealSense camera not detected or pyrealsense2 not installed"
    echo "Make sure your camera is connected and drivers are installed"
    exit 1
else
    echo "✅ RealSense camera detected"
fi

# Menu selection
echo ""
echo "Select demo to run:"
echo "1) Direct VLM Analysis (recommended)"
echo "2) YOLO + VLM Pipeline"
echo "3) Exit"
echo ""
read -p "Enter choice (1-3): " choice

case $choice in
    1)
        echo "Starting Direct VLM Analysis demo..."
        cd "$SCRIPT_DIR"
        python3 realsense_vlm_demo.py
        ;;
    2)
        echo "Starting YOLO + VLM Pipeline demo..."
        cd "$SCRIPT_DIR"
        python3 realsense_yolo_vlm_demo.py
        ;;
    3)
        echo "Goodbye!"
        exit 0
        ;;
    *)
        echo "Invalid choice. Starting default demo..."
        cd "$SCRIPT_DIR"
        python3 realsense_vlm_demo.py
        ;;
esac