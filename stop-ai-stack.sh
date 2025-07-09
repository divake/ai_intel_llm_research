#!/bin/bash

# AI Intel Research Stack Stopper

echo "================================================"
echo "Stopping AI Intel Research Stack"
echo "================================================"

# Stop Ollama
echo "Stopping Ollama..."
pkill -f "ollama serve" 2>/dev/null || true
if [ -f "/tmp/ollama-ipex-llm.pid" ]; then
    PID=$(cat "/tmp/ollama-ipex-llm.pid")
    kill $PID 2>/dev/null || true
    rm -f "/tmp/ollama-ipex-llm.pid"
fi

# Stop Open WebUI
if command -v docker &> /dev/null; then
    echo "Stopping Open WebUI..."
    docker stop open-webui 2>/dev/null || true
fi

echo ""
echo "✅ All AI services stopped"
echo "================================================"