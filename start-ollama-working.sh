#!/bin/bash
# Simple Ollama Startup Script (Using Standard Ollama)
# Created: October 15, 2025
# This script starts the working version of Ollama + Web UI

echo "================================================"
echo "Starting Ollama (Standard Version - Working)"
echo "================================================"

# Stop any old broken IPEX-LLM processes
echo "Cleaning up old processes..."
pkill -9 -f "ollama-ipex-llm" 2>/dev/null
sudo lsof -ti:11434 | xargs sudo kill -9 2>/dev/null
sleep 2

# Start standard Ollama server
echo "Starting Ollama server..."
/usr/local/bin/ollama serve &
OLLAMA_PID=$!
sleep 5

# Check if Ollama started successfully
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "✅ Ollama server is running (PID: $OLLAMA_PID)"

    # Show available models
    echo ""
    echo "Available models:"
    /usr/local/bin/ollama list
else
    echo "❌ Ollama failed to start"
    exit 1
fi

# Start Open WebUI
echo ""
echo "Starting Open WebUI..."
docker start open-webui
sleep 3

if docker ps | grep -q open-webui; then
    echo "✅ Open WebUI is running"
else
    echo "❌ Open WebUI failed to start"
fi

echo ""
echo "================================================"
echo "✅ AI Stack is Ready!"
echo "================================================"
echo "📌 Ollama API: http://localhost:11434"
echo "📌 Web UI: http://localhost:8080"
echo ""
echo "To stop everything, run: ./stop-ollama-working.sh"
echo "================================================"
