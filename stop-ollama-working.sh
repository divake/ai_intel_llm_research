#!/bin/bash
# Stop Ollama Services
# Created: October 15, 2025

echo "================================================"
echo "Stopping Ollama Services"
echo "================================================"

# Stop Ollama server
echo "Stopping Ollama server..."
pkill -f "/usr/local/bin/ollama serve"
sudo lsof -ti:11434 | xargs sudo kill -9 2>/dev/null
sleep 2

# Stop Open WebUI
echo "Stopping Open WebUI..."
docker stop open-webui

echo ""
echo "✅ All services stopped"
echo "================================================"
