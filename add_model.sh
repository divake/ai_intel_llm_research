#!/bin/bash

# Model addition helper script

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OLLAMA_DIR="$SCRIPT_DIR/frameworks/ollama-ipex-llm-2.3.0b20250429-ubuntu"

echo "================================================"
echo "AI Intel Research - Model Manager"
echo "================================================"
echo ""

# Function to display menu
show_menu() {
    echo "Popular Models:"
    echo ""
    echo "Small Models (< 2GB):"
    echo "  1) tinyllama (1.1B) - Already installed"
    echo "  2) phi3:mini (3.8B) - Microsoft's efficient model"
    echo "  3) gemma2:2b (2B) - Google's small model"
    echo ""
    echo "Medium Models (3-5GB):"
    echo "  4) llama3.2:3b (3B) - Latest Llama 3.2"
    echo "  5) deepseek-coder:1.3b (1.3B) - Fast code completion"
    echo "  6) stablelm2:1.6b (1.6B) - Stable small model"
    echo ""
    echo "Large Models (5-8GB):"
    echo "  7) mistral:7b (7B) - High quality general model"
    echo "  8) llama3.1:8b (8B) - Latest Llama 3.1"
    echo "  9) codellama:7b (7B) - Meta's code model"
    echo ""
    echo "Vision Models:"
    echo "  10) llava:7b (7B) - General vision model"
    echo "  11) bakllava:7b (7B) - Alternative vision model"
    echo ""
    echo "Specialized:"
    echo "  12) deepseek-math:7b - Mathematics"
    echo "  13) sqlcoder:7b - SQL queries"
    echo "  14) phi3:medium - Reasoning tasks"
    echo ""
    echo "Other Options:"
    echo "  15) Enter custom model name"
    echo "  16) List installed models"
    echo "  17) Remove a model"
    echo "  0) Exit"
    echo ""
}

# Function to pull model
pull_model() {
    local model=$1
    echo ""
    echo "Downloading $model..."
    cd "$OLLAMA_DIR"
    ./ollama pull "$model"
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ Successfully downloaded $model"
        echo ""
        echo "Test it with:"
        echo "  cd $OLLAMA_DIR"
        echo "  ./ollama run $model"
    else
        echo ""
        echo "❌ Failed to download $model"
    fi
}

# Function to list models
list_models() {
    echo ""
    echo "Installed models:"
    cd "$OLLAMA_DIR"
    ./ollama list
}

# Function to remove model
remove_model() {
    list_models
    echo ""
    read -p "Enter model name to remove: " model_name
    if [ ! -z "$model_name" ]; then
        cd "$OLLAMA_DIR"
        ./ollama rm "$model_name"
    fi
}

# Main loop
while true; do
    show_menu
    read -p "Select option (0-17): " choice
    
    case $choice in
        1) echo "tinyllama is already installed!" ;;
        2) pull_model "phi3:mini" ;;
        3) pull_model "gemma2:2b" ;;
        4) pull_model "llama3.2:3b" ;;
        5) pull_model "deepseek-coder:1.3b" ;;
        6) pull_model "stablelm2:1.6b" ;;
        7) pull_model "mistral:7b" ;;
        8) pull_model "llama3.1:8b" ;;
        9) pull_model "codellama:7b" ;;
        10) pull_model "llava:7b" ;;
        11) pull_model "bakllava:7b" ;;
        12) pull_model "deepseek-math:7b" ;;
        13) pull_model "sqlcoder:7b" ;;
        14) pull_model "phi3:medium" ;;
        15) 
            read -p "Enter model name (e.g., model:tag): " custom_model
            if [ ! -z "$custom_model" ]; then
                pull_model "$custom_model"
            fi
            ;;
        16) list_models ;;
        17) remove_model ;;
        0) 
            echo "Exiting..."
            break 
            ;;
        *) echo "Invalid option. Please try again." ;;
    esac
    
    echo ""
    read -p "Press Enter to continue..."
    clear
done

echo ""
echo "Model management complete!"
echo ""
echo "To use your models:"
echo "1. Start the AI stack: $SCRIPT_DIR/start-ai-stack.sh"
echo "2. Access via Web UI: http://localhost:8080"
echo "3. Or use command line: cd $OLLAMA_DIR && ./ollama run <model-name>"
echo ""