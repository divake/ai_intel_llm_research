#!/bin/bash

# AI Intel Research Status Check

echo "================================================"
echo "AI Intel Research - System Status"
echo "================================================"
echo ""

# Check Ollama
echo "1. Ollama Status:"
if curl -s http://localhost:11434 > /dev/null 2>&1; then
    echo "   ✅ Ollama is running"
    echo "   Available models:"
    curl -s http://localhost:11434/api/tags | jq -r '.models[].name' | sed 's/^/      - /'
else
    echo "   ❌ Ollama is not running"
fi

echo ""

# Check Web UI
echo "2. Web UI Status:"
if docker ps | grep -q open-webui; then
    echo "   ✅ Web UI is running at http://localhost:8080"
else
    echo "   ❌ Web UI is not running"
fi

echo ""

# Check Intel GPU
echo "3. Intel GPU Status:"
if [ -e /dev/dri/renderD128 ]; then
    echo "   ✅ Intel GPU detected"
    clinfo -l 2>/dev/null | grep "Intel.*Graphics" | sed 's/^/      /'
else
    echo "   ❌ Intel GPU not detected"
fi

echo ""

# Check NPU
echo "4. Intel NPU Status:"
if [ -e /dev/accel/accel0 ]; then
    echo "   ✅ Intel NPU detected"
else
    echo "   ❌ Intel NPU not detected"
fi

echo ""

# Latest benchmark results
echo "5. Latest Benchmark Results:"
latest_result=$(ls -t AI-Intel-Research/benchmarks/results/benchmark_results_*.json 2>/dev/null | head -1)
if [ -f "$latest_result" ]; then
    echo "   Latest: $(basename $latest_result)"
    python3 -c "
import json
with open('$latest_result', 'r') as f:
    data = json.load(f)
    for bench in data['benchmarks']:
        model = bench['model']
        speeds = []
        for prompt in bench['prompts']:
            if 'average' in prompt:
                speeds.append(prompt['average']['tokens_per_second'])
        if speeds:
            avg_speed = sum(speeds) / len(speeds)
            print(f'   {model}: {avg_speed:.1f} tokens/s')
"
else
    echo "   No benchmark results found"
fi

echo ""
echo "================================================"