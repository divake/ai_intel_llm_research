#!/usr/bin/env python3
"""
LLM Benchmark Script for Intel Hardware
Tests performance on CPU, GPU, and NPU (when available)
"""

import json
import time
import requests
import argparse
import subprocess
import os
from datetime import datetime
from typing import Dict, List, Any

class LLMBenchmark:
    def __init__(self, ollama_url: str = "http://localhost:11434"):
        self.ollama_url = ollama_url
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "system": self._get_system_info(),
            "benchmarks": []
        }
    
    def _get_system_info(self) -> Dict[str, str]:
        """Get system information"""
        info = {
            "hostname": os.uname().nodename,
            "cpu": "Intel Core Ultra 7 165H",
            "gpu": "Intel Arc Graphics",
            "npu": "Intel AI Boost (3rd Gen)",
            "ram": "64GB DDR5",
        }
        
        # Get Intel GPU info
        try:
            result = subprocess.run(['clinfo', '-l'], capture_output=True, text=True)
            info["opencl_platforms"] = result.stdout.strip()
        except:
            info["opencl_platforms"] = "Not available"
        
        return info
    
    def check_ollama_status(self) -> bool:
        """Check if Ollama is running"""
        try:
            response = requests.get(f"{self.ollama_url}/api/version")
            return response.status_code == 200
        except:
            return False
    
    def get_available_models(self) -> List[str]:
        """Get list of available models"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags")
            if response.status_code == 200:
                data = response.json()
                return [model["name"] for model in data.get("models", [])]
        except:
            pass
        return []
    
    def benchmark_inference(self, model: str, prompt: str, num_runs: int = 5) -> Dict[str, Any]:
        """Benchmark inference performance"""
        print(f"\nBenchmarking {model} with prompt: '{prompt[:50]}...'")
        
        results = {
            "model": model,
            "prompt": prompt,
            "runs": []
        }
        
        for i in range(num_runs):
            print(f"  Run {i+1}/{num_runs}...", end="", flush=True)
            
            start_time = time.time()
            first_token_time = None
            tokens = 0
            
            try:
                # Make streaming request
                response = requests.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": model,
                        "prompt": prompt,
                        "stream": True
                    },
                    stream=True
                )
                
                full_response = ""
                for line in response.iter_lines():
                    if line:
                        data = json.loads(line)
                        if first_token_time is None:
                            first_token_time = time.time()
                        
                        if "response" in data:
                            full_response += data["response"]
                            tokens += 1
                        
                        if data.get("done", False):
                            break
                
                end_time = time.time()
                
                # Calculate metrics
                total_time = end_time - start_time
                time_to_first_token = first_token_time - start_time if first_token_time else 0
                tokens_per_second = tokens / total_time if total_time > 0 else 0
                
                run_result = {
                    "run": i + 1,
                    "total_time": total_time,
                    "time_to_first_token": time_to_first_token,
                    "tokens_generated": tokens,
                    "tokens_per_second": tokens_per_second,
                    "response_length": len(full_response)
                }
                
                results["runs"].append(run_result)
                print(f" {tokens_per_second:.1f} tokens/s")
                
            except Exception as e:
                print(f" Error: {e}")
                results["runs"].append({
                    "run": i + 1,
                    "error": str(e)
                })
        
        # Calculate averages
        valid_runs = [r for r in results["runs"] if "error" not in r]
        if valid_runs:
            results["average"] = {
                "total_time": sum(r["total_time"] for r in valid_runs) / len(valid_runs),
                "time_to_first_token": sum(r["time_to_first_token"] for r in valid_runs) / len(valid_runs),
                "tokens_per_second": sum(r["tokens_per_second"] for r in valid_runs) / len(valid_runs),
            }
        
        return results
    
    def run_benchmarks(self, models: List[str] = None, prompts: List[str] = None):
        """Run benchmarks on specified models and prompts"""
        if not self.check_ollama_status():
            print("Error: Ollama is not running!")
            return
        
        available_models = self.get_available_models()
        if not available_models:
            print("Error: No models available!")
            return
        
        # Use provided models or all available
        if models:
            models = [m for m in models if m in available_models]
        else:
            models = available_models
        
        # Default prompts
        if not prompts:
            prompts = [
                "What is the capital of France?",
                "Explain quantum computing in simple terms.",
                "Write a haiku about artificial intelligence.",
                "List 5 benefits of renewable energy.",
                "What is 42 times 17?"
            ]
        
        print(f"Running benchmarks on {len(models)} model(s) with {len(prompts)} prompt(s)")
        
        for model in models:
            model_results = {
                "model": model,
                "prompts": []
            }
            
            for prompt in prompts:
                prompt_result = self.benchmark_inference(model, prompt)
                model_results["prompts"].append(prompt_result)
            
            self.results["benchmarks"].append(model_results)
    
    def save_results(self, filename: str = None):
        """Save benchmark results to JSON file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_results_{timestamp}.json"
        
        filepath = os.path.join(
            os.path.dirname(__file__), 
            "..", 
            "results", 
            filename
        )
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\nResults saved to: {filepath}")
        
        # Print summary
        self._print_summary()
    
    def _print_summary(self):
        """Print benchmark summary"""
        print("\n" + "="*60)
        print("BENCHMARK SUMMARY")
        print("="*60)
        
        for model_result in self.results["benchmarks"]:
            model = model_result["model"]
            print(f"\nModel: {model}")
            
            all_speeds = []
            for prompt_result in model_result["prompts"]:
                if "average" in prompt_result:
                    avg = prompt_result["average"]
                    all_speeds.append(avg["tokens_per_second"])
                    print(f"  Prompt: {prompt_result['prompt'][:40]}...")
                    print(f"    Avg Speed: {avg['tokens_per_second']:.1f} tokens/s")
                    print(f"    Avg Time to First Token: {avg['time_to_first_token']*1000:.1f}ms")
            
            if all_speeds:
                overall_avg = sum(all_speeds) / len(all_speeds)
                print(f"\n  Overall Average: {overall_avg:.1f} tokens/s")
        
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Benchmark LLM performance on Intel hardware")
    parser.add_argument("--models", nargs="+", help="Models to benchmark (default: all)")
    parser.add_argument("--prompts", nargs="+", help="Custom prompts to use")
    parser.add_argument("--runs", type=int, default=5, help="Number of runs per prompt (default: 5)")
    parser.add_argument("--output", help="Output filename for results")
    
    args = parser.parse_args()
    
    # Create benchmark instance
    benchmark = LLMBenchmark()
    
    # Run benchmarks
    benchmark.run_benchmarks(models=args.models, prompts=args.prompts)
    
    # Save results
    benchmark.save_results(args.output)


if __name__ == "__main__":
    main()