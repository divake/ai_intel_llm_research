#!/usr/bin/env python3
"""
Test script to monitor hardware utilization during VLM inference
This will help us understand current CPU/GPU/NPU usage patterns
"""

import psutil
import time
import subprocess
import json
import threading
from datetime import datetime
import requests

class HardwareMonitor:
    def __init__(self):
        self.monitoring = False
        self.data = []
        
    def get_cpu_usage(self):
        """Get CPU usage percentage"""
        return psutil.cpu_percent(interval=1)
    
    def get_memory_usage(self):
        """Get memory usage"""
        mem = psutil.virtual_memory()
        return {
            'used_gb': mem.used / (1024**3),
            'total_gb': mem.total / (1024**3),
            'percent': mem.percent
        }
    
    def get_gpu_usage(self):
        """Get Intel GPU usage if available"""
        try:
            # Try intel_gpu_top command
            result = subprocess.run(['intel_gpu_top', '-o', '-', '-s', '1'], 
                                  capture_output=True, text=True, timeout=2)
            
            if result.returncode == 0:
                # Parse intel_gpu_top output
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if 'Render/3D' in line:
                        # Extract GPU usage percentage
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if '%' in part:
                                return float(part.replace('%', ''))
            return 0.0
        except:
            return 0.0
    
    def get_npu_usage(self):
        """Get NPU usage if available"""
        try:
            # Check if NPU device exists
            result = subprocess.run(['ls', '/dev/accel/'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                # NPU device exists but we need proper monitoring tool
                # For now, return 0 as we don't have direct NPU monitoring
                return 0.0
            return 0.0
        except:
            return 0.0
    
    def get_temperature(self):
        """Get system temperature"""
        try:
            temps = psutil.sensors_temperatures()
            if 'coretemp' in temps:
                return max([t.current for t in temps['coretemp']])
            return 0.0
        except:
            return 0.0
    
    def get_power_draw(self):
        """Estimate power draw (basic implementation)"""
        try:
            # This is a rough estimation based on CPU usage
            cpu_usage = psutil.cpu_percent()
            # Base power (~15W) + additional based on usage
            estimated_power = 15 + (cpu_usage / 100 * 30)
            return estimated_power
        except:
            return 0.0
    
    def collect_metrics(self):
        """Collect all hardware metrics"""
        timestamp = datetime.now().isoformat()
        
        metrics = {
            'timestamp': timestamp,
            'cpu_percent': self.get_cpu_usage(),
            'memory': self.get_memory_usage(),
            'gpu_percent': self.get_gpu_usage(),
            'npu_percent': self.get_npu_usage(),
            'temperature_c': self.get_temperature(),
            'power_draw_w': self.get_power_draw()
        }
        
        return metrics
    
    def monitor_during_vlm(self, duration=30):
        """Monitor hardware during VLM inference"""
        print(f"Starting hardware monitoring for {duration} seconds...")
        print("Please start your VLM demo in another terminal:")
        print("cd vlm_env_description && python3 realsense_vlm_demo.py")
        print("-" * 60)
        
        start_time = time.time()
        
        while time.time() - start_time < duration:
            metrics = self.collect_metrics()
            self.data.append(metrics)
            
            # Print real-time metrics
            print(f"\r[{metrics['timestamp'].split('T')[1][:8]}] "
                  f"CPU: {metrics['cpu_percent']:5.1f}% | "
                  f"GPU: {metrics['gpu_percent']:5.1f}% | "
                  f"NPU: {metrics['npu_percent']:5.1f}% | "
                  f"MEM: {metrics['memory']['percent']:5.1f}% | "
                  f"TEMP: {metrics['temperature_c']:5.1f}°C | "
                  f"PWR: {metrics['power_draw_w']:5.1f}W", 
                  end="", flush=True)
            
            time.sleep(1)
        
        print("\nMonitoring complete!")
        return self.data
    
    def test_vlm_load(self):
        """Test VLM inference load"""
        print("Testing VLM inference load...")
        
        # Check if Ollama is running
        try:
            response = requests.get("http://localhost:11434/api/tags")
            if response.status_code != 200:
                print("❌ Ollama is not running! Please start it first.")
                return None
        except:
            print("❌ Cannot connect to Ollama!")
            return None
        
        # Start monitoring in background
        monitor_thread = threading.Thread(target=self.background_monitor)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        # Simulate VLM inference
        test_prompt = "Describe this scene in detail including all objects, people, and activities you can see."
        
        try:
            print("Sending test prompt to VLM...")
            start_time = time.time()
            
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "llava:7b",
                    "prompt": test_prompt,
                    "stream": False
                },
                timeout=30
            )
            
            end_time = time.time()
            inference_time = end_time - start_time
            
            if response.status_code == 200:
                print(f"✅ VLM inference completed in {inference_time:.2f}s")
                return True
            else:
                print(f"❌ VLM inference failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Error during VLM inference: {e}")
            return False
    
    def background_monitor(self):
        """Background monitoring for VLM test"""
        for _ in range(30):  # Monitor for 30 seconds
            metrics = self.collect_metrics()
            self.data.append(metrics)
            time.sleep(1)
    
    def save_results(self, filename="hardware_monitoring_results.json"):
        """Save monitoring results"""
        with open(filename, 'w') as f:
            json.dump(self.data, f, indent=2)
        print(f"Results saved to {filename}")
    
    def analyze_results(self):
        """Analyze monitoring results"""
        if not self.data:
            print("No data to analyze!")
            return
        
        print("\n" + "="*60)
        print("HARDWARE UTILIZATION ANALYSIS")
        print("="*60)
        
        # Calculate averages
        cpu_avg = sum(d['cpu_percent'] for d in self.data) / len(self.data)
        gpu_avg = sum(d['gpu_percent'] for d in self.data) / len(self.data)
        npu_avg = sum(d['npu_percent'] for d in self.data) / len(self.data)
        mem_avg = sum(d['memory']['percent'] for d in self.data) / len(self.data)
        temp_avg = sum(d['temperature_c'] for d in self.data) / len(self.data)
        power_avg = sum(d['power_draw_w'] for d in self.data) / len(self.data)
        
        # Calculate peaks
        cpu_peak = max(d['cpu_percent'] for d in self.data)
        gpu_peak = max(d['gpu_percent'] for d in self.data)
        npu_peak = max(d['npu_percent'] for d in self.data)
        mem_peak = max(d['memory']['percent'] for d in self.data)
        temp_peak = max(d['temperature_c'] for d in self.data)
        power_peak = max(d['power_draw_w'] for d in self.data)
        
        print(f"CPU Usage    - Avg: {cpu_avg:5.1f}%  Peak: {cpu_peak:5.1f}%")
        print(f"GPU Usage    - Avg: {gpu_avg:5.1f}%  Peak: {gpu_peak:5.1f}%")
        print(f"NPU Usage    - Avg: {npu_avg:5.1f}%  Peak: {npu_peak:5.1f}%")
        print(f"Memory Usage - Avg: {mem_avg:5.1f}%  Peak: {mem_peak:5.1f}%")
        print(f"Temperature  - Avg: {temp_avg:5.1f}°C Peak: {temp_peak:5.1f}°C")
        print(f"Power Draw   - Avg: {power_avg:5.1f}W  Peak: {power_peak:5.1f}W")
        
        # Determine primary processing unit
        if gpu_avg > cpu_avg and gpu_avg > npu_avg:
            primary_unit = "GPU"
        elif npu_avg > cpu_avg and npu_avg > gpu_avg:
            primary_unit = "NPU"
        else:
            primary_unit = "CPU"
        
        print(f"\nPrimary Processing Unit: {primary_unit}")
        print(f"Workload Distribution: CPU({cpu_avg:.1f}%) GPU({gpu_avg:.1f}%) NPU({npu_avg:.1f}%)")
        
        # Simple offload recommendation
        total_load = max(cpu_avg, gpu_avg, npu_avg)
        if total_load > 80:
            recommendation = "🔴 OFFLOAD TO SERVER"
        elif total_load > 60:
            recommendation = "🟡 MONITOR - PREPARE OFFLOAD"
        else:
            recommendation = "🟢 EDGE PROCESSING OK"
        
        print(f"Recommendation: {recommendation}")
        print("="*60)

def main():
    monitor = HardwareMonitor()
    
    print("Intel Hardware Utilization Test")
    print("="*40)
    print("1. Test VLM inference load")
    print("2. Manual monitoring (30s)")
    print("3. Exit")
    
    choice = input("\nEnter choice (1-3): ")
    
    if choice == "1":
        monitor.test_vlm_load()
        time.sleep(2)  # Wait for background monitoring
        monitor.analyze_results()
        monitor.save_results()
        
    elif choice == "2":
        monitor.monitor_during_vlm()
        monitor.analyze_results()
        monitor.save_results()
        
    else:
        print("Goodbye!")

if __name__ == "__main__":
    main()