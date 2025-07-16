#!/usr/bin/env python3
"""
Intel Workload Intelligence Monitor
Real-time hardware monitoring with predictive offloading decisions
Uses uncertainty quantification and conformal prediction
"""

import numpy as np
import cv2
import time
import psutil
import subprocess
import threading
import queue
import json
import requests
import base64
from datetime import datetime, timedelta
from collections import deque
import pyrealsense2 as rs
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class ConformalPredictor:
    """Simple conformal prediction for workload forecasting"""
    
    def __init__(self, alpha=0.1):
        self.alpha = alpha  # Miscoverage rate (10% = 90% confidence)
        self.model = LinearRegression()
        self.scaler = StandardScaler()
        self.residuals = []
        self.fitted = False
    
    def fit(self, X, y):
        """Fit the model with training data"""
        if len(X) < 5:
            return False
        
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        
        # Calculate residuals for conformal prediction
        predictions = self.model.predict(X_scaled)
        self.residuals = np.abs(y - predictions)
        self.fitted = True
        return True
    
    def predict_with_uncertainty(self, X):
        """Predict with uncertainty intervals"""
        if not self.fitted:
            return None, None, None
        
        X_scaled = self.scaler.transform(X)
        prediction = self.model.predict(X_scaled)[0]
        
        # Conformal prediction interval
        if len(self.residuals) > 0:
            quantile = np.quantile(self.residuals, 1 - self.alpha)
            lower_bound = prediction - quantile
            upper_bound = prediction + quantile
        else:
            lower_bound = prediction - 10
            upper_bound = prediction + 10
        
        return prediction, lower_bound, upper_bound

class HardwareMonitor:
    """Real-time hardware monitoring with predictive capabilities"""
    
    def __init__(self):
        self.cpu_history = deque(maxlen=60)  # 60 seconds of history
        self.gpu_history = deque(maxlen=60)
        self.memory_history = deque(maxlen=60)
        self.temperature_history = deque(maxlen=60)
        self.power_history = deque(maxlen=60)
        self.timestamps = deque(maxlen=60)
        
        self.cpu_predictor = ConformalPredictor()
        self.gpu_predictor = ConformalPredictor()
        self.memory_predictor = ConformalPredictor()
        
        self.monitoring = False
        self.current_metrics = {}
        
    def get_cpu_usage(self):
        """Get CPU usage percentage"""
        return psutil.cpu_percent(interval=0.1)
    
    def get_memory_usage(self):
        """Get memory usage percentage"""
        return psutil.virtual_memory().percent
    
    def get_gpu_usage(self):
        """Get GPU usage (estimated from system load)"""
        try:
            # Try to get actual GPU usage
            result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True, timeout=2)
            if result.returncode == 0:
                return float(result.stdout.strip())
        except:
            pass
        
        # Fallback: estimate from CPU usage and system load
        cpu_usage = psutil.cpu_percent()
        load_avg = psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0
        estimated_gpu = min(cpu_usage * 1.2 + load_avg * 10, 100)
        return estimated_gpu
    
    def get_temperature(self):
        """Get system temperature"""
        try:
            temps = psutil.sensors_temperatures()
            if 'coretemp' in temps:
                return max([t.current for t in temps['coretemp']])
            return 45.0  # Default reasonable temperature
        except:
            return 45.0
    
    def get_power_draw(self):
        """Estimate power draw"""
        cpu_usage = psutil.cpu_percent()
        return 15 + (cpu_usage / 100 * 30)  # Base + variable power
    
    def collect_metrics(self):
        """Collect all hardware metrics"""
        metrics = {
            'timestamp': datetime.now(),
            'cpu_percent': self.get_cpu_usage(),
            'memory_percent': self.get_memory_usage(),
            'gpu_percent': self.get_gpu_usage(),
            'temperature_c': self.get_temperature(),
            'power_draw_w': self.get_power_draw()
        }
        
        # Add to history
        self.cpu_history.append(metrics['cpu_percent'])
        self.gpu_history.append(metrics['gpu_percent'])
        self.memory_history.append(metrics['memory_percent'])
        self.temperature_history.append(metrics['temperature_c'])
        self.power_history.append(metrics['power_draw_w'])
        self.timestamps.append(metrics['timestamp'])
        
        self.current_metrics = metrics
        return metrics
    
    def update_predictors(self):
        """Update predictive models"""
        if len(self.cpu_history) < 10:
            return
        
        # Create features (time-based)
        X = []
        for i in range(5, len(self.cpu_history)):
            # Use last 5 values as features
            features = [
                self.cpu_history[i-5], self.cpu_history[i-4], self.cpu_history[i-3],
                self.cpu_history[i-2], self.cpu_history[i-1]
            ]
            X.append(features)
        
        # Targets (current values)
        y_cpu = list(self.cpu_history)[5:]
        y_gpu = list(self.gpu_history)[5:]
        y_memory = list(self.memory_history)[5:]
        
        if len(X) >= 5:
            self.cpu_predictor.fit(np.array(X), np.array(y_cpu))
            self.gpu_predictor.fit(np.array(X), np.array(y_gpu))
            self.memory_predictor.fit(np.array(X), np.array(y_memory))
    
    def predict_workload(self, seconds_ahead=30):
        """Predict workload with uncertainty"""
        if len(self.cpu_history) < 10:
            return None
        
        # Use last 5 values as features for prediction
        last_features = [
            self.cpu_history[-5], self.cpu_history[-4], self.cpu_history[-3],
            self.cpu_history[-2], self.cpu_history[-1]
        ]
        
        X_pred = np.array([last_features])
        
        # Get predictions with uncertainty
        cpu_pred, cpu_lower, cpu_upper = self.cpu_predictor.predict_with_uncertainty(X_pred)
        gpu_pred, gpu_lower, gpu_upper = self.gpu_predictor.predict_with_uncertainty(X_pred)
        memory_pred, memory_lower, memory_upper = self.memory_predictor.predict_with_uncertainty(X_pred)
        
        return {
            'cpu': {'prediction': cpu_pred, 'lower': cpu_lower, 'upper': cpu_upper},
            'gpu': {'prediction': gpu_pred, 'lower': gpu_lower, 'upper': gpu_upper},
            'memory': {'prediction': memory_pred, 'lower': memory_lower, 'upper': memory_upper}
        }
    
    def get_offload_decision(self):
        """Make offload decision based on current and predicted load"""
        if not self.current_metrics:
            return "UNKNOWN", "No data"
        
        # Current load analysis
        current_cpu = self.current_metrics['cpu_percent']
        current_gpu = self.current_metrics['gpu_percent']
        current_memory = self.current_metrics['memory_percent']
        current_temp = self.current_metrics['temperature_c']
        
        # Get predictions
        predictions = self.predict_workload()
        
        # Decision logic
        current_max_load = max(current_cpu, current_gpu, current_memory)
        
        # Temperature factor
        temp_factor = 1.0
        if current_temp > 80:
            temp_factor = 1.3
        elif current_temp > 70:
            temp_factor = 1.1
        
        adjusted_load = current_max_load * temp_factor
        
        # Prediction factor
        prediction_factor = 1.0
        if predictions:
            predicted_max = max(
                predictions['cpu']['upper'] if predictions['cpu']['upper'] else 0,
                predictions['gpu']['upper'] if predictions['gpu']['upper'] else 0,
                predictions['memory']['upper'] if predictions['memory']['upper'] else 0
            )
            if predicted_max > 80:
                prediction_factor = 1.2
        
        total_load_score = adjusted_load * prediction_factor
        
        # Make decision
        if total_load_score > 85:
            return "OFFLOAD", f"High load ({total_load_score:.1f}%) - Move to server"
        elif total_load_score > 70:
            return "PREPARE", f"Moderate load ({total_load_score:.1f}%) - Prepare offload"
        else:
            return "EDGE", f"Low load ({total_load_score:.1f}%) - Stay on edge"

class VLMWorkloadMonitor:
    """Main application combining VLM with workload monitoring"""
    
    def __init__(self, vlm_model="llava:7b"):
        self.vlm_model = vlm_model
        self.ollama_url = "http://localhost:11434"
        self.hardware_monitor = HardwareMonitor()
        self.description_queue = queue.Queue()
        self.latest_description = "Waiting for analysis..."
        self.analyzing = False
        
        # Initialize RealSense
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        
        # Start monitoring threads
        self.monitoring_thread = threading.Thread(target=self._monitoring_worker)
        self.monitoring_thread.daemon = True
        
        self.vlm_thread = threading.Thread(target=self._vlm_worker)
        self.vlm_thread.daemon = True
        
    def _encode_image(self, cv_image):
        """Encode image for VLM"""
        _, buffer = cv2.imencode('.jpg', cv_image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buffer).decode('utf-8')
    
    def _analyze_scene(self, image):
        """Analyze scene with VLM"""
        try:
            base64_image = self._encode_image(image)
            
            prompt = """Analyze this scene and provide a detailed description including:
- Objects present and their colors
- People and their activities
- Overall scene context
- Any notable features or changes
Be concise but informative."""
            
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.vlm_model,
                    "prompt": prompt,
                    "images": [base64_image],
                    "stream": False
                },
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json().get('response', 'No description')
            else:
                return f"Error: {response.status_code}"
                
        except Exception as e:
            return f"Analysis error: {str(e)}"
    
    def _monitoring_worker(self):
        """Background hardware monitoring"""
        while self.hardware_monitor.monitoring:
            self.hardware_monitor.collect_metrics()
            
            # Update predictors every 10 seconds
            if len(self.hardware_monitor.cpu_history) % 10 == 0:
                self.hardware_monitor.update_predictors()
            
            time.sleep(1)
    
    def _vlm_worker(self):
        """Background VLM processing"""
        while True:
            try:
                image = self.description_queue.get(timeout=1)
                if image is not None:
                    self.analyzing = True
                    self.latest_description = "Analyzing scene..."
                    description = self._analyze_scene(image)
                    self.latest_description = description
                    self.analyzing = False
            except queue.Empty:
                continue
    
    def run(self):
        """Main application loop"""
        print("Intel Workload Intelligence Monitor")
        print("Real-time VLM with Predictive Offloading")
        print("=" * 50)
        print("Controls:")
        print("SPACE - Analyze scene")
        print("C - Toggle continuous analysis")
        print("Q - Quit")
        print("=" * 50)
        
        # Start pipeline
        self.pipeline.start(self.config)
        
        # Start monitoring
        self.hardware_monitor.monitoring = True
        self.monitoring_thread.start()
        self.vlm_thread.start()
        
        continuous_mode = False
        last_analysis = 0
        
        try:
            while True:
                # Get frames
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                
                if not color_frame:
                    continue
                
                color_image = np.asanyarray(color_frame.get_data())
                
                # Create display layout
                display_height = 720
                display_width = 1280
                display = np.zeros((display_height, display_width, 3), dtype=np.uint8)
                
                # Resize camera feed
                cam_height = 360
                cam_width = 640
                camera_resized = cv2.resize(color_image, (cam_width, cam_height))
                
                # Place camera feed (bottom left)
                display[display_height-cam_height:, :cam_width] = camera_resized
                
                # Hardware monitoring panel (top)
                self._draw_hardware_panel(display)
                
                # VLM description panel (bottom right)
                self._draw_vlm_panel(display)
                
                # Offload decision panel (top right)
                self._draw_offload_panel(display)
                
                # Auto-analyze in continuous mode
                current_time = time.time()
                if continuous_mode and not self.analyzing:
                    if current_time - last_analysis > 5:  # Every 5 seconds
                        if self.description_queue.empty():
                            self.description_queue.put(color_image.copy())
                            last_analysis = current_time
                
                # Show display
                cv2.namedWindow('Intel Workload Intelligence Monitor', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('Intel Workload Intelligence Monitor', display_width, display_height)
                cv2.imshow('Intel Workload Intelligence Monitor', display)
                
                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    if not self.analyzing and self.description_queue.empty():
                        self.description_queue.put(color_image.copy())
                elif key == ord('c'):
                    continuous_mode = not continuous_mode
                    print(f"Continuous mode: {'ON' if continuous_mode else 'OFF'}")
                
        finally:
            self.hardware_monitor.monitoring = False
            self.pipeline.stop()
            cv2.destroyAllWindows()
    
    def _draw_hardware_panel(self, display):
        """Draw hardware monitoring panel"""
        if not self.hardware_monitor.current_metrics:
            return
        
        metrics = self.hardware_monitor.current_metrics
        
        # Panel dimensions
        panel_x = 10
        panel_y = 10
        panel_width = 1260
        panel_height = 180
        
        # Draw panel background
        cv2.rectangle(display, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height), 
                     (40, 40, 40), -1)
        cv2.rectangle(display, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height), 
                     (255, 255, 255), 2)
        
        # Title
        cv2.putText(display, "HARDWARE MONITORING", 
                   (panel_x + 10, panel_y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Metrics
        y_offset = 55
        
        # CPU
        cpu_val = metrics['cpu_percent']
        self._draw_metric_bar(display, "CPU", cpu_val, "%", 
                             panel_x + 20, panel_y + y_offset, 200, (0, 255, 0))
        
        # GPU
        gpu_val = metrics['gpu_percent']
        self._draw_metric_bar(display, "GPU", gpu_val, "%", 
                             panel_x + 250, panel_y + y_offset, 200, (0, 255, 255))
        
        # Memory
        mem_val = metrics['memory_percent']
        self._draw_metric_bar(display, "MEM", mem_val, "%", 
                             panel_x + 480, panel_y + y_offset, 200, (255, 0, 255))
        
        # Temperature
        temp_val = metrics['temperature_c']
        self._draw_metric_bar(display, "TEMP", temp_val, "°C", 
                             panel_x + 710, panel_y + y_offset, 200, (0, 128, 255))
        
        # Power
        power_val = metrics['power_draw_w']
        self._draw_metric_bar(display, "PWR", power_val, "W", 
                             panel_x + 940, panel_y + y_offset, 200, (255, 128, 0))
        
        # Timestamp
        timestamp = metrics['timestamp'].strftime("%H:%M:%S")
        cv2.putText(display, f"Updated: {timestamp}", 
                   (panel_x + 10, panel_y + panel_height - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    def _draw_metric_bar(self, display, label, value, unit, x, y, width, color):
        """Draw a metric bar"""
        bar_height = 20
        max_val = 100 if unit in ["%"] else (100 if unit == "°C" else 50)
        
        # Label
        cv2.putText(display, label, (x, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Bar background
        cv2.rectangle(display, (x, y + 10), (x + width, y + 10 + bar_height), 
                     (100, 100, 100), -1)
        
        # Bar fill
        fill_width = int((value / max_val) * width)
        cv2.rectangle(display, (x, y + 10), (x + fill_width, y + 10 + bar_height), 
                     color, -1)
        
        # Value text
        cv2.putText(display, f"{value:.1f}{unit}", (x, y + 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def _draw_vlm_panel(self, display):
        """Draw VLM description panel"""
        panel_x = 650
        panel_y = 360
        panel_width = 620
        panel_height = 350
        
        # Draw panel background
        cv2.rectangle(display, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height), 
                     (20, 20, 20), -1)
        cv2.rectangle(display, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height), 
                     (255, 255, 255), 2)
        
        # Title
        status_color = (0, 255, 255) if self.analyzing else (0, 255, 0)
        cv2.putText(display, "VLM SCENE ANALYSIS", 
                   (panel_x + 10, panel_y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        # Description
        if self.latest_description:
            words = self.latest_description.split()
            lines = []
            current_line = []
            
            for word in words:
                current_line.append(word)
                test_line = ' '.join(current_line)
                (w, h), _ = cv2.getTextSize(test_line, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                if w > panel_width - 20:
                    if len(current_line) > 1:
                        current_line.pop()
                        lines.append(' '.join(current_line))
                        current_line = [word]
                    else:
                        lines.append(test_line)
                        current_line = []
            
            if current_line:
                lines.append(' '.join(current_line))
            
            # Draw lines
            for i, line in enumerate(lines[:15]):
                cv2.putText(display, line, (panel_x + 10, panel_y + 50 + i * 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def _draw_offload_panel(self, display):
        """Draw offload decision panel"""
        panel_x = 650
        panel_y = 200
        panel_width = 620
        panel_height = 150
        
        # Draw panel background
        cv2.rectangle(display, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height), 
                     (30, 30, 30), -1)
        cv2.rectangle(display, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height), 
                     (255, 255, 255), 2)
        
        # Title
        cv2.putText(display, "PREDICTIVE OFFLOAD DECISION", 
                   (panel_x + 10, panel_y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Get offload decision
        decision, reason = self.hardware_monitor.get_offload_decision()
        
        # Decision color
        if decision == "OFFLOAD":
            decision_color = (0, 0, 255)  # Red
            flag = "🔴"
        elif decision == "PREPARE":
            decision_color = (0, 255, 255)  # Yellow
            flag = "🟡"
        else:
            decision_color = (0, 255, 0)  # Green
            flag = "🟢"
        
        # Decision text
        cv2.putText(display, f"Decision: {decision}", 
                   (panel_x + 10, panel_y + 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, decision_color, 2)
        
        cv2.putText(display, reason, 
                   (panel_x + 10, panel_y + 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Predictions
        predictions = self.hardware_monitor.predict_workload()
        if predictions:
            pred_text = f"Predicted CPU: {predictions['cpu']['prediction']:.1f}% "
            pred_text += f"({predictions['cpu']['lower']:.1f}-{predictions['cpu']['upper']:.1f}%)"
            cv2.putText(display, pred_text, 
                       (panel_x + 10, panel_y + 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

def main():
    # Check if Ollama is running
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code != 200:
            print("❌ Ollama is not running! Please start it first.")
            return
        
        models = response.json().get('models', [])
        vision_models = [m['name'] for m in models if 'llava' in m['name'].lower()]
        
        if not vision_models:
            print("❌ No vision models found! Please install LLaVA first.")
            return
        
        print(f"✅ Using model: {vision_models[0]}")
        
    except Exception as e:
        print(f"❌ Error connecting to Ollama: {e}")
        return
    
    # Run the monitor
    monitor = VLMWorkloadMonitor(vision_models[0])
    monitor.run()

if __name__ == "__main__":
    main()