#!/usr/bin/env python3
"""
RealSense D455 + VLM Demo
Real-time scene description using Vision Language Models
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import time
import base64
import requests
import json
from io import BytesIO
import threading
import queue

class RealSenseVLM:
    def __init__(self, model="llava:7b", ollama_url="http://localhost:11434"):
        self.model = model
        self.ollama_url = ollama_url
        self.description_queue = queue.Queue()
        self.latest_description = "Waiting for analysis..."
        self.analyzing = False
        
        # Configure RealSense
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # Enable streams
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        
        # Start pipeline
        self.pipeline.start(self.config)
        
        # Start description thread
        self.desc_thread = threading.Thread(target=self._description_worker)
        self.desc_thread.daemon = True
        self.desc_thread.start()
    
    def _encode_image(self, cv_image):
        """Encode OpenCV image to base64"""
        _, buffer = cv2.imencode('.jpg', cv_image)
        return base64.b64encode(buffer).decode('utf-8')
    
    def _analyze_image(self, image):
        """Send image to VLM for analysis"""
        try:
            # Encode image
            base64_image = self._encode_image(image)
            
            # Create prompt
            prompt = """Describe what you see in this image. Include:
- Objects and their colors
- People (if any) and their appearance
- Activities happening
- Overall scene description
Be concise but detailed."""
            
            # Send to Ollama
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
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
    
    def _description_worker(self):
        """Worker thread for image analysis"""
        while True:
            try:
                image = self.description_queue.get(timeout=1)
                if image is not None:
                    self.analyzing = True
                    self.latest_description = "Analyzing..."
                    description = self._analyze_image(image)
                    self.latest_description = description
                    self.analyzing = False
            except queue.Empty:
                continue
    
    def run(self):
        """Main loop"""
        print("RealSense + VLM Demo")
        print("Press 'SPACE' to analyze current view")
        print("Press 'C' for continuous mode (analyze every 5 seconds)")
        print("Press 'Q' to quit")
        print("-" * 50)
        
        continuous_mode = False
        last_analysis_time = 0
        analysis_interval = 5  # seconds
        
        try:
            while True:
                # Get frames
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                
                if not color_frame or not depth_frame:
                    continue
                
                # Convert to numpy arrays
                color_image = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())
                
                # Apply colormap to depth
                depth_colormap = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_image, alpha=0.03), 
                    cv2.COLORMAP_JET
                )
                
                # Create display image
                display = color_image.copy()
                
                # Add status text
                status_color = (0, 255, 0) if not self.analyzing else (0, 255, 255)
                cv2.putText(display, 
                           f"Mode: {'Continuous' if continuous_mode else 'Manual'} | Model: {self.model}",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                
                if self.analyzing:
                    cv2.putText(display, "Analyzing...", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Display description
                if self.latest_description and self.latest_description != "Waiting for analysis...":
                    # Word wrap the description
                    words = self.latest_description.split()
                    lines = []
                    current_line = []
                    
                    for word in words:
                        current_line.append(word)
                        test_line = ' '.join(current_line)
                        (w, h), _ = cv2.getTextSize(test_line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        if w > 600:  # Max width
                            if len(current_line) > 1:
                                current_line.pop()
                                lines.append(' '.join(current_line))
                                current_line = [word]
                            else:
                                lines.append(test_line)
                                current_line = []
                    
                    if current_line:
                        lines.append(' '.join(current_line))
                    
                    # Draw background for text
                    y_offset = display.shape[0] - (len(lines) * 25 + 20)
                    cv2.rectangle(display, (5, y_offset), (635, display.shape[0] - 5), 
                                 (0, 0, 0), -1)
                    cv2.rectangle(display, (5, y_offset), (635, display.shape[0] - 5), 
                                 (255, 255, 255), 2)
                    
                    # Draw text lines
                    for i, line in enumerate(lines[:8]):  # Max 8 lines
                        cv2.putText(display, line,
                                   (10, y_offset + 20 + i * 25), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Auto-analyze in continuous mode
                current_time = time.time()
                if continuous_mode and not self.analyzing:
                    if current_time - last_analysis_time > analysis_interval:
                        if self.description_queue.empty():
                            self.description_queue.put(color_image.copy())
                            last_analysis_time = current_time
                
                # Stack images
                images = np.hstack((display, depth_colormap))
                
                # Show images
                cv2.namedWindow('RealSense VLM Demo', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('RealSense VLM Demo', images)
                
                # Handle key press
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord(' '):  # Space - analyze current frame
                    if not self.analyzing and self.description_queue.empty():
                        self.description_queue.put(color_image.copy())
                        print("Analyzing current frame...")
                elif key == ord('c'):  # Toggle continuous mode
                    continuous_mode = not continuous_mode
                    print(f"Continuous mode: {'ON' if continuous_mode else 'OFF'}")
                elif key == ord('s'):  # Save snapshot
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    cv2.imwrite(f"vlm_snapshot_{timestamp}.jpg", color_image)
                    with open(f"vlm_description_{timestamp}.txt", 'w') as f:
                        f.write(self.latest_description)
                    print(f"Saved snapshot and description: {timestamp}")
                
        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()

def main():
    # Check if Ollama is running
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code != 200:
            print("Error: Ollama is not running!")
            print("Start it with: ./start-ai-stack.sh")
            return
        
        # Check for vision models
        models = response.json().get('models', [])
        vision_models = [m['name'] for m in models if any(vm in m['name'].lower() 
                        for vm in ['llava', 'bakllava', 'cogvlm', 'qwen-vl'])]
        
        if not vision_models:
            print("No vision models found! Install one with:")
            print("  cd frameworks/ollama-ipex-llm-*/")
            print("  ./ollama pull llava:7b")
            return
        
        print(f"Found vision models: {vision_models}")
        
        # Use first available vision model
        model = vision_models[0]
        print(f"Using model: {model}")
        
    except Exception as e:
        print(f"Error checking Ollama: {e}")
        return
    
    # Run demo
    demo = RealSenseVLM(model=model)
    demo.run()

if __name__ == "__main__":
    main()