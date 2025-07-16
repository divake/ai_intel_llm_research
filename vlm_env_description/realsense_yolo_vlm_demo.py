#!/usr/bin/env python3
"""
RealSense D455 + YOLO + VLM Demo
Detect objects with YOLO, then describe them with VLM
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import time
import base64
import requests
import json
import threading
import queue
from ultralytics import YOLO
import torch

class RealSenseYOLOVLM:
    def __init__(self, yolo_model="yolo11n.pt", vlm_model="llava:7b", ollama_url="http://localhost:11434"):
        self.vlm_model = vlm_model
        self.ollama_url = ollama_url
        self.description_queue = queue.Queue()
        self.descriptions = {}
        self.analyzing_ids = set()
        
        # Load YOLO model
        print(f"Loading YOLO model: {yolo_model}")
        self.yolo = YOLO(yolo_model)
        
        # Configure RealSense
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # Enable streams
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        
        # Start pipeline
        self.profile = self.pipeline.start(self.config)
        
        # Get depth scale
        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        
        # Start description thread
        self.desc_thread = threading.Thread(target=self._description_worker)
        self.desc_thread.daemon = True
        self.desc_thread.start()
        
        # Track objects
        self.object_tracker = {}
        self.next_id = 0
    
    def _encode_image(self, cv_image):
        """Encode OpenCV image to base64"""
        _, buffer = cv2.imencode('.jpg', cv_image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buffer).decode('utf-8')
    
    def _get_object_prompt(self, class_name):
        """Get appropriate prompt based on object class"""
        if class_name == "person":
            return """Describe this person in detail:
- Gender appearance (if visible)
- Clothing colors and style
- Hair color/style (if visible)
- Any distinctive features
- What they appear to be doing
Be respectful and factual."""
        
        elif class_name in ["car", "truck", "bus", "motorcycle", "bicycle"]:
            return """Describe this vehicle:
- Type and color
- Any distinctive features
- Condition/state
- Any visible details"""
        
        else:
            return f"""Describe this {class_name}:
- Color and appearance
- Size (relative to surroundings)
- Condition/state
- Any distinctive features"""
    
    def _analyze_object(self, image, class_name, obj_id):
        """Send cropped object to VLM for analysis"""
        try:
            # Encode image
            base64_image = self._encode_image(image)
            
            # Get appropriate prompt
            prompt = self._get_object_prompt(class_name)
            
            # Send to Ollama
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.vlm_model,
                    "prompt": prompt,
                    "images": [base64_image],
                    "stream": False
                },
                timeout=20
            )
            
            if response.status_code == 200:
                desc = response.json().get('response', 'No description')
                # Clean up description
                desc = ' '.join(desc.split())  # Remove extra whitespace
                return desc
            else:
                return f"Error: {response.status_code}"
                
        except Exception as e:
            return f"Analysis error: {str(e)}"
    
    def _description_worker(self):
        """Worker thread for object analysis"""
        while True:
            try:
                item = self.description_queue.get(timeout=1)
                if item is not None:
                    image, class_name, obj_id = item
                    self.analyzing_ids.add(obj_id)
                    description = self._analyze_object(image, class_name, obj_id)
                    self.descriptions[obj_id] = {
                        'class': class_name,
                        'description': description,
                        'timestamp': time.time()
                    }
                    self.analyzing_ids.discard(obj_id)
            except queue.Empty:
                continue
    
    def _get_depth_at_point(self, depth_frame, x, y):
        """Get depth value at specific pixel"""
        return depth_frame.get_distance(int(x), int(y))
    
    def run(self):
        """Main loop"""
        print("RealSense + YOLO + VLM Demo")
        print("Press 'A' to toggle auto-analysis mode")
        print("Press 'C' to clear descriptions")
        print("Press 'S' to save snapshot")
        print("Press 'Q' to quit")
        print("-" * 50)
        
        auto_analyze = True
        frame_count = 0
        
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
                
                # Run YOLO detection
                results = self.yolo(color_image, verbose=False)
                
                # Create display image
                display = color_image.copy()
                
                # Process detections
                if len(results) > 0 and results[0].boxes is not None:
                    boxes = results[0].boxes
                    
                    for i, box in enumerate(boxes):
                        # Get box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        conf = box.conf[0].cpu().numpy()
                        cls = int(box.cls[0].cpu().numpy())
                        class_name = self.yolo.names[cls]
                        
                        # Get center point for depth
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                        depth = self._get_depth_at_point(depth_frame, cx, cy)
                        
                        # Create unique ID for tracking
                        obj_key = f"{class_name}_{i}"
                        
                        # Draw bounding box
                        color = (0, 255, 0) if obj_key not in self.analyzing_ids else (0, 255, 255)
                        cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                        
                        # Draw label with depth
                        label = f"{class_name} ({conf:.2f}) - {depth:.2f}m"
                        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        cv2.rectangle(display, (x1, y1 - label_size[1] - 4), 
                                    (x1 + label_size[0], y1), color, -1)
                        cv2.putText(display, label, (x1, y1 - 2), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                        
                        # Check if we should analyze this object
                        if auto_analyze and frame_count % 150 == 0:  # Every 5 seconds at 30fps
                            if (obj_key not in self.descriptions or 
                                time.time() - self.descriptions[obj_key]['timestamp'] > 10):
                                if obj_key not in self.analyzing_ids and self.description_queue.qsize() < 3:
                                    # Crop object with padding
                                    pad = 20
                                    y1_pad = max(0, y1 - pad)
                                    y2_pad = min(color_image.shape[0], y2 + pad)
                                    x1_pad = max(0, x1 - pad)
                                    x2_pad = min(color_image.shape[1], x2 + pad)
                                    
                                    cropped = color_image[y1_pad:y2_pad, x1_pad:x2_pad]
                                    if cropped.size > 0:
                                        self.description_queue.put((cropped.copy(), class_name, obj_key))
                        
                        # Display description if available
                        if obj_key in self.descriptions:
                            desc_info = self.descriptions[obj_key]
                            desc_lines = desc_info['description'].split('. ')
                            
                            # Draw description box below object
                            desc_y = y2 + 5
                            max_width = 0
                            line_height = 15
                            
                            # Calculate box size
                            for line in desc_lines[:3]:  # Show first 3 sentences
                                if line:
                                    (w, h), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                                    max_width = max(max_width, w)
                            
                            if max_width > 0 and desc_y + len(desc_lines) * line_height < display.shape[0]:
                                # Draw background
                                cv2.rectangle(display, 
                                            (x1, desc_y), 
                                            (x1 + max_width + 10, desc_y + len(desc_lines[:3]) * line_height + 5),
                                            (0, 0, 0), -1)
                                cv2.rectangle(display, 
                                            (x1, desc_y), 
                                            (x1 + max_width + 10, desc_y + len(desc_lines[:3]) * line_height + 5),
                                            (255, 255, 255), 1)
                                
                                # Draw text
                                for j, line in enumerate(desc_lines[:3]):
                                    if line:
                                        cv2.putText(display, line,
                                                  (x1 + 5, desc_y + 12 + j * line_height),
                                                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                # Add status text
                status = f"Auto-Analyze: {'ON' if auto_analyze else 'OFF'} | Queue: {self.description_queue.qsize()} | Descriptions: {len(self.descriptions)}"
                cv2.putText(display, status, (10, 25), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                # Apply colormap to depth
                depth_colormap = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_image, alpha=0.03), 
                    cv2.COLORMAP_JET
                )
                
                # Stack images
                images = np.hstack((display, depth_colormap))
                
                # Show images
                cv2.namedWindow('RealSense YOLO+VLM Demo', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('RealSense YOLO+VLM Demo', images)
                
                # Handle key press
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('a'):
                    auto_analyze = not auto_analyze
                    print(f"Auto-analyze: {'ON' if auto_analyze else 'OFF'}")
                elif key == ord('c'):
                    self.descriptions.clear()
                    print("Cleared all descriptions")
                elif key == ord('s'):
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    cv2.imwrite(f"yolo_vlm_snapshot_{timestamp}.jpg", display)
                    # Save descriptions
                    with open(f"yolo_vlm_descriptions_{timestamp}.json", 'w') as f:
                        json.dump(self.descriptions, f, indent=2)
                    print(f"Saved snapshot: {timestamp}")
                
                frame_count += 1
                
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
        model = vision_models[0]
        print(f"Using VLM model: {model}")
        
    except Exception as e:
        print(f"Error checking Ollama: {e}")
        return
    
    # Check for YOLO
    try:
        import ultralytics
        print("YOLO is available")
    except ImportError:
        print("Installing YOLO...")
        import subprocess
        subprocess.check_call(["pip", "install", "ultralytics"])
    
    # Run demo
    demo = RealSenseYOLOVLM(vlm_model=model)
    demo.run()

if __name__ == "__main__":
    main()