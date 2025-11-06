#!/usr/bin/env python3
'''
Sensor Fusion People Counter using RGB and Thermal Modalities
Combines RGB camera (YOLOv10m pretrained) and Thermal camera (YOLOv8n) for enhanced people detection.
Based on the approach from "Human Detection Combining RGB and Thermal Modalities" paper.
'''

import cv2
import numpy as np
from ultralytics import YOLO
try:
    from yolov10 import YOLOv10
except ImportError:
    print("YOLOv10 not available, falling back to YOLOv8m for RGB")
    YOLOv10 = None
import time
import datetime
import threading
import queue
from flask import Flask, Response, render_template_string, jsonify
import requests

# Constants
THERMAL_MODEL_PATH = 'yolov8n.pt'  # YOLOv8n fine-tuned for thermal
# RGB_MODEL_PATH = 'yolov10m.pt'  # YOLOv10m for RGB (using pretrained instead)
THERMAL_STREAM_URL = 'http://172.30.1.27:5000/video'  # Thermal MJPEG stream
RGB_STREAM_URL = 'http://172.30.1.27:5000/rgbvideo'  # RGB MJPEG stream
CONFIDENCE_THRESHOLD = 0.25  # Detection confidence threshold
CLASS_PERSON = 0  # Class index for 'person' in COCO dataset

# Flask app
app = Flask(__name__)

# Global variables for frames
current_fused_frame = None
hud_data = {
    'temperature': None,
    'humidity': None,
    'max_temp': 0.0,
    'min_temp': 0.0,
    'rgb_count': 0,
    'thermal_count': 0,
    'fused_count': 0
}

class StreamCapture:
    def __init__(self, url, name, max_retries=3, retry_delay=5):
        self.url = url
        self.name = name
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.stream = None
        self.connect()

    def connect(self):
        for attempt in range(self.max_retries):
            try:
                print(f"Attempting to connect to {self.name} stream (attempt {attempt + 1}/{self.max_retries})...")
                self.stream = cv2.VideoCapture(self.url)
                
                # Try to read a test frame to verify connection
                ret, _ = self.stream.read()
                if ret:
                    print(f"Successfully connected to {self.name} stream")
                    return
                else:
                    print(f"Failed to read from {self.name} stream")
                    self.stream.release()
                    
            except Exception as e:
                print(f"{self.name} connection attempt {attempt + 1} failed: {str(e)}")
            
            if attempt < self.max_retries - 1:
                print(f"Retrying {self.name} in {self.retry_delay} seconds...")
                time.sleep(self.retry_delay)
        
        raise RuntimeError(f"Failed to connect to {self.name} stream at {self.url} after {self.max_retries} attempts")

    def read_frame(self):
        if self.stream is None:
            return False, None
        ret, frame = self.stream.read()
        if not ret:
            print(f"Failed to read frame from {self.name} stream")
        return ret, frame

    def release(self):
        if self.stream:
            self.stream.release()

def load_models():
    """Loads the YOLO models for RGB and Thermal"""
    try:
        thermal_model = YOLO(THERMAL_MODEL_PATH)
        print(f"Thermal model loaded successfully from {THERMAL_MODEL_PATH}")
        
        if YOLOv10 is not None:
            rgb_model = YOLOv10.from_pretrained('jameslahm/yolov10m')
            print("RGB model loaded successfully from pretrained 'jameslahm/yolov10m'")
        else:
            rgb_model = YOLO('yolov8m.pt')  # Fallback to YOLOv8m
            print("RGB model loaded successfully from yolov8m.pt (fallback)")
        
        return thermal_model, rgb_model
    except Exception as e:
        raise RuntimeError(f"Failed to load models: {str(e)}")

def fuse_detections(thermal_results, rgb_results, thermal_frame, rgb_frame):
    """
    Fuse detections from thermal and RGB modalities
    Based on the paper: Combine detections where both modalities agree
    """
    fused_boxes = []
    
    if len(thermal_results) > 0 and len(rgb_results) > 0:
        thermal_boxes = thermal_results[0].boxes
        rgb_boxes = rgb_results[0].boxes
        
        # Simple fusion: Keep detections that appear in both modalities
        # This reduces false positives by requiring confirmation from both sensors
        for t_box in thermal_boxes:
            t_x1, t_y1, t_x2, t_y2 = map(int, t_box.xyxy[0].tolist())
            t_conf = float(t_box.conf[0])
            
            # Check if there's a corresponding detection in RGB
            for r_box in rgb_boxes:
                r_x1, r_y1, r_x2, r_y2 = map(int, r_box.xyxy[0].tolist())
                r_conf = float(r_box.conf[0])
                
                # Calculate IoU (Intersection over Union) to see if boxes overlap
                iou = calculate_iou((t_x1, t_y1, t_x2, t_y2), (r_x1, r_y1, r_x2, r_y2))
                
                if iou > 0.3:  # If boxes overlap significantly
                    # Fuse the detection: use average confidence and thermal box coordinates
                    fused_conf = (t_conf + r_conf) / 2
                    fused_boxes.append({
                        'box': (t_x1, t_y1, t_x2, t_y2),
                        'conf': fused_conf
                    })
                    break  # Only match once
    
    return fused_boxes

def calculate_iou(box1, box2):
    """Calculate Intersection over Union of two bounding boxes"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0

def draw_fused_detections(frame, detections):
    """Draw fused detection boxes and labels on the frame"""
    for detection in detections:
        x1, y1, x2, y2 = detection['box']
        conf = detection['conf']
        
        # Draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
        
        # Draw label background
        label = f"Fused Person: {conf:.2f}"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - 20), (x1 + label_w, y1), (0, 0, 0), -1)
        
        # Draw label text
        cv2.putText(frame, label, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # Draw total count in top-left corner
    count = len(detections)
    cv2.putText(frame, f"Fused People Count: {count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return frame

def process_streams(thermal_model, rgb_model, thermal_stream, rgb_stream):
    """Process both video streams and perform sensor fusion"""
    frame_count = 0
    start_time = time.time()
    
    # Create windows
    cv2.namedWindow('Thermal Detection', cv2.WINDOW_NORMAL)
    cv2.namedWindow('RGB Detection', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Fused Detection', cv2.WINDOW_NORMAL)

    while True:
        try:
            # Read frames from both streams
            thermal_ret, thermal_frame = thermal_stream.read_frame()
            rgb_ret, rgb_frame = rgb_stream.read_frame()
            
            if not thermal_ret or not rgb_ret:
                print("Failed to read frames from one or both streams")
                time.sleep(1)
                continue

            # Run detections
            thermal_results = thermal_model.predict(
                source=thermal_frame,
                conf=CONFIDENCE_THRESHOLD,
                classes=[CLASS_PERSON],
                verbose=False
            )
            
            rgb_results = rgb_model.predict(
                source=rgb_frame,
                conf=CONFIDENCE_THRESHOLD,
                classes=[CLASS_PERSON],
                verbose=False
            )

            # Fuse detections
            fused_detections = fuse_detections(thermal_results, rgb_results, thermal_frame, rgb_frame)

            # Update HUD data with counts
            global hud_data
            hud_data['rgb_count'] = len(rgb_results[0].boxes) if len(rgb_results) > 0 else 0
            hud_data['thermal_count'] = len(thermal_results[0].boxes) if len(thermal_results) > 0 else 0
            hud_data['fused_count'] = len(fused_detections)

            # Draw detections on frames
            thermal_frame = draw_detections(thermal_frame, thermal_results, "Thermal")
            rgb_frame = draw_detections(rgb_frame, rgb_results, "RGB")
            
            # Create fused frame (side by side or overlay)
            fused_frame = create_fused_frame(thermal_frame, rgb_frame, fused_detections)

            # Update global fused frame for web streaming
            global current_fused_frame
            current_fused_frame = fused_frame.copy()

            # Show frames
            cv2.imshow('Thermal Detection', thermal_frame)
            cv2.imshow('RGB Detection', rgb_frame)
            cv2.imshow('Fused Detection', fused_frame)
            
            # Get current timestamp
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Print count with timestamp
            print(f"[{current_time}] Fused People detected: {len(fused_detections)}")
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):  # Press 'q' to quit
                print("\nStopping...")
                break

            frame_count += 1
            if frame_count % 30 == 0:  # Print FPS every 30 frames
                elapsed_time = time.time() - start_time
                fps = frame_count / elapsed_time
                print(f"FPS: {fps:.2f}")

        except KeyboardInterrupt:
            print("\nStopping...")
            break
        except Exception as e:
            print(f"Error processing frames: {str(e)}")
            continue

def draw_detections(frame, results, modality):
    """Draw detection boxes and labels on the frame"""
    if len(results) > 0:
        boxes = results[0].boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)  # Yellow box
            
            label = f"{modality} Person: {conf:.2f}"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - 20), (x1 + label_w, y1), (0, 0, 0), -1)
            
            cv2.putText(frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    count = len(results[0].boxes) if len(results) > 0 else 0
    cv2.putText(frame, f"{modality} Count: {count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    return frame

def create_fused_frame(thermal_frame, rgb_frame, fused_detections):
    """Create a fused frame showing both modalities and fused results"""
    # Resize frames to same height for side-by-side display
    height = min(thermal_frame.shape[0], rgb_frame.shape[0])
    thermal_resized = cv2.resize(thermal_frame, (int(thermal_frame.shape[1] * height / thermal_frame.shape[0]), height))
    rgb_resized = cv2.resize(rgb_frame, (int(rgb_frame.shape[1] * height / rgb_frame.shape[0]), height))
    
    # Concatenate horizontally
    combined = np.concatenate((thermal_resized, rgb_resized), axis=1)
    
    # Draw fused count on combined frame
    count = len(fused_detections)
    cv2.putText(combined, f"Fused People Count: {count}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return combined

def fetch_hud_data():
    """Fetch HUD data from the thermal camera API"""
    global hud_data
    try:
        response = requests.get('http://172.30.1.27:5000/api/hud', timeout=5)
        if response.status_code == 200:
            data = response.json()
            # Update only temperature-related fields, keep counts
            hud_data.update({
                'temperature': data.get('temperature'),
                'humidity': data.get('humidity'),
                'max_temp': data.get('max_temp', 0.0),
                'min_temp': data.get('min_temp', 0.0)
            })
    except Exception as e:
        print(f"Error fetching HUD data: {e}")

def generate_video_feed():
    """Generate video feed for fused frame"""
    while True:
        if current_fused_frame is not None:
            ret, jpeg = cv2.imencode('.jpg', current_fused_frame)
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
        time.sleep(0.1)

@app.route('/fused_video')
def fused_video_feed():
    return Response(generate_video_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/hud')
def get_hud():
    fetch_hud_data()  # Fetch latest data
    return jsonify(hud_data)

@app.route('/')
def index():
    html = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Fused Frame Viewer</title>
        <style>
            body { margin: 0; padding: 0; font-family: Arial, sans-serif; }
            .container { display: flex; flex-direction: column; height: 100vh; }
            .video-section { flex: 1; background: black; display: flex; justify-content: center; align-items: center; }
            .hud-section { flex: 1; background: #f0f0f0; padding: 20px; display: flex; flex-wrap: wrap; justify-content: space-around; align-items: center; }
            .hud-item { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); text-align: center; min-width: 150px; margin: 10px; }
            .hud-label { font-size: 14px; color: #666; margin-bottom: 5px; }
            .hud-value { font-size: 24px; font-weight: bold; color: #333; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="video-section">
                <img src="/fused_video" style="max-width: 100%; max-height: 100%;" />
            </div>
            <div class="hud-section" id="hud-section">
                <!-- HUD data will be loaded here -->
            </div>
        </div>
        <script>
            function updateHUD() {
                fetch('/api/hud')
                    .then(response => response.json())
                    .then(data => {
                        const hudSection = document.getElementById('hud-section');
                        hudSection.innerHTML = `
                            <div class="hud-item">
                                <div class="hud-label">Temperature</div>
                                <div class="hud-value">${data.temperature !== null ? data.temperature + '°C' : 'N/A'}</div>
                            </div>
                            <div class="hud-item">
                                <div class="hud-label">Humidity</div>
                                <div class="hud-value">${data.humidity !== null ? data.humidity + '%' : 'N/A'}</div>
                            </div>
                            <div class="hud-item">
                                <div class="hud-label">Max Temp</div>
                                <div class="hud-value">${data.max_temp}°C</div>
                            </div>
                            <div class="hud-item">
                                <div class="hud-label">Min Temp</div>
                                <div class="hud-value">${data.min_temp}°C</div>
                            </div>
                            <div class="hud-item">
                                <div class="hud-label">RGB Count</div>
                                <div class="hud-value">${data.rgb_count}</div>
                            </div>
                            <div class="hud-item">
                                <div class="hud-label">Thermal Count</div>
                                <div class="hud-value">${data.thermal_count}</div>
                            </div>
                            <div class="hud-item">
                                <div class="hud-label">Fused Count</div>
                                <div class="hud-value">${data.fused_count}</div>
                            </div>
                        `;
                    })
                    .catch(error => console.error('Error fetching HUD data:', error));
            }
            setInterval(updateHUD, 1000); // Update every second
            updateHUD(); // Initial load
        </script>
    </body>
    </html>
    '''
    return render_template_string(html)

def main():
    try:
        # Load models
        thermal_model, rgb_model = load_models()
        
        # Create stream captures
        thermal_stream = StreamCapture(THERMAL_STREAM_URL, "Thermal")
        rgb_stream = StreamCapture(RGB_STREAM_URL, "RGB")
        
        # Start Flask app in a separate thread
        flask_thread = threading.Thread(target=lambda: app.run(host='0.0.0.0', port=5001, debug=False, use_reloader=False))
        flask_thread.daemon = True
        flask_thread.start()
        
        print("Starting sensor fusion people detection...")
        print("Web view available at http://localhost:5001")
        print("Press Ctrl+C to stop")
        
        # Process streams
        process_streams(thermal_model, rgb_model, thermal_stream, rgb_stream)
        
    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        if 'thermal_stream' in locals():
            thermal_stream.release()
        if 'rgb_stream' in locals():
            rgb_stream.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()