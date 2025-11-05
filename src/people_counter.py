#!/usr/bin/env python3
'''
MJPEG Stream People Counter using YOLOv11n fine-tuned model
Monitors an MJPEG stream and counts people in real-time using the fine-tuned YOLOv11n model.
'''

import cv2
import numpy as np
from ultralytics import YOLO
import time
import datetime
import urllib.request

# Constants
MODEL_PATH = 'yolov8n.pt'  # YOLOv8n pre-trained model
STREAM_URL = 'http://172.30.1.27:5000/video'  # MJPEG stream URL
CONFIDENCE_THRESHOLD = 0.25  # Detection confidence threshold (lowered for YOLOv8n)
CLASS_PERSON = 0  # Class index for 'person' in COCO dataset

def create_stream_capture(max_retries=3, retry_delay=5):
    """Creates a video capture object for MJPEG stream with retry logic"""
    for attempt in range(max_retries):
        try:
            print(f"Attempting to connect to stream (attempt {attempt + 1}/{max_retries})...")
            stream = cv2.VideoCapture(STREAM_URL)
            
            # Try to read a test frame to verify connection
            ret, _ = stream.read()
            if ret:
                print("Successfully connected to stream")
                return stream
            else:
                print("Failed to read from stream")
                stream.release()
                
        except Exception as e:
            print(f"Connection attempt {attempt + 1} failed: {str(e)}")
        
        if attempt < max_retries - 1:
            print(f"Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
    
    raise RuntimeError(f"Failed to connect to stream at {STREAM_URL} after {max_retries} attempts")

def load_model():
    """Loads the YOLO model"""
    try:
        model = YOLO(MODEL_PATH)
        print(f"Model loaded successfully from {MODEL_PATH}")
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {str(e)}")

def draw_detections(frame, results):
    """Draw detection boxes and labels on the frame"""
    if len(results) > 0:
        boxes = results[0].boxes
        for box in boxes:
            # Get box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            
            # Get confidence
            conf = float(box.conf[0])
            
            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)  # Yellow box
            
            # Draw label background
            label = f"Person: {conf:.2f}"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - 20), (x1 + label_w, y1), (0, 0, 0), -1)
            
            # Draw label text
            cv2.putText(frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    # Draw total count in top-left corner
    count = len(results[0].boxes) if len(results) > 0 else 0
    cv2.putText(frame, f"People Count: {count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    return frame

def process_stream(model, stream):
    """Process the video stream and count people"""
    frame_count = 0
    start_time = time.time()
    consecutive_failures = 0
    MAX_FAILURES = 5  # Maximum number of consecutive failures before reconnecting
    
    # Create window
    cv2.namedWindow('People Detection', cv2.WINDOW_NORMAL)

    while True:
        try:
            # Read frame from stream
            ret, frame = stream.read()
            if not ret:
                consecutive_failures += 1
                print(f"Failed to receive frame from stream (failure {consecutive_failures}/{MAX_FAILURES})")
                if consecutive_failures >= MAX_FAILURES:
                    print("Too many consecutive failures. Attempting to reconnect...")
                    stream.release()
                    stream = create_stream_capture()
                    consecutive_failures = 0
                continue

            # Run detection
            results = model.predict(
                source=frame,
                conf=CONFIDENCE_THRESHOLD,
                classes=[CLASS_PERSON],  # Only detect people
                verbose=False
            )

            # Draw detections on frame
            if len(results) > 0:
                # Draw boxes and count
                frame = draw_detections(frame, results)
                detections = len(results[0].boxes)
                
                # Get current timestamp
                current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # Print count with timestamp
                print(f"[{current_time}] People detected: {detections}")
            
            # Show frame
            cv2.imshow('People Detection', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):  # Press 'q' to quit
                print("\nStopping...")
                break
            elif key == ord('f'):  # Press 'f' to toggle fullscreen
                if cv2.getWindowProperty('People Detection', cv2.WND_PROP_FULLSCREEN) == cv2.WINDOW_FULLSCREEN:
                    cv2.setWindowProperty('People Detection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                else:
                    cv2.setWindowProperty('People Detection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

            frame_count += 1
            if frame_count % 30 == 0:  # Print FPS every 30 frames
                elapsed_time = time.time() - start_time
                fps = frame_count / elapsed_time
                print(f"FPS: {fps:.2f}")

        except KeyboardInterrupt:
            print("\nStopping...")
            break
        except Exception as e:
            print(f"Error processing frame: {str(e)}")
            continue

def main():
    try:
        # Load YOLO model
        model = load_model()
        
        # Create stream capture
        stream = create_stream_capture()
        
        print("Starting people detection...")
        print("Press Ctrl+C to stop")
        
        # Process stream
        process_stream(model, stream)
        
    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        if 'stream' in locals():
            stream.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()