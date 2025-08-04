from flask import Flask, render_template, request, Response, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import time
import io
from datetime import datetime
import base64
import tempfile
import os

app = Flask(__name__)

# Load the YOLO model
model = YOLO('yolov8x.pt')

target_classes = {
    0: 'person',
    2: 'car',
    5: 'bus',
    3: 'motorcycle',  
}

class ObjectTracker:
    def __init__(self):
        self.objects = {}  # Dictionary to store object history
        self.id_counter = 0
        self.max_distance = 50  # Maximum distance to consider same object

    def get_distance(self, pt1, pt2):
        return np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)

    def update(self, detections):
        current_objects = {}
        
        for cls, bbox, conf in detections:
            center = ((bbox[0] + bbox[2])//2, (bbox[1] + bbox[3])//2)
            
            matched = False
            # Try to match with existing objects
            for obj_id, obj_data in self.objects.items():
                if obj_data['class'] == cls:
                    last_pos = obj_data['positions'][-1]
                    if self.get_distance(center, last_pos) < self.max_distance:
                        current_objects[obj_id] = {
                            'class': cls,
                            'positions': obj_data['positions'] + [center],
                            'confidence': conf,
                            'last_seen': time.time()
                        }
                        matched = True
                        break
            
            # If no match found, create new object
            if not matched:
                self.id_counter += 1
                current_objects[self.id_counter] = {
                    'class': cls,
                    'positions': [center],
                    'confidence': conf,
                    'last_seen': time.time()
                }
        
        self.objects = current_objects
        return current_objects

class VideoProcessor:
    def __init__(self):
        self.counter = {
            'person': 0,
            'car': 0,
            'bus': 0,
            'motorcycle': 0,
            'total': 0
        }
        self.tracker = ObjectTracker()
        self.unique_objects = set()
        self.start_time = None
        self.end_time = None

    def reset_counter(self):
        self.counter = {
            'person': 0,
            'car': 0,
            'bus': 0,
            'motorcycle': 0,
            'total': 0
        }
        self.tracker = ObjectTracker()
        self.unique_objects = set()

    def process_frame(self, frame):
        results = model(frame, conf=0.3)
        current_detections = []
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                
                if cls in target_classes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    current_detections.append((cls, (x1, y1, x2, y2), conf))
        
        # Update tracker
        tracked_objects = self.tracker.update(current_detections)
        
        # Process tracked objects
        for obj_id, obj_data in tracked_objects.items():
            cls = obj_data['class']
            class_name = target_classes[cls]
            if obj_id not in self.unique_objects and len(obj_data['positions']) >= 3:
                self.unique_objects.add(obj_id)
                self.counter[target_classes[cls]] += 1
                self.counter['total'] += 1
            
            # Enhanced visualization with both ID and class name
            last_pos = obj_data['positions'][-1]
            x, y = last_pos
            
            # Draw circle at centroid
            cv2.circle(frame, last_pos, 5, (0, 255, 0), -1)
            
            # Draw ID and class name with better formatting
            label = f"ID:{obj_id} ({class_name})"
            cv2.putText(frame, label, (x-10, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Display counts
        y_pos = 30
        for class_name, count in self.counter.items():
            text = f'{class_name}: {count}'
            cv2.putText(frame, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            y_pos += 40
        
        return frame

    def get_final_stats(self):
        processing_time = (self.end_time - self.start_time) if self.start_time and self.end_time else 0
        return {
            'processing_time': processing_time,
            'counts': self.counter.copy()
        }

# Global video processor instance
video_processor = VideoProcessor()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({'error': 'No video file selected'}), 400
    
    # Read video data into memory
    video_data = video_file.read()
    
    # Create a temporary file in memory
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
        temp_file.write(video_data)
        temp_video_path = temp_file.name
    
    try:
        # Reset counter for new video
        video_processor.reset_counter()
        video_processor.start_time = time.time()
        
        # Process video and return streaming response
        return Response(
            generate_processed_video(temp_video_path),
            mimetype='multipart/x-mixed-replace; boundary=frame'
        )
    except Exception as e:
        return jsonify({'error': f'Error processing video: {str(e)}'}), 500
    finally:
        # Clean up temporary file
        if os.path.exists(temp_video_path):
            os.unlink(temp_video_path)

def generate_processed_video(video_path):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        yield b'--frame\r\nContent-Type: text/plain\r\n\r\nError: Could not open video file\r\n\r\n'
        return
    
    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            
            # Process frame
            processed_frame = video_processor.process_frame(frame)
            
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            if not ret:
                continue
                
            frame_bytes = buffer.tobytes()
            
            # Yield frame in streaming format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    finally:
        video_processor.end_time = time.time()
        cap.release()

@app.route('/stats')
def get_stats():
    return jsonify(video_processor.get_final_stats())

@app.route('/reset')
def reset_counter():
    video_processor.reset_counter()
    return jsonify({'message': 'Counter reset successfully'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
