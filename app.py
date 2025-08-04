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

from threading import Lock

app = Flask(__name__)
video_data_buffer = None  # Store uploaded video in memory
video_data_lock = Lock()

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
        self.counted_ids = set()  # Track counted object IDs
        self.line_position = None  # Will be set based on frame size
        self.line_orientation = 'horizontal'  # or 'vertical'
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
        self.counted_ids = set()
        self.line_position = None

    def process_frame(self, frame):
        # Set line position if not set
        if self.line_position is None:
            h, w = frame.shape[:2]
            if self.line_orientation == 'horizontal':
                self.line_position = int(h * 2 / 3)  # Set line at 2/3 down the frame
            else:
                self.line_position = w // 2

        # Draw the counting line
        if self.line_orientation == 'horizontal':
            cv2.line(frame, (0, self.line_position), (frame.shape[1], self.line_position), (0, 0, 255), 2)
        else:
            cv2.line(frame, (self.line_position, 0), (self.line_position, frame.shape[0]), (0, 0, 255), 2)

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
            positions = obj_data['positions']
            last_pos = positions[-1]

            # Only count the first time an object crosses the line, in any direction
            if obj_id not in self.counted_ids and len(positions) >= 2:
                prev_pos = positions[-2]
                crossed = False
                if self.line_orientation == 'horizontal':
                    # Check if object crossed the horizontal line between previous and current position (any direction)
                    if (prev_pos[1] < self.line_position and last_pos[1] >= self.line_position) or \
                       (prev_pos[1] > self.line_position and last_pos[1] <= self.line_position):
                        crossed = True
                else:
                    # Check if object crossed the vertical line (any direction)
                    if (prev_pos[0] < self.line_position and last_pos[0] >= self.line_position) or \
                       (prev_pos[0] > self.line_position and last_pos[0] <= self.line_position):
                        crossed = True
                if crossed:
                    self.counted_ids.add(obj_id)
                    self.counter[class_name] += 1
                    self.counter['total'] += 1

            # Enhanced visualization with both ID and class name
            x, y = last_pos
            cv2.circle(frame, last_pos, 5, (0, 255, 0), -1)
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


# Upload endpoint: stores video in memory
@app.route('/upload', methods=['POST'])
def upload_video():
    global video_data_buffer
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({'error': 'No video file selected'}), 400
    video_data = video_file.read()
    with video_data_lock:
        video_data_buffer = video_data
    video_processor.reset_counter()
    return jsonify({'success': True})


# Video feed endpoint: streams processed video
@app.route('/video_feed')
def video_feed():
    return Response(generate_processed_video(), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_processed_video():
    global video_data_buffer
    with video_data_lock:
        if not video_data_buffer:
            yield b'--frame\r\nContent-Type: text/plain\r\n\r\nNo video uploaded\r\n\r\n'
            return
        # Write buffer to temp file for OpenCV
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
            temp_file.write(video_data_buffer)
            temp_video_path = temp_file.name

    cap = cv2.VideoCapture(temp_video_path)
    if not cap.isOpened():
        yield b'--frame\r\nContent-Type: text/plain\r\n\r\nError: Could not open video file\r\n\r\n'
        os.unlink(temp_video_path)
        return
    try:
        video_processor.start_time = time.time()
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            processed_frame = video_processor.process_frame(frame)
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            if not ret:
                continue
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    finally:
        video_processor.end_time = time.time()
        cap.release()
        os.unlink(temp_video_path)

@app.route('/stats')
def get_stats():
    return jsonify(video_processor.get_final_stats())

@app.route('/reset')
def reset_counter():
    video_processor.reset_counter()
    return jsonify({'message': 'Counter reset successfully'})

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
