# Vehicle Counter Flask App

A web-based vehicle counting application using YOLOv8 object detection and Flask.

## Features

- Upload video files through web interface
- Real-time vehicle detection and counting
- Supports detection of: persons, cars, buses, motorcycles
- Live video stream with detection overlays
- Real-time statistics display
- Object tracking with unique IDs

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. The application will automatically download the YOLOv8x model on first run.

## Usage

1. Run the Flask application:
```bash
python app.py
```

2. Open your browser and go to `http://localhost:5000`

3. Upload a video file using the web interface

4. Watch the processed video stream with vehicle detection and counting

5. View real-time statistics showing counts for each vehicle type

## API Endpoints

- `GET /` - Main web interface
- `POST /upload` - Upload video for processing
- `GET /stats` - Get current detection statistics
- `GET /reset` - Reset all counters

## Technical Details

- Uses YOLOv8x model for object detection
- Implements object tracking to avoid duplicate counting
- Processes video frames in real-time
- Streams processed video back to browser
- Stores uploaded video in memory (no disk storage)

## Supported Video Formats

- MP4, AVI, MOV, WMV, and other common video formats supported by OpenCV

## Requirements

- Python 3.8+
- Sufficient RAM for video processing
- GPU recommended for faster processing (optional)
