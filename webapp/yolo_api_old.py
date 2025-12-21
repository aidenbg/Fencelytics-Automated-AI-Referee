from flask import Flask, request, jsonify, send_file
import torch
import cv2
import numpy as np
import requests
from io import BytesIO
import tempfile
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load your custom YOLOv5 model
# Replace 'path/to/your/model.pt' with your actual model path
model = torch.hub.load('ultralytics/yolov5', 'custom', path='/Users/aidenburagohain/Coding/FencingAI/models/NEWFencingJul18best.pt')

@app.route('/analyze', methods=['POST'])
def analyze_video():
    try:
        data = request.json
        video_url = data.get('video_url')
        
        if not video_url:
            return jsonify({'error': 'video_url is required'}), 400
        
        # Download video from URL
        response = requests.get(video_url)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(response.content)
            video_path = tmp_file.name
        
        # Create output video path
        output_path = tempfile.mktemp(suffix='_output.mp4')
        
        # Process video with YOLOv5
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        detections = []
        frame_count = 0
        
        # Process EVERY frame
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run YOLOv5 inference
            results = model(frame)
            
            # Draw detections on frame
            annotated_frame = results.render()[0]
            out.write(annotated_frame)
            
            # Extract detections
            for *box, conf, cls in results.xyxy[0].cpu().numpy():
                if conf > 0.5:  # Confidence threshold
                    x1, y1, x2, y2 = box
                    detections.append({
                        'class': model.names[int(cls)],
                        'confidence': float(conf),
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'frame': frame_count
                    })
            
            frame_count += 1
        
        cap.release()
        out.release()
        os.unlink(video_path)  # Clean up temp file
        
        # Calculate metrics
        object_counts = {}
        for detection in detections:
            obj_class = detection['class']
            object_counts[obj_class] = object_counts.get(obj_class, 0) + 1
        
        total_detections = len(detections)
        avg_confidence = np.mean([d['confidence'] for d in detections]) if detections else 0
        
        # Save output video info for download
        video_id = os.path.basename(output_path)
        
        return jsonify({
            'detections': detections,
            'metrics': {
                'total_detections': total_detections,
                'object_counts': object_counts,
                'average_confidence': float(avg_confidence),
                'frames_analyzed': frame_count,
                'total_frames': total_frames
            },
            'output_video_id': video_id
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download/<video_id>', methods=['GET'])
def download_video(video_id):
    video_path = os.path.join(tempfile.gettempdir(), video_id)
    if os.path.exists(video_path):
        return send_file(video_path, mimetype='video/mp4')
    else:
        return jsonify({'error': 'Video not found'}), 404

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'model_loaded': True})

if __name__ == '__main__':
    print("Starting YOLOv5 API server...")
    print("Model loaded successfully!")
    app.run(host='0.0.0.0', port=5000, debug=True)
