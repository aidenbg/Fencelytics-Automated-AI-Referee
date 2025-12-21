from flask import Flask, request, jsonify
import torch
import cv2
import os
import tempfile
import requests
import shutil
from flask_cors import CORS
import pandas as pd
import numpy as np
from ultralytics import YOLO
import logging
from datetime import datetime
import time
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file if it exists
# Get credentials from environment
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_SERVICE_KEY')

app = Flask(__name__)
CORS(app)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Load models once at startup
logger.info("========== STARTING YOLO API SERVER ==========")
logger.info("Loading detection model...")
start_time = time.time()
YOLO_REPO_PATH = "/Users/aidenburagohain/Coding/FencingAI/yolov5-v6.0"
detection_model = torch.hub.load(YOLO_REPO_PATH, 'custom', 
                                path="/Users/aidenburagohain/Coding/FencingAI/models/NEWFencingJul18best.pt", 
                                source='local')
logger.info(f"Detection model loaded in {time.time() - start_time:.2f}s")

logger.info("Loading pose model...")
start_time = time.time()
pose_model = YOLO("yolo11m-pose.pt")
logger.info(f"Pose model loaded in {time.time() - start_time:.2f}s")
logger.info("All models loaded successfully!")

# Pose skeleton connections
SKELETON = [
    [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],
    [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
    [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
    [2, 4], [3, 5], [4, 6], [5, 7]
]

@app.route('/process', methods=['POST'])
def process_video():
    request_start = time.time()
    logger.info("========== NEW VIDEO PROCESSING REQUEST ==========")
    
    try:
        data = request.json
        video_url = data['video_url']
        video_id = data['video_id']
        
        logger.info(f"Video ID: {video_id}")
        logger.info(f"Video URL: {video_url[:50]}...")
        
        # Download video
        logger.info("Creating temporary directory...")
        work_dir = tempfile.mkdtemp()
        input_path = os.path.join(work_dir, 'input.mp4')
        logger.info(f"Work directory: {work_dir}")
        
        logger.info("Downloading video...")
        download_start = time.time()
        response = requests.get(video_url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        downloaded = 0
        with open(input_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    progress = (downloaded / total_size) * 100
                    if downloaded % (1024 * 1024) == 0:  # Log every MB
                        logger.info(f"Download progress: {progress:.1f}% ({downloaded/1024/1024:.1f}MB/{total_size/1024/1024:.1f}MB)")
        
        logger.info(f"Download complete in {time.time() - download_start:.2f}s")
        logger.info(f"File size: {os.path.getsize(input_path)/1024/1024:.1f}MB")
        
        # Process video
        logger.info("Starting video processing...")
        process_start = time.time()
        process_all_videos(input_path, work_dir, video_id)
        logger.info(f"Video processing complete in {time.time() - process_start:.2f}s")
        
        # Upload results
        logger.info("Uploading processed videos...")
        upload_start = time.time()
        urls = {
            'detections': upload_video(os.path.join(work_dir, 'detections.mp4'), f"{video_id}_detections.mp4"),
            'pose': upload_video(os.path.join(work_dir, 'pose.mp4'), f"{video_id}_pose.mp4"),
            'all': upload_video(os.path.join(work_dir, 'all.mp4'), f"{video_id}_all.mp4")
        }
        logger.info(f"Upload complete in {time.time() - upload_start:.2f}s")
        
        # Cleanup
        logger.info("Cleaning up temporary files...")
        shutil.rmtree(work_dir)
        
        total_time = time.time() - request_start
        logger.info(f"========== REQUEST COMPLETE in {total_time:.2f}s ==========")
        
        return jsonify({'success': True, 'urls': urls, 'processing_time': total_time})
        
    except Exception as e:
        logger.error(f"ERROR: {type(e).__name__}: {str(e)}")
        logger.exception("Full traceback:")
        return jsonify({'error': str(e)}), 500

def process_all_videos(input_path, output_dir, video_id):
    """Process video with both models in one pass"""
    
    # Open video
    logger.info("Opening video file...")
    cap = cv2.VideoCapture(input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    logger.info(f"Video properties: {width}x{height} @ {fps}fps, {total_frames} frames")
    
    # Create output writers
    logger.info("Creating output video writers...")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_detections = cv2.VideoWriter(os.path.join(output_dir, 'detections.mp4'), fourcc, fps, (width, height))
    out_pose = cv2.VideoWriter(os.path.join(output_dir, 'pose.mp4'), fourcc, fps, (width, height))
    out_all = cv2.VideoWriter(os.path.join(output_dir, 'all.mp4'), fourcc, fps, (width, height))
    
    TARGET_CLASSES = ["Fencer", "Foil", "Torso", "Bellguard"]
    MAX_PER_CLASS = 2
    
    frame_count = 0
    process_start = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Log progress every 30 frames (roughly 1 second at 30fps)
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            elapsed = time.time() - process_start
            fps_processing = frame_count / elapsed
            eta = (total_frames - frame_count) / fps_processing if fps_processing > 0 else 0
            logger.info(f"Processing: {progress:.1f}% ({frame_count}/{total_frames} frames) - {fps_processing:.1f} fps - ETA: {eta:.1f}s")
        
        # Run detection model
        detection_results = detection_model(frame)
        detections_df = detection_results.pandas().xyxy[0]
        
        # Filter detections
        filtered_indices = []
        for class_name in TARGET_CLASSES:
            class_detections = detections_df[detections_df['name'] == class_name]
            if len(class_detections) > 0:
                top_indices = class_detections.nlargest(MAX_PER_CLASS, 'confidence').index.tolist()
                filtered_indices.extend(top_indices)
        
        detection_results.xyxy[0] = detection_results.xyxy[0][filtered_indices] if filtered_indices else torch.empty((0, 6))
        
        # Create detection frame
        frame_detections = detection_results.render()[0]
        
        # Run pose model
        pose_results = pose_model(frame)
        frame_pose = frame.copy()
        
        if pose_results[0].keypoints is not None and pose_results[0].keypoints.data.shape[0] > 0:
            keypoints_data = pose_results[0].keypoints.data.cpu().numpy()
            
            # Get top 2 people by keypoint count
            keypoint_counts = []
            for i in range(keypoints_data.shape[0]):
                visible_count = np.sum(keypoints_data[i, :, 2] > 0.5)
                keypoint_counts.append((i, visible_count))
            
            keypoint_counts.sort(key=lambda x: x[1], reverse=True)
            top_indices = [x[0] for x in keypoint_counts[:2]]
            filtered_keypoints = keypoints_data[top_indices]
            
            # Draw pose on frame
            frame_pose = draw_poses(frame_pose, filtered_keypoints)
        
        # Create combined frame
        frame_all = frame.copy()
        
        # First draw pose
        if pose_results[0].keypoints is not None and 'filtered_keypoints' in locals() and len(filtered_keypoints) > 0:
            frame_all = draw_poses(frame_all, filtered_keypoints)
        
        # Then overlay detections
        frame_all = np.ascontiguousarray(frame_all)  # Make writable
        detection_results.imgs[0] = frame_all  # Update the image in results
        frame_all = detection_results.render()[0]
        
        # Write frames
        out_detections.write(frame_detections)
        out_pose.write(frame_pose)
        out_all.write(frame_all)
    
    # Cleanup
    cap.release()
    out_detections.release()
    out_pose.release()
    out_all.release()
    
    processing_time = time.time() - process_start
    avg_fps = frame_count / processing_time
    logger.info(f"Processed {frame_count} frames in {processing_time:.2f}s (avg {avg_fps:.1f} fps)")
    
    # Log output file sizes
    for filename in ['detections.mp4', 'pose.mp4', 'all.mp4']:
        file_path = os.path.join(output_dir, filename)
        if os.path.exists(file_path):
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            logger.info(f"Output {filename}: {size_mb:.1f}MB")

def draw_poses(frame, keypoints):
    """Draw keypoints and skeleton"""
    skeleton_color = (255, 0, 0)  # Blue
    keypoint_color = (0, 255, 0)  # Green
    
    for person_kpts in keypoints:
        # Draw skeleton
        for connection in SKELETON:
            kpt1_idx, kpt2_idx = connection[0] - 1, connection[1] - 1
            
            if (kpt1_idx < len(person_kpts) and kpt2_idx < len(person_kpts) and
                person_kpts[kpt1_idx, 2] > 0.5 and person_kpts[kpt2_idx, 2] > 0.5):
                
                pt1 = (int(person_kpts[kpt1_idx, 0]), int(person_kpts[kpt1_idx, 1]))
                pt2 = (int(person_kpts[kpt2_idx, 0]), int(person_kpts[kpt2_idx, 1]))
                cv2.line(frame, pt1, pt2, skeleton_color, 2)
        
        # Draw keypoints
        for kpt in person_kpts:
            if kpt[2] > 0.5:
                x, y = int(kpt[0]), int(kpt[1])
                cv2.circle(frame, (x, y), 5, keypoint_color, -1)
                cv2.circle(frame, (x, y), 3, (255, 255, 255), -1)
    
    return frame

def upload_video(file_path, filename):
    """Upload to Supabase with proper folder structure"""

    if not SUPABASE_URL or not SUPABASE_KEY:
        logger.error("Missing Supabase credentials!")
        return f"https://fake-url.com/{filename}"
    
    # Determine folder based on filename
    if '_detections.mp4' in filename:
        folder = 'detections'
    elif '_pose.mp4' in filename:
        folder = 'pose'
    elif '_all.mp4' in filename:
        folder = 'all'
    else:
        folder = 'original'  # Default folder
    
    logger.info(f"Uploading {filename} to {folder} folder...")
    
    try:
        # Create Supabase client
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        
        # Upload file to specific folder
        with open(file_path, 'rb') as f:
            file_data = f.read()
            
        # Upload to 'videos' bucket in appropriate folder
        path = f'{folder}/{filename}'
        response = supabase.storage.from_('videos').upload(
            path,
            file_data,
            {"content-type": "video/mp4"}
        )
        
        # Get public URL
        url = supabase.storage.from_('videos').get_public_url(path)
        
        logger.info(f"Upload successful to {folder}: {url}")
        return url
        
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        return f"file://{file_path}"


@app.route('/health', methods=['GET'])
def health():
    logger.info("Health check requested")
    return jsonify({'status': 'healthy', 'models_loaded': True})

if __name__ == '__main__':
    logger.info(f"Starting Flask server on port 5001...")
    app.run(host='0.0.0.0', port=5001)