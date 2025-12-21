import cv2
import mediapipe as mp
import numpy as np
import os
import argparse
import sys
import torch
from collections import defaultdict

# YOLO model path - change this to your custom model
YOLO_MODEL_PATH = 'FencingOLDJul16best.pt'  # Your YOLOv5 model

# YOLOv5 model loading
print(f"Loading YOLOv5 model: {YOLO_MODEL_PATH}")
try:
    # Load YOLOv5 model
    yolov5_model = torch.hub.load('ultralytics/yolov5', 'custom', 
                                  path=YOLO_MODEL_PATH, 
                                  force_reload=False,
                                  trust_repo=True)  # Add trust_repo to avoid warning
    
    # Print model classes to help identify correct class index
    print("Model classes:", yolov5_model.names)
    print("Model loaded successfully!")
    
except Exception as e:
    print(f"Error loading model: {e}")
    print("\nMake sure you have installed YOLOv5 dependencies:")
    print("pip install seaborn pandas")
    sys.exit(1)

class TemporalSmoother:
    """Smooth pose landmarks across frames for stability"""
    def __init__(self, alpha=0.7):
        self.alpha = alpha
        self.prev_landmarks = {}
    
    def smooth(self, fencer_id, landmarks):
        if fencer_id in self.prev_landmarks:
            # Smooth with previous frame
            for i, lm in enumerate(landmarks.landmark):
                prev = self.prev_landmarks[fencer_id].landmark[i]
                lm.x = self.alpha * lm.x + (1 - self.alpha) * prev.x
                lm.y = self.alpha * lm.y + (1 - self.alpha) * prev.y
        
        self.prev_landmarks[fencer_id] = landmarks
        return landmarks

class FencingPoseExtractor:
    def __init__(self, use_smoothing=False, show_bbox=False):
        # Use the global YOLOv5 model
        self.yolo = yolov5_model
        
        # Initialize MediaPipe
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Single pose detector (we'll run it on each person)
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,  # Important for cropped images
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Temporal smoother (only if enabled)
        self.use_smoothing = use_smoothing
        if self.use_smoothing:
            self.smoother = TemporalSmoother(alpha=0.7)
            print("Temporal smoothing enabled")
        
        # Show bounding boxes option
        self.show_bbox = show_bbox
        if self.show_bbox:
            print("Bounding box display enabled")
        
        # Track fencer IDs across frames
        self.fencer_tracking = {}
        self.next_id = 0
    
    def track_fencer(self, bbox, existing_fencers):
        """
        Simple tracking: match bbox to existing fencers or create new ID
        """
        x1, y1, x2, y2 = bbox
        center = [(x1 + x2) / 2, (y1 + y2) / 2]
        
        # Find closest existing fencer
        min_dist = float('inf')
        matched_id = None
        
        for fid, prev_bbox in existing_fencers.items():
            px1, py1, px2, py2 = prev_bbox
            prev_center = [(px1 + px2) / 2, (py1 + py2) / 2]
            
            dist = np.sqrt((center[0] - prev_center[0])**2 + (center[1] - prev_center[1])**2)
            
            if dist < min_dist and dist < 100:  # Within 100 pixels
                min_dist = dist
                matched_id = fid
        
        if matched_id is None:
            # New fencer
            matched_id = self.next_id
            self.next_id += 1
        
        return matched_id
    
    def detect_fencers(self, frame):
        """
        Detect fencers using YOLOv5
        """
        # Run YOLOv5 inference
        self.yolo.conf = 0.5  # Set confidence threshold
        results = self.yolo(frame)
        
        fencer_bboxes = []
        
        # YOLOv5 returns results in a different format
        # Get the pandas dataframe with detections
        detections = results.pandas().xyxy[0]
        
        for idx, detection in detections.iterrows():
            # For YOLOv5, check the class name or class index
            # Update this based on your model's class names
            # If your model has 'Fencer' class, use: if detection['name'] == 'Fencer':
            # If using class index: if int(detection['class']) == 0:
            
            # Check if this is a Fencer class
            # You can use either class name or class index
            class_name = detection['name']
            class_id = int(detection['class'])
            
            # Update this condition based on your model
            # Option 1: Check by class name
            if class_name == 'Fencer':  # Change 'Fencer' to your actual class name
            # Option 2: Check by class index
            # if class_id == 9:  # Change 9 to your fencer class index
                x1 = int(detection['xmin'])
                y1 = int(detection['ymin'])
                x2 = int(detection['xmax'])
                y2 = int(detection['ymax'])
                conf = float(detection['confidence'])
                
                fencer_bboxes.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': conf
                })
        
        # Sort by x-coordinate to maintain left/right consistency
        fencer_bboxes.sort(key=lambda x: x['bbox'][0])
        
        # Keep only top 2 fencers by confidence
        fencer_bboxes = sorted(fencer_bboxes, key=lambda x: x['confidence'], reverse=True)[:2]
        
        # Re-sort by x-coordinate
        fencer_bboxes.sort(key=lambda x: x['bbox'][0])
        
        return fencer_bboxes
    
    def extract_pose_from_bbox(self, frame, bbox, padding=30):
        """
        Extract pose from a bounding box region
        """
        x1, y1, x2, y2 = bbox
        height, width = frame.shape[:2]
        
        # Add padding
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(width, x2 + padding)
        y2 = min(height, y2 + padding)
        
        # Crop fencer
        fencer_crop = frame[y1:y2, x1:x2]
        
        if fencer_crop.size == 0:
            return None
        
        # Run MediaPipe on crop
        rgb_crop = cv2.cvtColor(fencer_crop, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_crop)
        
        if results.pose_landmarks:
            # Adjust coordinates back to full frame
            crop_height = y2 - y1
            crop_width = x2 - x1
            
            for landmark in results.pose_landmarks.landmark:
                # Convert from crop coordinates to full frame coordinates
                landmark.x = (landmark.x * crop_width + x1) / width
                landmark.y = (landmark.y * crop_height + y1) / height
            
            return results.pose_landmarks
        
        return None

    def process_video(self, input_path, output_dir, overlay=False):
        """
        Process video with YOLOv5 detection and MediaPipe pose estimation
        
        Args:
            input_path: Path to input video
            output_dir: Directory for output video
            overlay: If True, overlay on original video; if False, use black background
        """
        # Check if input exists
        if not os.path.exists(input_path):
            print(f"Error: Input video '{input_path}' not found!")
            return
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
        
        # Generate output filename based on input
        input_filename = os.path.basename(input_path)
        name_without_ext = os.path.splitext(input_filename)[0]
        suffix = "_poses_overlay" if overlay else "_poses_black"
        if self.use_smoothing:
            suffix += "_smoothed"
        if self.show_bbox:
            suffix += "_bbox"
        output_path = os.path.join(output_dir, f"{name_without_ext}{suffix}.mp4")
        
        print(f"Processing: {input_path}")
        print(f"Mode: {'Overlay on original' if overlay else 'Black background'}")
        print(f"Smoothing: {'Enabled' if self.use_smoothing else 'Disabled'}")
        print(f"Bounding boxes: {'Shown' if self.show_bbox else 'Hidden'}")
        print(f"Output will be saved to: {output_path}")
        
        # Open video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"Error: Could not open video '{input_path}'")
            return
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video info: {width}x{height} @ {fps} FPS, {total_frames} frames")
        
        # Create output video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            output_path = output_path.replace('.mp4', '.avi')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        frames_with_two_fencers = 0
        frames_with_one_fencer = 0
        frames_with_no_fencers = 0
        
        print(f"Processing frames...")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Create output frame based on overlay setting
            if overlay:
                output_frame = frame.copy()
            else:
                output_frame = np.zeros_like(frame)
            
            # Detect fencers using YOLOv5
            fencer_detections = self.detect_fencers(frame)
            
            # Update tracking
            current_fencers = {}
            detected_poses = []
            
            for i, detection in enumerate(fencer_detections):
                bbox = detection['bbox']
                
                # Track fencer ID
                fencer_id = self.track_fencer(bbox, self.fencer_tracking)
                current_fencers[fencer_id] = bbox
                
                # Extract pose
                pose_landmarks = self.extract_pose_from_bbox(frame, bbox)
                
                if pose_landmarks:
                    # Apply temporal smoothing if enabled
                    if self.use_smoothing:
                        pose_landmarks = self.smoother.smooth(fencer_id, pose_landmarks)
                    
                    detected_poses.append({
                        'id': fencer_id,
                        'landmarks': pose_landmarks,
                        'bbox': bbox,
                        'confidence': detection['confidence'],
                        'side': 'left' if i == 0 else 'right'
                    })
            
            # Update tracking dictionary
            self.fencer_tracking = current_fencers
            
            # Draw poses
            for pose_data in detected_poses:
                # Draw bounding box if enabled
                if self.show_bbox:
                    x1, y1, x2, y2 = pose_data['bbox']
                    # Draw rectangle
                    cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)  # Yellow box
                    # Add confidence if available
                    if 'confidence' in pose_data:
                        conf_text = f"{pose_data['confidence']:.2f}"
                        cv2.putText(output_frame, conf_text, (x1, y1 - 25),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                
                # Draw pose skeleton
                self.mp_drawing.draw_landmarks(
                    output_frame,
                    pose_data['landmarks'],
                    self.mp_pose.POSE_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),  # Green keypoints
                    self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)  # Blue connections
                )
                
                # Add fencer label
                x1, y1, x2, y2 = pose_data['bbox']
                label = f"Fencer {pose_data['id'] + 1} ({pose_data['side'].capitalize()})"
                label_y = y1 - 10 if not self.show_bbox else y1 - 40  # Adjust position if bbox shown
                cv2.putText(output_frame, label, (x1, label_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Update statistics
            num_detected = len(detected_poses)
            if num_detected == 2:
                frames_with_two_fencers += 1
            elif num_detected == 1:
                frames_with_one_fencer += 1
            else:
                frames_with_no_fencers += 1
            
            # Add frame counter
            cv2.putText(output_frame, f"Frame: {frame_count} | Detected: {num_detected}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Write frame
            out.write(output_frame)
            
            if frame_count % 30 == 0:
                print(f"Processed {frame_count}/{total_frames} frames...")
            
            frame_count += 1
        
        # Clean up
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        # Verify output
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            file_size = os.path.getsize(output_path) / (1024 * 1024)
            
            print(f"\n✓ Success!")
            print(f"  - Processed frames: {frame_count}")
            print(f"  - Frames with 2 fencers: {frames_with_two_fencers} ({frames_with_two_fencers/frame_count*100:.1f}%)")
            print(f"  - Frames with 1 fencer: {frames_with_one_fencer} ({frames_with_one_fencer/frame_count*100:.1f}%)")
            print(f"  - Frames with 0 fencers: {frames_with_no_fencers} ({frames_with_no_fencers/frame_count*100:.1f}%)")
            print(f"  - Output file: {output_path} ({file_size:.1f} MB)")
        else:
            print(f"\n✗ Error: Output file was not created properly")

def main():
    parser = argparse.ArgumentParser(
        description='Extract poses using YOLOv5 detection + MediaPipe pose estimation'
    )
    parser.add_argument('--input', required=True, help='Path to input video file')
    parser.add_argument('--output', required=True, help='Output directory for processed video')
    parser.add_argument('--overlay', action='store_true', 
                       help='Overlay on original video (default: black background)')
    parser.add_argument('--smooth', action='store_true',
                       help='Enable temporal smoothing for more stable poses')
    parser.add_argument('--showbbox', action='store_true',
                       help='Show bounding boxes from YOLO detection')
    
    args = parser.parse_args()
    
    # Create processor with smoothing and bbox options
    processor = FencingPoseExtractor(use_smoothing=args.smooth, show_bbox=args.showbbox)
    
    # Process video
    processor.process_video(args.input, args.output, args.overlay)

if __name__ == "__main__":
    main()

# Usage:
# Black background without smoothing:
# python pose_extractor.py --input video.mp4 --output output_dir/
#
# Black background with temporal smoothing:
# python pose_extractor.py --input video.mp4 --output output_dir/ --smooth
#
# Overlay on original video with smoothing:
# python pose_extractor.py --input video.mp4 --output output_dir/ --overlay --smooth
#
# Show bounding boxes from YOLO detection:
# python pose_extractor.py --input video.mp4 --output output_dir/ --showbbox
#
# Combine all options:
# python pose_extractor.py --input video.mp4 --output output_dir/ --overlay --smooth --showbbox
#
# To use your custom YOLOv5 model, update the YOLO_MODEL_PATH variable at the top
# 
# IMPORTANT: Check your model's class names when the script starts:
# - If 'Fencer' is not the class name, update line 133: if class_name == 'YOUR_CLASS_NAME':
# - Or use class index instead: if class_id == YOUR_CLASS_INDEX: