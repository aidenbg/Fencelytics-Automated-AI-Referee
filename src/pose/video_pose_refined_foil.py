import cv2 as cv
from ultralytics import YOLO
import os
import argparse
import sys
import numpy as np
import torch

"""
This script takes the path to a video file, runs pose estimation on each frame
using a yolo pose model, and overlays the video with pose keypoints and skeletons
drawn only for the two people with the most detected keypoints.
Optionally, it can overlay artificial foils using bellguard detection.

It takes 4 arguments:
--input: Path to the input video file
--output: Directory to save the output video
--overlay: Boolean flag to determine if the output should overlay on the original frame 
or use a black background
--foil: Path to custom YOLOv5 model for bellguard detection to overlay foils
"""

def run_pose_extraction_on_video(input_video_path, output_dir, overlay=True, foil_model_path=None):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output video filename based on input video name
    input_basename = os.path.splitext(os.path.basename(input_video_path))[0]
    output_suffix = "_pose_overlay" if overlay else "_pose_black"
    if foil_model_path:
        output_suffix += "_foil"
    output_video_path = os.path.join(output_dir, f"{input_basename}{output_suffix}.mp4")
    
    # Load Pose Model
    pose_model = YOLO("yolo11m-pose.pt")
    
    # Load Bellguard Detection Model if provided
    bellguard_model = None
    if foil_model_path:
        try:
            bellguard_model = torch.hub.load('/Users/aidenburagohain/Coding/FencingAI/yolov5-v6.0', 'custom', 
                                           path=foil_model_path, force_reload=False, source='local')
            print(f"[INFO] Loaded bellguard detection model from {foil_model_path}")
        except Exception as e:
            print(f"[ERROR] Failed to load bellguard model: {e}")
            return
    
    # Open the input video
    cap = cv.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise IOError(f"Failed to open video: {input_video_path}")
    
    # Get video properties
    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv.CAP_PROP_FPS)
    
    # Define the codec and create VideoWriter object
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out = cv.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    
    print(f"[INFO] Processing video. Mode: {'Overlay' if overlay else 'Black Background'}. " + 
          f"Foil: {'Enabled' if foil_model_path else 'Disabled'}. Press 'q' to quit early.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run pose detection
        results = pose_model(frame)
        
        # Run bellguard detection if model is loaded
        bellguard_detections = None
        if bellguard_model:
            bellguard_results = bellguard_model(frame)
            bellguard_detections = bellguard_results.pandas().xyxy[0]
            # Filter for Bellguard class only
            bellguard_detections = bellguard_detections[bellguard_detections['name'] == 'Bellguard']
        
        # Create output frame based on overlay setting
        if overlay:
            output_frame = frame.copy()
        else:
            output_frame = np.zeros_like(frame)  # Black background
        
        # Check if any poses were detected
        if results[0].keypoints is not None and results[0].keypoints.data.shape[0] > 0:
            keypoints_data = results[0].keypoints.data.cpu().numpy()  # Shape: (N, 17, 3)
            
            # Count visible keypoints for each person
            keypoint_counts = []
            for i in range(keypoints_data.shape[0]):
                # Count keypoints with confidence > 0.5
                visible_count = np.sum(keypoints_data[i, :, 2] > 0.5)
                keypoint_counts.append((i, visible_count))
            
            # Sort by keypoint count (descending) and get top 2
            keypoint_counts.sort(key=lambda x: x[1], reverse=True)
            top_indices = [x[0] for x in keypoint_counts[:2]]  # Get indices of top 2
            
            # Create filtered keypoints array with only top 2 people
            filtered_keypoints = keypoints_data[top_indices]
            
            # Draw foils if bellguard detection is enabled
            if bellguard_model and bellguard_detections is not None and len(bellguard_detections) > 0:
                output_frame = draw_foils(output_frame, filtered_keypoints, bellguard_detections, frame_height)
            
            # Manually draw the filtered poses
            output_frame = draw_poses_on_frame(output_frame, filtered_keypoints)
            
            # Optional: Display keypoint counts
            for rank, (idx, count) in enumerate(keypoint_counts[:2]):
                cv.putText(output_frame, f"Person {rank+1}: {count} keypoints", 
                          (10, 30 + rank*30), cv.FONT_HERSHEY_SIMPLEX, 
                          0.7, (255, 255, 255), 2)
        
        # Write the frame to the output video
        out.write(output_frame)
        
        # Display the frame (optional)
        window_name = "Pose Estimation - Overlay" if overlay else "Pose Estimation - Black"
        cv.imshow(window_name, output_frame)
        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    cap.release()
    out.release()
    cv.destroyAllWindows()
    
    print(f"Output video saved to: {output_video_path}")

def draw_foils(frame, keypoints, bellguard_detections, frame_height):
    """
    Draw foils based on bellguard detections and wrist keypoints
    """
    # COCO keypoint indices
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    
    # Foil parameters
    FOIL_LENGTH = 160  # Length of foil in pixels (adjust as needed)
    FOIL_THICKNESS = 2
    FOIL_COLOR = (0, 0, 255)  # Red in BGR
    
    for person_idx, person_kpts in enumerate(keypoints):
        # Try both wrists
        for wrist_idx in [LEFT_WRIST, RIGHT_WRIST]:
            if person_kpts[wrist_idx, 2] > 0.5:  # Wrist is visible
                wrist_x = person_kpts[wrist_idx, 0]
                wrist_y = person_kpts[wrist_idx, 1]
                
                # Find closest bellguard to this wrist
                closest_bellguard = None
                min_distance = float('inf')
                
                for _, bellguard in bellguard_detections.iterrows():
                    # Calculate center of bellguard bounding box
                    bell_center_x = (bellguard['xmin'] + bellguard['xmax']) / 2
                    bell_center_y = (bellguard['ymin'] + bellguard['ymax']) / 2
                    
                    # Calculate distance to wrist
                    distance = np.sqrt((bell_center_x - wrist_x)**2 + (bell_center_y - wrist_y)**2)
                    
                    # Consider bellguard if it's reasonably close to wrist (within 150 pixels)
                    if distance < 150 and distance < min_distance:
                        min_distance = distance
                        closest_bellguard = (bell_center_x, bell_center_y)
                
                if closest_bellguard:
                    bell_x, bell_y = closest_bellguard
                    
                    # Calculate angle from bellguard to wrist
                    angle = np.arctan2(wrist_y - bell_y, wrist_x - bell_x)
                    
                    # Calculate foil endpoint (extending from bellguard away from wrist)
                    foil_end_x = int(bell_x - FOIL_LENGTH * np.cos(angle))
                    foil_end_y = int(bell_y - FOIL_LENGTH * np.sin(angle))
                    
                    # Draw foil as a line
                    cv.line(frame, 
                           (int(bell_x), int(bell_y)), 
                           (foil_end_x, foil_end_y), 
                           FOIL_COLOR, 
                           FOIL_THICKNESS)
                    
                    # Optional: Draw bellguard position as a small circle
                    cv.circle(frame, (int(bell_x), int(bell_y)), 3, (255, 255, 0), -1)
    
    return frame

def draw_poses_on_frame(frame, keypoints):
    """
    Manually draw keypoints and skeleton on the frame
    """
    # COCO skeleton connections
    skeleton = [
        [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],
        [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
        [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
        [2, 4], [3, 5], [4, 6], [5, 7]
    ]
    
    # Colors: Blue skeleton, Green keypoints (same for both fencers)
    skeleton_color = (255, 0, 0)  # Blue in BGR
    keypoint_color = (0, 255, 0)  # Green in BGR
    
    for person_idx, person_kpts in enumerate(keypoints):
        # Draw skeleton with blue color
        for connection in skeleton:
            kpt1_idx, kpt2_idx = connection[0] - 1, connection[1] - 1  # Convert to 0-indexed
            
            if (kpt1_idx < len(person_kpts) and kpt2_idx < len(person_kpts) and
                person_kpts[kpt1_idx, 2] > 0.5 and person_kpts[kpt2_idx, 2] > 0.5):
                
                pt1 = (int(person_kpts[kpt1_idx, 0]), int(person_kpts[kpt1_idx, 1]))
                pt2 = (int(person_kpts[kpt2_idx, 0]), int(person_kpts[kpt2_idx, 1]))
                
                cv.line(frame, pt1, pt2, skeleton_color, 2)
        
        # Draw keypoints with green color
        for kpt in person_kpts:
            if kpt[2] > 0.5:  # Only draw if confidence is high enough
                x, y = int(kpt[0]), int(kpt[1])
                cv.circle(frame, (x, y), 5, keypoint_color, -1)
                cv.circle(frame, (x, y), 3, (255, 255, 255), -1)  # White center
    
    return frame

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 Pose Estimation - Top 2 People with Optional Foil Overlay")
    parser.add_argument("--input", required=True, help="Path to input video file")
    parser.add_argument("--output", required=True, help="Path to output video directory")
    parser.add_argument("--overlay", type=str, choices=['True', 'False'], default='True',
                        help="If true, overlay on original video. If false, use black background (default: true)")
    parser.add_argument("--foil", type=str, default=None,
                        help="Path to custom YOLOv5 model for bellguard detection to overlay foils")
    args = parser.parse_args()
    
    # Convert string to boolean
    overlay = args.overlay.lower() == 'true'
    
    run_pose_extraction_on_video(args.input, args.output, overlay, args.foil)