
"""
This script takes the path to a video file, runs pose estimation on each frame
using mediapipe, and overlays the video with pose keypoints and skeletons
drawn only for the two people with the most detected keypoints
It takes 3 arguments:
--input: Path to the input video file
--output: Directory to save the output video
--overlay: adding --overlay to the run command will overlay the pose on the original frame,
otherwise it will use a black background.
"""

import cv2
import mediapipe as mp
import numpy as np
import argparse
import os

def process_video_split_method(input_path, output_dir, overlay=False):
    """
    Simple method: Split frame in half, run pose on each half
    """
    # Initialize MediaPipe
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    
    # Two pose detectors for left and right
    pose_left = mp_pose.Pose(
        static_image_mode=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    pose_right = mp_pose.Pose(
        static_image_mode=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Setup paths
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.splitext(os.path.basename(input_path))[0]
    suffix = "_poses_overlay" if overlay else "_poses_black"
    output_path = os.path.join(output_dir, f"{filename}{suffix}.mp4")
    
    # Open video
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Processing: {input_path}")
    print(f"Output: {output_path}")
    
    frame_count = 0
    mid_x = width // 2
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Create output frame
        output_frame = frame.copy() if overlay else np.zeros_like(frame)
        
        # Process left half
        left_half = frame[:, :mid_x]
        rgb_left = cv2.cvtColor(left_half, cv2.COLOR_BGR2RGB)
        results_left = pose_left.process(rgb_left)
        
        # Process right half  
        right_half = frame[:, mid_x:]
        rgb_right = cv2.cvtColor(right_half, cv2.COLOR_BGR2RGB)
        results_right = pose_right.process(rgb_right)
        
        # Draw left person (same color as right)
        if results_left.pose_landmarks:
            # Adjust coordinates to full frame
            for landmark in results_left.pose_landmarks.landmark:
                landmark.x = landmark.x / 2
            
            mp_drawing.draw_landmarks(
                output_frame,
                results_left.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),  # Green keypoints
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)  # Blue connections
            )
            
            # Count keypoints
            left_keypoints = sum(1 for lm in results_left.pose_landmarks.landmark if lm.visibility > 0.5)
            cv2.putText(output_frame, f"Left: {left_keypoints} pts", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw right person (same color)
        if results_right.pose_landmarks:
            # Adjust coordinates to full frame
            for landmark in results_right.pose_landmarks.landmark:
                landmark.x = (landmark.x + 1) / 2
            
            mp_drawing.draw_landmarks(
                output_frame,
                results_right.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),  # Green keypoints
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)  # Blue connections
            )
            
            # Count keypoints
            right_keypoints = sum(1 for lm in results_right.pose_landmarks.landmark if lm.visibility > 0.5)
            cv2.putText(output_frame, f"Right: {right_keypoints} pts", (width - 150, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add frame counter
        cv2.putText(output_frame, f"Frame: {frame_count}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        out.write(output_frame)
        
        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames...")
        
        frame_count += 1
    
    # Cleanup
    cap.release()
    out.release()
    pose_left.close()
    pose_right.close()
    cv2.destroyAllWindows()
    
    print(f"Done! Processed {frame_count} frames")
    print(f"Output saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simple two-person pose estimation')
    parser.add_argument('--input', required=True, help='Input video path')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--overlay', action='store_true', help='Overlay on original')
    
    args = parser.parse_args()
    process_video_split_method(args.input, args.output, args.overlay)