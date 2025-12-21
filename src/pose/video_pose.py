import cv2 as cv
from ultralytics import YOLO
import os
import argparse
import sys
import numpy as np


"""
This script takes the path to a video file, runs pose estimation on each frame
using a yolo pose model,
and overlays the video with pose keypoints and skeletons drawn on it, saving
the result to a specified output directory
"""

def run_pose_extraction_on_video(input_video_path, output_dir):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Generate output video filename based on input video name
    input_basename = os.path.splitext(os.path.basename(input_video_path))[0]
    output_video_path = os.path.join(output_dir, f"{input_basename}_pose.mp4")

    # Load Pose Model
    model = YOLO("yolo11m-pose.pt") 
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

    print("[INFO] Processing video. Press 'q' to quit early.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run pose detection
        results = model(frame)

        # Draw keypoints and skeletons
        plotted_frame = results[0].plot(
            boxes=False,
            labels=False
        )

        # Write the frame to the output video
        out.write(plotted_frame)

        # Display the frame (optional)
        cv.imshow("Pose Estimation", plotted_frame)
        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    out.release()
    cv.destroyAllWindows()
    print(f"Output video saved to: {output_video_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 Pose Estimation on Video")
    parser.add_argument("--input", required=True, help="Path to input video file")
    parser.add_argument("--output", required=True, help="Path to output video directory")
    args = parser.parse_args()
    run_pose_extraction_on_video(args.input, args.output)