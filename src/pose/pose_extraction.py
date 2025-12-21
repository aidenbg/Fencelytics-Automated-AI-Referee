#This script extracts multiple poses from images of two fencers, and saves
#The image with the pose keypoints and skeleton drawn on it in a 
#specified output directory

import os
import cv2 as cv
import argparse
from ultralytics import YOLO


#Runs poses on images, saves them to new directory
def run_pose_extraction(input_dir, output_dir):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load YOLOv8-Pose model
    model = YOLO("yolov8n-pose.pt")  # You can upgrade to yolov8m/l-pose.pt for better accuracy

    # Iterate through all image files
    for file_name in os.listdir(input_dir):
        if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(input_dir, file_name)

            # Run pose detection
            results = model(img_path)

            # Draw keypoints and skeletons
            plotted_img = results[0].plot(
                boxes=False,      # Do not draw bounding boxes
                labels=False      # Do not draw labels
            )

            # Save result
            output_path = os.path.join(output_dir, file_name)
            cv.imwrite(output_path, plotted_img)
            print(f"Processed: {file_name} -> {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 Pose Estimation on Images in Folder")
    parser.add_argument("--input", required=True, help="Folder path to images")
    parser.add_argument("--output", required=True, help="Output folder directory")
    args = parser.parse_args()
    run_pose_extraction(args.input, args.output)