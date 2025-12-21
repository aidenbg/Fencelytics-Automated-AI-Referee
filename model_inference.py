import torch
import cv2
import os

# Configuration variables
YOLO_REPO_PATH = "/Users/aidenburagohain/Coding/FencingAI/yolov5-v6.0"  # Path to local YOLOv5 repository
MODEL_PATH = "/Users/aidenburagohain/Coding/FencingAI/models/NEWFencingJul18best.pt"  # Path to custom .pt model file
INPUT_VIDEO_PATH = "/Users/aidenburagohain/Coding/FencingAI/data/clipped_videos/2025TbilisiFINALChoivBorodachev/2025TbilisiFINALChoivBorodachev-00.00.01.757-00.00.09.005.mp4"  # Path to input video
OUTPUT_DIR = "/Users/aidenburagohain/Coding/FencingAI/data/model_inference"  # Output directory

# Load model
print("Loading model...")
model = torch.hub.load(YOLO_REPO_PATH, 'custom', path=MODEL_PATH, source='local')

# Create output directory if needed
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Open video
cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Setup output video
output_path = os.path.join(OUTPUT_DIR, f"detected_{os.path.basename(INPUT_VIDEO_PATH)}")
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

# Process video
print("Processing video...")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run inference
    results = model(frame)
    
    # Draw boxes and save frame
    annotated_frame = results.render()[0]
    out.write(annotated_frame)

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Done! Output saved to: {output_path}")