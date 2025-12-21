import torch
import cv2
import os
import argparse

# Variables
yolo_path = '/Users/aidenburagohain/Coding/FencingAI/yolov5-v6.0'  # Path to YOLOv5 repository
model_path = '/Users/aidenburagohain/Coding/FencingAI/models/NEWFencingJul18best.pt'
image_path = '/Users/aidenburagohain/Coding/FencingAI/data/Images/SageVNguyen-00.00.18.326-00.00.21.953/frame_000012.jpg'  # Change to your image
output_dir = '/Users/aidenburagohain/Coding/FencingAI/'

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['yolo', 'manual'], default='yolo')
args = parser.parse_args()

# Load model
print("Loading model...")
model = torch.hub.load(yolo_path, 'custom', path=model_path, source='local')

# Load image
img = cv2.imread(image_path)
if img is None:
    print(f"Error: Cannot read {image_path}")
    exit()

# Run inference
results = model(img)

# Annotate based on mode
if args.mode == 'yolo':
    # Use YOLOv5 render
    annotated = results.render()[0]
else:
    # Manual drawing
    annotated = img.copy()
    for *box, conf, cls in results.xyxy[0].cpu().numpy():
        x1, y1, x2, y2 = map(int, box)
        # Draw box with thickness=1
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 1)
        # Add label
        label = f"{model.names[int(cls)]} {conf:.2f}"
        cv2.putText(annotated, label, (x1, y1-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

# Save output
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, f'output_{args.mode}.jpg')
cv2.imwrite(output_path, annotated)
print(f"Saved to {output_path}")

# Show result
print("Press 'q' to quit")
while True:
    cv2.imshow(f'Result ({args.mode} mode)', annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()