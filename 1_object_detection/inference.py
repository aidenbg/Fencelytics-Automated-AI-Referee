#!/usr/bin/env python3
"""
inference.py

YOLOv7 inference: runs object detection on an image, a video, or a
folder containing a mix of both, and saves the results -- detections drawn
as boxes with class + confidence labels -- to an output directory.

Supported inputs:
    Images: .jpg, .jpeg, .png
    Videos: .mp4, .mov

Setup:
    pip install torch opencv-python

Usage:
    python inference.py --source image.jpg
    python inference.py --source video.mp4
    python inference.py --source path/to/folder/ --output results/

    Edit MODEL_PATH below to point at your weights, or override per run with
    --weights.
"""

import argparse
from pathlib import Path

import cv2
import torch

# Edit this to point at your trained weights. Overridable with --weights.
MODEL_PATH = "best.pt"

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
VIDEO_EXTS = {".mp4", ".mov"}


def load_model(weights, conf_thres, iou_thres, device):
    """Loads a custom-trained YOLOv7 model via torch.hub -- no local clone of
    the YOLOv7 repo needed, it's fetched and cached automatically.

    YOLOv7's hubconf.py unconditionally calls check_requirements() at import
    time, which compares installed package versions against YOLOv7's own
    (stale, ~2022) requirements.txt -- e.g. numpy<1.24 -- and on any mismatch
    shells out to `pip install` to "fix" it. There's no parameter to disable
    this. In a modern environment (numpy>=1.24 is normally needed by other
    packages anyway) this either fails outright (no `pip` on PATH in a uv
    venv) or, worse, would succeed and downgrade numpy under you, breaking
    every other package that needs numpy>=1.24. Patching pkg_resources.require
    to a no-op makes check_requirements() see no mismatches at all, so it
    never reaches the pip-install step -- without touching or vendoring any
    of YOLOv7's own code. Requires setuptools<81 installed (pkg_resources was
    removed in later setuptools); see requirements.txt.
    """
    try:
        import pkg_resources
        pkg_resources.require = lambda *a, **k: None
    except ImportError:
        pass  # nothing to patch; check_requirements() will just fail the same way YOLOv7 itself would

    model = torch.hub.load("WongKinYiu/yolov7", "custom", weights, trust_repo=True)
    model.conf = conf_thres
    model.iou = iou_thres
    return model.to(device)


def render_detections(results):
    """Uses YOLOv7's own built-in rendering (boxes, labels, per-class colors)
    rather than drawing them by hand. results.render() draws onto the RGB
    image that was passed to the model and returns it; converted back to BGR
    here since that's what cv2 needs for correct-looking saved output."""
    rendered_rgb = results.render()[0]
    return cv2.cvtColor(rendered_rgb, cv2.COLOR_RGB2BGR)


def process_image(model, path, output_dir):
    frame = cv2.imread(str(path))
    if frame is None:
        print(f"  Could not read image, skipping: {path}")
        return

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # model expects RGB; cv2 loads BGR
    results = model(rgb)
    annotated = render_detections(results)

    out_path = Path(output_dir) / path.name
    cv2.imwrite(str(out_path), annotated)
    print(f"  Saved: {out_path}")


def process_video(model, path, output_dir):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = Path(output_dir) / f"{path.stem}_out.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    frame_count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_count += 1

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(rgb)
        annotated = render_detections(results)
        writer.write(annotated)

        if frame_count % 30 == 0:
            print(f"  ...{frame_count} frames processed")

    cap.release()
    writer.release()
    print(f"  Saved: {out_path} ({frame_count} frames)")


def main():
    parser = argparse.ArgumentParser(description="YOLOv7 inference on images and videos.")
    parser.add_argument("--weights", default=MODEL_PATH, help=f"Path to a YOLOv7 .pt weights file (default: {MODEL_PATH})")
    parser.add_argument("--source", required=True, help="Image, video, or folder of them")
    parser.add_argument("--output", default="outputs", help="Output directory (default: outputs/)")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold (default: 0.45)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run on (default: cuda if available, else cpu)")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.weights} on {args.device}...")
    model = load_model(args.weights, args.conf_thres, args.iou_thres, args.device)

    source = Path(args.source)
    files = sorted(source.iterdir()) if source.is_dir() else [source]

    for f in files:
        ext = f.suffix.lower()
        if ext in IMAGE_EXTS:
            print(f"Image: {f}")
            process_image(model, f, output_dir)
        elif ext in VIDEO_EXTS:
            print(f"Video: {f}")
            process_video(model, f, output_dir)
        else:
            print(f"Skipping unsupported file: {f}")

    print("Done.")


if __name__ == "__main__":
    main()
