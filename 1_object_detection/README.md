# Object Detection

Fencelytics uses YOLOv7 for object detection in fencing video. The detector identifies four classes:

* **Fencer** — detects the two fencers in the bout. These detections are used to locate each fencer and to exclude the fencers from background feature matching during ORB-based camera-motion compensation.
* **Foil** — originally intended to support foil/blade position estimation and future weapon-motion analysis.
* **Bellguard** — originally intended to support foil/blade position estimation and future weapon-motion analysis.
* **Scoreboard** — detects the electronic scoreboard region for downstream score-event detection.

The scoreboard detections are passed to a later stage of the pipeline, where the scoreboard region is analyzed to identify scoring events.

The foil and bellguard detections were included to support weapon estimation, but they were not accurate enough for reliable downstream use. Thin foil blades are particularly difficult to detect consistently in fencing video because of their small size, motion, occlusion, and visual appearance.

The final YOLOv7 model was trained on fencing images from multiple video sequences and achieved an AP50 of 0.957 and AP50-95 of 0.804 on the validation set.

## Model

The trained model is located in:

```text
model/yolov7_v4.pt
```

## Inference

`inference.py` provides inference on individual images, videos, or folders of images.

Example outputs are provided in:

```text
examples/
```
