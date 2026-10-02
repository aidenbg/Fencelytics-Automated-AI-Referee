# Piste Segmentation

Fencelytics uses **RF-DETR** to identify the fencing piste in video and establish its longitudinal direction.

This is necessary because fencing movement cannot always be interpreted correctly using image coordinates alone. A simple horizontal position or movement measurement assumes that the piste is horizontal in the video frame. However, fencing footage can be recorded from different camera positions and angles, meaning the piste may appear diagonal or otherwise rotated relative to the image.

If movement were tracked only horizontally, a fencer moving along the piste could therefore appear to have reduced or incorrect movement, while camera perspective could introduce additional apparent motion.

## Piste-Relative Coordinate System

Fencelytics uses the detected piste to construct a **piste-relative coordinate system**.

The segmentation model identifies the piste, after which the system estimates its longitudinal direction using PCA and robust line fitting. The center of the piste is then used to define a central axis.

Fencer positions are projected onto this axis, producing a one-dimensional position along the piste. This allows movement to be measured relative to the actual fencing strip rather than the orientation of the video frame.

This piste-relative position is subsequently used to calculate movement features such as:

* Position along the piste
* Velocity
* Acceleration

These features are used by the action-recognition and right-of-way components of the pipeline.

## Example

https://github.com/user-attachments/assets/7cba372f-ae77-415e-bbed-7e6c9b06aa42

## Dataset

The piste segmentation model was trained using 130 images from 13 fencing videos, with variation in camera orientation, lighting, and viewpoint.

## Limitations

Piste segmentation can be affected by unusual camera angles, occlusion, lighting conditions, and footage where the boundaries of the piste are difficult to distinguish.

The system therefore uses the estimated piste geometry as a reference for movement analysis rather than assuming a fixed image orientation.
