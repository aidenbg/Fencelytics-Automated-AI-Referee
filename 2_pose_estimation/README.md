# Pose Estimation

Fencelytics uses **MediaPipe Pose (BlazePose)** to estimate human body landmarks for each fencer.

Pose estimation is necessary because many fencing actions cannot be identified reliably from object position alone. The system uses changes in body geometry and posture over time to provide features for action recognition and right-of-way analysis.

MediaPipe provides body landmarks in both image coordinates and world coordinates. Image landmarks are used for scene-relative positioning, while world landmarks provide more stable measurements for calculating body angles and other pose-based features.

The pose information is used to calculate features such as:

* Elbow angle
* Front-knee angle
* Stance width
* Armpit angle
* Changes in these measurements over time

These features are later used by the action-recognition models to distinguish fencing actions such as **Extend, Unextend, Parry, Lunge, and Recover**.

## Tracking

Pose landmarks are tracked across video frames to provide a continuous representation of each fencer's movement. Fencer detections from the object-detection stage are used to associate poses with the appropriate fencer.

Using temporal pose information provides more stable action features than relying on a single frame.

## Limitations

Pose estimation can become less reliable when fencers overlap, are partially occluded, move rapidly, or appear at difficult camera angles. These issues can affect downstream action recognition.

The system therefore uses pose estimation as one component of a larger pipeline rather than relying on pose alone to determine fencing actions or right-of-way.
