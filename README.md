# Fencelytics: AI-Based Fencing Analysis

Fencelytics is a computer vision and machine learning system designed to analyze foil fencing bouts and assist with automated refereeing.

The system combines fencing-specific object detection, human pose estimation, piste segmentation, motion tracking, temporal action recognition, and rule-based reasoning to analyze fencing exchanges and determine right-of-way.

## The Problem

Foil fencing presents a challenging problem for automated sports analysis.

Determining the outcome of a fencing exchange is not based solely on detecting whether a touch occurred. In foil, the referee must interpret the sequence of actions performed by both fencers and apply the rules of right-of-way to determine which fencer has priority for the touch.

This requires understanding multiple aspects of a bout simultaneously:

* Where the fencers and their equipment are located
* How the fencers are moving along the piste
* The positions and movements of their bodies
* What fencing actions are being performed
* Whether a scoring event occurred
* How the sequence of actions determines right-of-way

Fencelytics explores whether these observations can be extracted from video and combined into an automated analysis pipeline.

## Approach

Fencelytics decomposes the problem into several stages:

**Object Detection → Pose Estimation → Piste Segmentation → Motion Tracking → Action Recognition → Right-of-Way**

### 1. Object Detection

A fencing-specific object detection model identifies the objects required for downstream analysis:

* Fencer
* Foil
* Bellguard
* Scoreboard

The detections provide the spatial information needed by later stages of the pipeline.

[View Object Detection →](object_detection/README.md)

### 2. Pose Estimation

Human pose estimation extracts skeletal landmarks from the detected fencers.

These landmarks provide information about body position and movement that can be used to characterize fencing actions.

[View Pose Estimation →](pose_estimation/README.md)

### 3. Piste Segmentation

The fencing piste is segmented to identify its boundaries and establish a geometric reference for the bout.

This allows subsequent movement analysis to operate relative to the piste rather than relying solely on image coordinates.

[View Piste Segmentation →](piste_segmentation/README.md)

### 4. Motion Tracking

Fencelytics tracks fencer movement along the piste axis.

The system also accounts for camera motion so that movement measurements are less dependent on changes in the camera's position.

[View Motion Tracking →](motion_tracking/README.md)

### 5. Action Recognition

Pose and motion information are transformed into temporal features describing the movements of the fencers.

Machine-learning models use these features to classify fencing actions.

[View Action Recognition →](action_recognition/README.md)

### 6. Right-of-Way

The final stage combines recognized actions, fencer movement, and scoring information using a rule-based state machine.

The goal is to determine the right-of-way outcome of a fencing exchange.

[View Right-of-Way →](right_of_way/README.md)

---

## Research Contribution

Fencelytics investigates an end-to-end approach to automated foil fencing analysis by combining computer vision, machine learning, and fencing-specific rule reasoning.

The system integrates several components that are individually useful but are designed here to work together around the requirements of foil fencing:

* **Fencing-specific object detection** for identifying fencers, foils, bellguards, and scoreboards
* **Piste segmentation** to establish a geometric reference for movement
* **Camera-motion correction** to distinguish fencer movement from camera movement
* **Pose and motion feature extraction** to represent fencing movement
* **Temporal action recognition** to classify fencing actions
* **Rule-based right-of-way reasoning** to connect recognized actions and scoring information to the rules of foil fencing

Rather than treating fencing as a generic human-action-recognition problem, Fencelytics uses the structure of the sport itself to inform the computer-vision and machine-learning pipeline.

> Specific claims regarding the novelty of this approach are described in the accompanying research paper and will be stated here once the relevant prior work has been fully documented.

---

## Repository

This repository contains the research data, dataset documentation, representative models, inference code, and example outputs associated with the Fencelytics project.

```text
data/                  Research datasets and dataset documentation

object_detection/     Fencing object detection
pose_estimation/      Human pose estimation
piste_segmentation/   Piste segmentation
motion_tracking/      Fencer movement tracking
action_recognition/   Fencing action recognition
right_of_way/         Right-of-way determination

examples/             Example outputs and demonstrations
```

Each component contains its own documentation describing its purpose, data, methodology, and implementation.

---

## Research Data

The `data/` directory documents the datasets used in the Fencelytics research.

Where redistribution is permitted, the relevant data and annotations are provided directly. For source material that cannot be redistributed, the repository provides appropriate documentation describing the source data and its role in the research.

[View Research Data →](data/README.md)

---

## Reproducibility

The code and models included in this repository are intended to provide a reference implementation of the methods used in Fencelytics.

Some models included in the repository may be representative models provided for demonstration and reproducibility rather than the exact model checkpoint used for every reported experiment.

Where applicable, the documentation for each component identifies the dataset, model, and methodology associated with that component.

---

## Research Paper

The Fencelytics research paper describes the complete methodology, experimental design, results, and analysis.

The paper is not included in this repository at this time.

---

## Project Status

Fencelytics is an ongoing research project exploring computer vision and machine learning for automated analysis of foil fencing.

---

## License

See [LICENSE](LICENSE).
