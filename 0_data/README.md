# Fencelytics Research Data

This directory documents the datasets used to develop and evaluate the Fencelytics computer vision and machine learning pipeline for automated foil fencing analysis.

The datasets support three primary learned components:

1. Object detection
2. Piste segmentation
3. Action recognition

The research uses recorded foil-fencing footage from professional broadcasts and self-recorded online footage. Short clips and frames were extracted from this footage for model development and evaluation.

---

## Dataset Overview

| Component          | Source Material      |           Dataset Size | Purpose                    |
| ------------------ | -------------------- | ---------------------: | -------------------------- |
| Object Detection   | 11 fencing sequences |           1,208 images | YOLOv7 detection           |
| Piste Segmentation | 13 fencing videos    |             130 images | RF-DETR piste segmentation |
| Action Recognition | 16 fencing videos    | 5,602 temporal samples | XGBoost action recognition |

---

# 1. Object Detection

## Purpose

The object detection dataset was created to train a YOLOv7 model to identify fencing-specific objects required by downstream stages of the Fencelytics pipeline.

The four detection classes are:

* **Fencer**
* **Foil**
* **Bellguard**
* **Scoreboard**

These detections provide information used for fencer tracking, pose association, weapon-related analysis, and scoreboard touch detection.

## Dataset

The final object-detection model, `yolov7_v4`, was trained using `master_dataset_v2`.

| Dataset             | Sequences | Images | Training | Validation |
| ------------------- | --------: | -----: | -------: | ---------: |
| `master_dataset_v2` |        11 |  1,208 |      962 |        246 |

The dataset was constructed from the following sequence datasets:

| Dataset                 |    Images | Training | Validation |
| ----------------------- | --------: | -------: | ---------: |
| `sequence_0001_dataset` |       162 |      129 |         33 |
| `sequence_0002_dataset` |       170 |      136 |         34 |
| `sequence_0003_dataset` |       147 |      117 |         30 |
| `sequence_0004_dataset` |        96 |       76 |         20 |
| `sequence_0005_dataset` |       117 |       93 |         24 |
| Additional sequences    |       516 |      411 |        105 |
| **Total**               | **1,208** |  **962** |    **246** |

The final YOLOv7 model achieved a validation AP50 of **0.957** and AP50–95 of **0.804**.

---

# 2. Piste Segmentation

## Purpose

The piste segmentation dataset was created to identify the visible fencing piste and provide the geometric information required to construct a piste-relative coordinate system.

The resulting segmentation mask is used to estimate the longitudinal direction of the piste. Fencer positions can then be projected onto this axis for movement analysis.

## Dataset

The research segmentation model was trained using `dataset_v1`.

| Dataset      | Source Videos | Images | Training | Validation |
| ------------ | ------------: | -----: | -------: | ---------: |
| `dataset_v1` |            13 |    130 |      106 |         24 |

Ten frames were included from each of the 13 videos.

The videos were selected to provide diversity in:

* Piste orientation
* Lighting conditions
* Camera angles

The sequences used to construct this dataset were:

```text
1, 2, 3, 4, 5, 7, 8, 10, 12, 13, 14, 15, 16
```

RF-DETR was used for piste segmentation. The resulting mask was processed using principal component analysis (PCA) to estimate the longitudinal direction of the piste, after which the piste axis was constructed for downstream motion analysis.

---

# 3. Action Recognition

## Purpose

The action-recognition dataset was created to train machine-learning models to recognize fencing actions from pose and motion features.

Rather than using a single model for all actions, Fencelytics uses separate upper-body and lower-body XGBoost models.

This separation allows upper- and lower-body actions to occur independently or simultaneously.

## Source Videos

The action-recognition dataset was created from **16 fencing videos**.

Videos were selected with an emphasis on:

* Diversity of fencing footage
* Clear examples of the target actions
* Variation in fencing sequences and movement

Action intervals were manually annotated for each fencer and body region.

Annotations were defined as temporal intervals identifying specific actions. A preprocessing script then converted these annotations into tabular feature datasets used for model training.

An example annotation is conceptually structured as:

```text
Fencer
├── Upper body
│   ├── Action
│   └── Frame interval
└── Lower body
    ├── Action
    └── Frame interval
```

Frames outside explicitly annotated intervals were assigned to `Neutral` when annotations existed for the relevant body region. Unannotated body regions were excluded rather than automatically assigned to `Neutral`.

## Upper-Body Dataset

The upper-body model recognizes:

* **Extend**
* **Unextend**
* **Parry**
* **Neutral**

The 15-frame temporal representation contains four upper-body features at each time step:

* Armpit angle
* Armpit-angle derivative
* Elbow angle
* Elbow-angle derivative

The resulting 15-frame upper-body dataset contains **3,153 temporal samples**.

## Lower-Body Dataset

The lower-body model recognizes:

* **Lunge**
* **Recover**
* **Neutral**

The lower-body model uses six features:

* Piste-relative position
* Velocity
* Acceleration
* Front-knee angle
* Stance width
* Stance-width derivative

The resulting 15-frame lower-body dataset contains **2,449 temporal samples**.

## Temporal Representations

Three temporal representations were evaluated:

1. **Single-frame features**
2. **15-frame temporal window**
3. **5/10-frame lag features**

The same video-level training/validation split was maintained across the three representations.

The 15-frame representation produced the highest validation macro-F1 for both models and was therefore selected for the reported action-recognition results.

| Representation  | Upper-body Macro-F1 | Lower-body Macro-F1 |
| --------------- | ------------------: | ------------------: |
| Single frame    |                0.56 |                0.68 |
| 15-frame window |            **0.62** |            **0.82** |
| 5/10-frame lags |                0.59 |                0.79 |

The public repository provides the upper- and lower-body datasets corresponding to the selected 15-frame models.

---

# Source Footage and Redistribution

The Fencelytics research datasets were derived from recorded foil-fencing footage from professional broadcasts and self-recorded online footage.

Not all original source footage is redistributed in this repository. In particular, publicly accessible footage is not necessarily footage that can itself be redistributed.

Where original source footage cannot be redistributed, the repository provides the derived research data and documentation where appropriate.

---

# Data Privacy

Any private or proprietary data included in the public research repository should be reviewed before publication to ensure that unnecessary personally identifying information is not exposed.

The repository should not contain:

* Passwords
* API keys
* Authentication credentials
* Private URLs or tokens
* Private server information
* Unrelated personal information

---

# Reproducibility

The datasets in this directory are intended to document the data used in the Fencelytics research and to support further research into automated foil fencing analysis.

The accompanying code and model files provide reference implementations for the corresponding components of the Fencelytics pipeline.
