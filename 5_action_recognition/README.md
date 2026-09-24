# Action Recognition

Fencelytics uses **XGBoost** to classify fencing actions from temporal pose and movement features.

Action recognition is necessary because determining foil right-of-way requires more than detecting where a fencer is located. The system must identify actions and their timing, such as an extension, parry, lunge, or recovery, and use their sequence to determine what occurred during an exchange.

## Feature Engineering

The action-recognition models use features produced by the pose-estimation and motion-tracking stages.

The features include:

* Piste-relative position
* Velocity
* Acceleration
* Elbow angle
* Elbow-angle derivative
* Front-knee angle
* Stance width
* Stance-width derivative
* Armpit angle
* Armpit-angle derivative

The signals are smoothed using a centered Savitzky-Golay filter before feature extraction.

## Separate Upper and Lower Models

Fencelytics uses separate models for the upper and lower body because the two regions provide different information about fencing actions.

**Upper-body model:**

* Extend
* Unextend
* Parry
* Neutral

The upper-body model uses armpit and elbow angles and their temporal derivatives.

**Lower-body model:**

* Lunge
* Recover
* Neutral

The lower-body model uses piste-relative position, velocity, acceleration, front-knee angle, stance width, and stance-width derivative.

Attack is subsequently represented by the combination of an **Extend** from the upper-body model and a **Lunge** from the lower-body model.

## Temporal Context

Rather than classifying each frame independently, Fencelytics evaluates features across temporal windows.

Three representations were evaluated:

* Single-frame features
* 15-frame temporal windows
* Lagged features using 5- and 10-frame histories

The **15-frame representation** produced the highest macro-F1 among the tested approaches, with:

* Upper body: **0.62 macro-F1**
* Lower body: **0.82 macro-F1**

The public dataset contains the temporal samples used for the selected 15-frame models.

## Limitations

Action recognition is limited by the size and diversity of the annotated dataset. The dataset contains a limited number of fencers, venues, viewpoints, and lighting conditions.

Some actions, particularly **Parry**, are difficult to distinguish reliably from pose and motion features alone. Complex sequences such as parry-riposte exchanges are also not fully represented by the current action-classification system.

The action-recognition models therefore provide temporal action estimates that are used as inputs to the downstream right-of-way reasoning system rather than independently determining the final fencing call.
