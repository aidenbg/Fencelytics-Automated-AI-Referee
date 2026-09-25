# Fencelytics: AI-Based Fencing Analysis

Fencelytics is a computer vision and machine learning system designed to analyze foil fencing bouts and assist with automated refereeing.

The system combines fencing-specific object detection, human pose estimation, piste segmentation, motion tracking, temporal action recognition, and rule-based reasoning to analyze fencing exchanges and determine right-of-way.

## Demo

The following example shows the Fencelytics pipeline producing an automated right-of-way analysis from a fencing video.

<video src="6_right_of_way/examples/ROW_vid_5.mp4" controls width="100%"></video>

The output includes recognized fencing actions, movement states, right-of-way transitions, scoring events, and the running score.

## The Problem

Foil fencing presents a challenging problem for automated sports analysis.

Determining the outcome of a fencing exchange is not based solely on detecting whether a point occurred. In foil, the referee must interpret the sequence of actions performed by both fencers and apply the rules of right-of-way to determine which fencer has priority for the touch.

This requires understanding multiple aspects of a bout simultaneously:

* Where the fencers and their equipment are located
* How the fencers are moving along the piste
* The positions and movements of their bodies
* What fencing actions are being performed
* Whether a scoring event occurred
* How the sequence of actions determines right-of-way

Fencelytics explores whether these observations can be extracted from video and combined into an automated analysis pipeline.
