# Motion Tracking

Fencelytics tracks fencer movement along the piste using pose-estimated body landmarks. The **hip key point** was selected as the primary reference point for movement because it provides a relatively stable representation of the fencer's overall position while remaining useful for tracking movement along the piste.

## The Camera-Motion Problem

Fencing footage often includes camera panning or other camera movement. This creates a problem when measuring fencer movement directly from video coordinates.

Camera motion can:

* Create the appearance of fencer movement when the fencer is stationary.
* Cause apparent movement in the opposite direction when the camera follows a fencer.
* Alter the measured displacement of a fencer even when their actual movement has not changed.

Without correcting for camera motion, these effects can distort velocity and acceleration measurements and potentially cause the action-recognition system to misinterpret a fencer's movement.

## ORB Camera-Motion Compensation

Fencelytics uses **ORB (Oriented FAST and Rotated BRIEF) feature registration** to estimate camera motion between frames.

Background features are matched between consecutive frames while the detected fencers are excluded from the camera-motion estimate. The median displacement of the background features is used as an estimate of camera movement.

The estimated camera displacement is then projected onto the previously established piste axis and subtracted from the observed fencer displacement.

This produces a camera-compensated movement signal that more closely represents the fencer's actual movement along the piste.

## Example

https://github.com/user-attachments/assets/7150f8c4-c487-4294-a866-4f0b725ff914


## Output

The example in this directory demonstrates the effect of camera-motion compensation by plotting movement for both fencers:

Before compensation — movement measured directly from the video:
<img width="1800" height="750" alt="before_vid_2" src="https://github.com/user-attachments/assets/4dc5248a-b61f-489a-bfa0-2f8174c0697a" />


After compensation** — movement after estimated camera motion has been removed:
<img width="1800" height="750" alt="after_vid_2" src="https://github.com/user-attachments/assets/36f3d68c-2289-427d-b722-3456aeb3a6ec" />


These signals are subsequently used to derive movement features such as velocity and acceleration for downstream action recognition.

## Limitations

ORB registration depends on having sufficient visible background features that remain consistent between frames. Heavy occlusion, rapid camera movement, motion blur, or limited background texture can make camera-motion estimation less reliable.

Camera compensation therefore reduces an important source of measurement error but does not completely eliminate all sources of tracking noise.
