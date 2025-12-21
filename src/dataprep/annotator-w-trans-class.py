import cv2 as cv
from ultralytics import YOLO
import os
import numpy as np
import json
from datetime import datetime

"""
FENCING ACTION ANNOTATION TOOL

This tool helps annotate fencing videos frame-by-frame with action labels for training
action recognition models. It uses YOLO pose detection to extract keypoints and allows
manual labeling of up to 2 simultaneous actions per fencer.

DETAILED ACTION DESCRIPTIONS FOR ANNOTATION:

1. ADVANCE (key: 1)
   - Front foot moves forward first, then back foot follows
   - Maintains en garde stance throughout
   - Distance between feet remains constant
   - Can be small (step) or large (cross-step) advances
   - Upper body remains relatively stable

2. RETREAT (key: 2)
   - Back foot moves backward first, then front foot follows
   - Mirror opposite of advance
   - Used to maintain distance or escape attacks
   - Should maintain balance and readiness to defend

3. LUNGE (key: 3)
   - Explosive forward movement with full arm extension
   - Back leg straightens completely, pushing body forward
   - Front knee bends deeply (90 degrees or more)
   - Arm extends before body movement (in foil/epee)
   - Recovery involves pulling back to en garde

4. ATTACK (key: 4)
   - Any offensive action with the blade intended to score
   - Includes: straight thrust, disengage, coupe, flick
   - Can be simple (one movement) or compound (multiple)
   - Distinguished by offensive intent and priority
   - May or may not include footwork

5. REMISE (key: 5)
   - Immediate replacement of point after being parried
   - Arm stays extended, no withdrawal
   - Often involves small wrist or finger movements
   - Must be immediate - any pause makes it a new attack
   - Common after opponent's failed riposte

6. COUNTERATTACK (key: 6)
   - Offensive action during opponent's attack
   - Includes stop-hits (into preparation) and time-hits
   - Requires precise timing and distance judgment
   - In foil: must land before attacker's final movement
   - In epee: simultaneous hits both score

7. PARRY (key: 7)
   - Defensive blade movement to deflect attacks
   - Types: lateral (4,6), vertical (7,8), circular, semi-circular
   - Can be simple (direct) or compound (multiple)
   - Opposition parries maintain blade contact
   - Beat parries strike opponent's blade away

8. LINE (key: 8)
   - Arm extended with point threatening target
   - Establishes right-of-way in foil
   - Forces opponent to deal with blade before attacking
   - Can be maintained while moving (advance/retreat in line)
   - Different from attack - more defensive/preparatory

9. INFIGHTING (key: 9)
   - Actions when fencers are too close for normal technique
   - May include: pommel work, blade wrestling, pushing
   - Often results in halt for corps-a-corps
   - Common after failed attacks or simultaneous actions
   - Requires different technique than normal distance

10. PROVOKE (key: p)
    - Actions designed to elicit specific responses
    - Includes: false attacks (feints), blade beats, invitations
    - Shows opening to encourage opponent's attack
    - Preparatory actions that aren't full attacks
    - Essential for setting up second-intention actions

11. IDLE (key: 0)
    - En garde position with minimal movement
    - Small adjustments for balance or comfort
    - Waiting phases between phrases
    - No tactical intent or significant action
    - Blade may have small movements but no threats

12. TRANSITION (key: t)
    - Movement between two distinct actions
    - Weight shifting between positions
    - Arm moving between lines
    - Recovery phases after actions
    - Preparation phases before attacks
    - Duration: typically 3-15 frames
    - Helps model understand action flow

ANNOTATION BEST PRACTICES WITH TRANSITIONS:

1. When to use TRANSITION:
   - Between advance and lunge (weight shifting forward)
   - After lunge during recovery to en garde
   - Between parry and riposte (blade clearing opponent's)
   - During direction changes (advance to retreat)
   - Complex preparations before attacks

2. When NOT to use TRANSITION:
   - Quick, fluid actions (use the dominant action)
   - Very short movements (<3 frames)
   - Within a continuous action (multiple advances)
   - Small blade adjustments (use idle instead)

3. Common transition patterns to watch for:
   - idle → transition → attack
   - advance → transition → lunge
   - parry → transition → riposte
   - lunge → transition → idle (recovery)
   - line → transition → attack

4. For ambiguous frames:
   - If mostly one action with elements of another: use dominant action
   - If equally between two actions: use transition
   - If preparing an action but not yet committed: use transition
   - If recovering from an action: use transition
"""

# Configuration variables
input_video_path = "path/to/your/fencing_video.mp4"  # Change to your video
output_dir = "./output"  # Change to your output directory

# See detailed action descriptions in the docstring above for annotation guidelines

# Action mapping with detailed descriptions
ACTIONS = {
    '1': 'advance',      # Moving forward toward opponent, front foot leads
    '2': 'retreat',      # Moving backward away from opponent, back foot leads
    '3': 'lunge',        # Explosive forward thrust with arm extended, back leg straight
    '4': 'attack',       # Any offensive blade action (thrust, cut, flick)
    '5': 'remise',       # Immediate renewal of attack without withdrawing arm
    '6': 'counterattack', # Attack into opponent's attack (stop-hit, time-hit)
    '7': 'parry',        # Defensive blade movement to block opponent's attack
    '8': 'line',         # Arm extended with point threatening opponent
    '9': 'infighting',   # Close-quarter combat, corps-a-corps situations
    'p': 'provoke',      # Actions to elicit response (feints, blade beats, invitations)
    '0': 'idle',         # En garde position, no significant action
    't': 'transition'    # Moving between actions, preparation phases
}

# Create action to index mapping with descriptions
ACTION_TO_INDEX = {
    'advance': 0,        # Forward footwork maintaining en garde stance
    'retreat': 1,        # Backward footwork maintaining en garde stance  
    'lunge': 2,          # Full extension attack - arm straight, back leg pushes
    'attack': 3,         # Offensive blade work - includes all types of attacks
    'remise': 4,         # Continuation of attack after being parried
    'counterattack': 5,  # Attacking into opponent's preparation or attack
    'parry': 6,          # Blade defense - lateral, circular, or semi-circular
    'line': 7,           # Point control - maintaining threat with extended arm
    'infighting': 8,     # Actions when fencers are very close together
    'provoke': 9,        # Preparatory actions to create openings
    'idle': 10,          # Waiting in guard position, small adjustments only
    'transition': 11     # Between distinct actions, weight shifts, preparations
}

# COCO skeleton connections (from your original code)
SKELETON = [
    [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],
    [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
    [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
    [2, 4], [3, 5], [4, 6], [5, 7]
]

def annotate_video(input_video_path, output_dir):
    """
    Main annotation function with checkpoint support
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get video name for file naming
    video_name = os.path.splitext(os.path.basename(input_video_path))[0]
    
    # Define all file paths in the output directory
    checkpoint_path = os.path.join(output_dir, f"{video_name}_checkpoint.json")
    left_output_path = os.path.join(output_dir, f"{video_name}_left_fencer.json")
    right_output_path = os.path.join(output_dir, f"{video_name}_right_fencer.json")
    
    # Load checkpoint if exists
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        with open(checkpoint_path, 'r') as f:
            checkpoint = json.load(f)
        start_frame = checkpoint['last_frame']
        left_annotations = checkpoint['left_annotations']
        right_annotations = checkpoint['right_annotations']
        # Try to load previous keypoints from checkpoint
        previous_keypoints = checkpoint.get('previous_keypoints', None)
        print(f"Resuming from frame {start_frame + 1}")
    else:
        start_frame = 0
        left_annotations = {}
        right_annotations = {}
        previous_keypoints = None
    
    # Load YOLO pose model
    print("Loading pose model...")
    model = YOLO("yolo11m-pose.pt")
    
    # Open video
    cap = cv.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise IOError(f"Failed to open video: {input_video_path}")
    
    # Video properties
    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv.CAP_PROP_FPS)
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    
    print(f"\nVideo: {video_name}")
    print(f"Resolution: {frame_width}x{frame_height} @ {fps} fps")
    print(f"Total frames: {total_frames}")
    print(f"\nOutput directory: {output_dir}")
    print(f"Checkpoint file: {checkpoint_path}")
    print(f"Output files will be:")
    print(f"  - {left_output_path}")
    print(f"  - {right_output_path}")
    print_controls()
    
    # Jump to start frame
    cap.set(cv.CAP_PROP_POS_FRAMES, start_frame)
    current_frame_idx = start_frame
    
    # UI state
    selected_fencer = None
    current_frame = None
    current_keypoints = None
    
    while current_frame_idx < total_frames:
        # Read and process frame
        if current_frame is None:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run pose detection
            results = model(frame)
            
            # Extract keypoints for top 2 people with tracking
            current_keypoints = extract_and_track_keypoints(results, frame_width, frame_height, previous_keypoints)
            current_frame = frame.copy()
            
            # Update previous keypoints for next frame
            previous_keypoints = current_keypoints
        
        # Get current annotations - NOW HANDLES MULTIPLE ACTIONS
        frame_key = str(current_frame_idx)
        left_actions = left_annotations.get(frame_key, {}).get('actions', [])
        right_actions = right_annotations.get(frame_key, {}).get('actions', [])
        
        # Draw annotated frame
        display_frame = draw_annotated_frame(
            current_frame,
            current_keypoints,
            current_frame_idx,
            total_frames,
            left_actions,
            right_actions,
            selected_fencer,
            fps
        )
        
        cv.imshow("Fencing Action Annotation Tool", display_frame)
        
        # Handle keyboard input - always wait for user input
        key = cv.waitKey(0) & 0xFF
        
        # Handle arrow keys (some systems use extended codes)
        if key == 81 or key == 2:  # Left arrow key
            if current_frame_idx > 0:
                current_frame_idx -= 1
                cap.set(cv.CAP_PROP_POS_FRAMES, current_frame_idx)
                current_frame = None
                # Reset tracking when going backwards
                previous_keypoints = None
                
        elif key == 83 or key == 3:  # Right arrow key
            if current_frame_idx < total_frames - 1:
                current_frame_idx += 1
                current_frame = None
                
        elif key == ord('q'):  # Quit
            break
            
        elif key == ord('s'):  # Save checkpoint
            save_checkpoint(checkpoint_path, current_frame_idx, left_annotations, 
                          right_annotations, previous_keypoints)
            print(f"\nCheckpoint saved at frame {current_frame_idx + 1}")
            print(f"Saved to: {checkpoint_path}")
            
        elif key == ord('l'):  # Select left fencer
            selected_fencer = 'left'
            print(f"Selected LEFT fencer")
            
        elif key == ord('r'):  # Select right fencer  
            selected_fencer = 'right'
            print(f"Selected RIGHT fencer")
            
        elif key == ord('c'):  # Clear actions for selected fencer
            if selected_fencer:
                frame_key = str(current_frame_idx)
                if selected_fencer == 'left':
                    if frame_key in left_annotations:
                        left_annotations[frame_key]['actions'] = []
                        print(f"Cleared LEFT fencer actions at frame {current_frame_idx + 1}")
                elif selected_fencer == 'right':
                    if frame_key in right_annotations:
                        right_annotations[frame_key]['actions'] = []
                        print(f"Cleared RIGHT fencer actions at frame {current_frame_idx + 1}")
            
        elif (chr(key) in ACTIONS or key == ord('p') or key == ord('t')) and selected_fencer and current_keypoints:
            # Assign action to selected fencer - NOW ADDS TO LIST
            action_key = chr(key) if chr(key) in ACTIONS else 'p' if key == ord('p') else 't' if key == ord('t') else None
            if action_key:
                action = ACTIONS[action_key]
                frame_key = str(current_frame_idx)
                
                if selected_fencer == 'left' and len(current_keypoints) > 0 and current_keypoints[0] is not None:
                    if frame_key not in left_annotations:
                        left_annotations[frame_key] = {
                            'actions': [],
                            'keypoints': current_keypoints[0],
                            'timestamp': current_frame_idx / fps
                        }
                    
                    # Add action if not already present and less than 2 actions
                    if action not in left_annotations[frame_key]['actions'] and len(left_annotations[frame_key]['actions']) < 2:
                        left_annotations[frame_key]['actions'].append(action)
                        left_annotations[frame_key]['keypoints'] = current_keypoints[0]
                        left_annotations[frame_key]['timestamp'] = current_frame_idx / fps
                        print(f"Frame {current_frame_idx + 1} - Left: {left_annotations[frame_key]['actions']}")
                    elif action in left_annotations[frame_key]['actions']:
                        print(f"Action '{action}' already assigned to LEFT fencer at frame {current_frame_idx + 1}")
                    else:
                        print(f"LEFT fencer already has 2 actions at frame {current_frame_idx + 1}")
                    
                elif selected_fencer == 'right' and len(current_keypoints) > 1 and current_keypoints[1] is not None:
                    if frame_key not in right_annotations:
                        right_annotations[frame_key] = {
                            'actions': [],
                            'keypoints': current_keypoints[1],
                            'timestamp': current_frame_idx / fps
                        }
                    
                    # Add action if not already present and less than 2 actions
                    if action not in right_annotations[frame_key]['actions'] and len(right_annotations[frame_key]['actions']) < 2:
                        right_annotations[frame_key]['actions'].append(action)
                        right_annotations[frame_key]['keypoints'] = current_keypoints[1]
                        right_annotations[frame_key]['timestamp'] = current_frame_idx / fps
                        print(f"Frame {current_frame_idx + 1} - Right: {right_annotations[frame_key]['actions']}")
                    elif action in right_annotations[frame_key]['actions']:
                        print(f"Action '{action}' already assigned to RIGHT fencer at frame {current_frame_idx + 1}")
                    else:
                        print(f"RIGHT fencer already has 2 actions at frame {current_frame_idx + 1}")
    
    # Save final annotations
    print("\nSaving final annotations...")
    save_final_annotations(video_name, left_annotations, right_annotations, 
                          fps, frame_width, frame_height, 
                          left_output_path, right_output_path)
    
    # Clean up
    cap.release()
    cv.destroyAllWindows()
    
    # Remove checkpoint file after successful completion
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print(f"Removed checkpoint file: {checkpoint_path}")
    
    print("\nAnnotation complete!")
    print(f"All data saved to: {output_dir}")

def calculate_keypoint_center(keypoints):
    """Calculate the center position of visible keypoints"""
    visible_keypoints = [kp for kp in keypoints if kp[2] > 0.5]
    if not visible_keypoints:
        return None
    center_x = np.mean([kp[0] for kp in visible_keypoints])
    center_y = np.mean([kp[1] for kp in visible_keypoints])
    return np.array([center_x, center_y])

def match_keypoints_to_previous(current_detections, previous_keypoints):
    """
    Match current detections to previous frame's keypoints using distance-based tracking
    Returns ordered list maintaining left/right consistency
    """
    if previous_keypoints is None or len(previous_keypoints) == 0:
        # First frame or no previous data - sort by x position
        return sorted(current_detections, key=lambda x: calculate_keypoint_center(x)[0] if calculate_keypoint_center(x) is not None else float('inf'))
    
    # Initialize result with None values
    matched_keypoints = [None, None]
    used_indices = set()
    
    # Try to match each previous person to current detections
    for prev_idx in range(min(2, len(previous_keypoints))):
        if previous_keypoints[prev_idx] is None:
            continue
            
        prev_center = calculate_keypoint_center(previous_keypoints[prev_idx])
        if prev_center is None:
            continue
        
        best_match_idx = -1
        best_distance = float('inf')
        
        # Find closest match in current detections
        for curr_idx, curr_kpts in enumerate(current_detections):
            if curr_idx in used_indices:
                continue
                
            curr_center = calculate_keypoint_center(curr_kpts)
            if curr_center is None:
                continue
            
            # Calculate distance between centers
            distance = np.linalg.norm(prev_center - curr_center)
            
            # Only consider matches within reasonable distance (e.g., 30% of frame width)
            if distance < best_distance and distance < 0.3:
                best_distance = distance
                best_match_idx = curr_idx
        
        # Assign the best match
        if best_match_idx != -1:
            matched_keypoints[prev_idx] = current_detections[best_match_idx]
            used_indices.add(best_match_idx)
    
    # Handle any unmatched detections (new person entered frame)
    for curr_idx, curr_kpts in enumerate(current_detections):
        if curr_idx not in used_indices:
            # Find first empty slot
            for i in range(2):
                if matched_keypoints[i] is None:
                    matched_keypoints[i] = curr_kpts
                    break
    
    return matched_keypoints

def extract_and_track_keypoints(results, frame_width, frame_height, previous_keypoints):
    """
    Extract normalized keypoints for top 2 people and maintain consistent tracking
    """
    current_detections = []
    
    if results[0].keypoints is not None and results[0].keypoints.data.shape[0] > 0:
        keypoints_data = results[0].keypoints.data.cpu().numpy()  # Shape: (N, 17, 3)
        
        # Extract all valid detections
        for i in range(keypoints_data.shape[0]):
            visible_count = np.sum(keypoints_data[i, :, 2] > 0.5)
            
            # Only consider people with at least 5 visible keypoints
            if visible_count >= 5:
                # Normalize keypoints
                normalized_kpts = []
                for kpt in keypoints_data[i]:
                    normalized_kpts.append([
                        float(kpt[0] / frame_width),   # x normalized
                        float(kpt[1] / frame_height),  # y normalized  
                        float(kpt[2])                  # confidence
                    ])
                current_detections.append(normalized_kpts)
        
        # Sort by number of visible keypoints (best detections first)
        current_detections.sort(key=lambda x: sum(1 for kp in x if kp[2] > 0.5), reverse=True)
        
        # Keep only top 2 detections
        current_detections = current_detections[:2]
    
    # Match to previous frame to maintain consistent left/right assignment
    tracked_keypoints = match_keypoints_to_previous(current_detections, previous_keypoints)
    
    return tracked_keypoints

def draw_annotated_frame(frame, keypoints, frame_idx, total_frames, 
                        left_actions, right_actions, selected_fencer, fps):
    """
    Draw frame with pose overlay and UI elements - UPDATED FOR MULTIPLE ACTIONS
    """
    display = frame.copy()
    height, width = display.shape[:2]
    
    # Draw poses - BOTH BLUE SKELETONS WITH GREEN/WHITE KEYPOINTS
    if keypoints:
        skeleton_color = (255, 0, 0)  # Blue for both skeletons
        keypoint_color = (0, 255, 0)  # Green for keypoints
        
        for person_idx, person_kpts in enumerate(keypoints):
            if person_kpts is None:
                continue
                
            # Draw skeleton with blue color
            for connection in SKELETON:
                kpt1_idx, kpt2_idx = connection[0] - 1, connection[1] - 1
                
                if (kpt1_idx < len(person_kpts) and kpt2_idx < len(person_kpts) and
                    person_kpts[kpt1_idx][2] > 0.5 and person_kpts[kpt2_idx][2] > 0.5):
                    
                    pt1 = (int(person_kpts[kpt1_idx][0] * width), 
                          int(person_kpts[kpt1_idx][1] * height))
                    pt2 = (int(person_kpts[kpt2_idx][0] * width), 
                          int(person_kpts[kpt2_idx][1] * height))
                    
                    cv.line(display, pt1, pt2, skeleton_color, 2)
            
            # Draw keypoints with green color and white center
            for kpt in person_kpts:
                if kpt[2] > 0.5:
                    x = int(kpt[0] * width)
                    y = int(kpt[1] * height)
                    cv.circle(display, (x, y), 5, keypoint_color, -1)  # Green outer
                    cv.circle(display, (x, y), 3, (255, 255, 255), -1)  # White center
            
            # Add person label to distinguish left/right - NOW CONSISTENT WITH TRACKING
            center = calculate_keypoint_center(person_kpts)
            if center is not None:
                label_x = int(center[0] * width)
                label_y = max(20, int(center[1] * height) - 20)
                person_label = "LEFT" if person_idx == 0 else "RIGHT"
                
                # Add background for better visibility
                label_size = cv.getTextSize(person_label, cv.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv.rectangle(display, 
                           (label_x - 5, label_y - label_size[1] - 5),
                           (label_x + label_size[0] + 5, label_y + 5),
                           (0, 0, 0), -1)
                cv.putText(display, person_label, (label_x, label_y),
                          cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Draw UI overlay - INCREASED HEIGHT FOR MULTIPLE ACTIONS AND 3 ROWS
    overlay_height = 145
    cv.rectangle(display, (0, 0), (width, overlay_height), (0, 0, 0), -1)
    
    # Frame info - Display 1-based frame numbers
    time_str = f"{frame_idx/fps:.2f}s"
    cv.putText(display, f"Frame: {frame_idx + 1}/{total_frames} ({time_str})", 
              (10, 25), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Selection indicator
    if selected_fencer:
        cv.putText(display, f"Selected: {selected_fencer.upper()}", 
                  (10, 55), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Actions with color coding - NOW SHOWS MULTIPLE ACTIONS
    left_color = (0, 255, 255) if selected_fencer == 'left' else (200, 200, 200)
    right_color = (0, 255, 255) if selected_fencer == 'right' else (200, 200, 200)
    
    # Format actions as comma-separated list
    left_action_str = ", ".join(left_actions) if left_actions else "none"
    right_action_str = ", ".join(right_actions) if right_actions else "none"
    
    cv.putText(display, f"Left: {left_action_str}", 
              (400, 25), cv.FONT_HERSHEY_SIMPLEX, 0.7, left_color, 2)
    cv.putText(display, f"Right: {right_action_str}", 
              (400, 55), cv.FONT_HERSHEY_SIMPLEX, 0.7, right_color, 2)
    
    # Action list - display in three rows for 12 actions
    action_items = list(ACTIONS.items())
    row1_y = 85
    row2_y = 100
    row3_y = 115
    
    # First row (1-4)
    for i in range(4):
        if i < len(action_items):
            key, action = action_items[i]
            cv.putText(display, f"{key}:{action}", 
                      (10 + i*180, row1_y), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    # Second row (5-8)
    for i in range(4):
        idx = i + 4
        if idx < len(action_items):
            key, action = action_items[idx]
            cv.putText(display, f"{key}:{action}", 
                      (10 + i*180, row2_y), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    # Third row (9, p, 0, t)
    for i in range(4):
        idx = i + 8
        if idx < len(action_items):
            key, action = action_items[idx]
            cv.putText(display, f"{key}:{action}", 
                      (10 + i*180, row3_y), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    # Note about multiple actions
    cv.putText(display, "Max 2 actions per fencer | Tracking enabled", 
              (10, 135), cv.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    
    # Controls at bottom - UPDATED WITH CLEAR AND TRANSITION
    cv.putText(display, "L/R:Select | 0-9,P,T:Action | C:Clear | ←/→:Navigate | S:Save | Q:Quit", 
              (10, height - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return display

def save_checkpoint(checkpoint_path, current_frame, left_annotations, right_annotations, previous_keypoints):
    """Save current progress including tracking state"""
    checkpoint_data = {
        'last_frame': current_frame,
        'left_annotations': left_annotations,
        'right_annotations': right_annotations,
        'previous_keypoints': previous_keypoints,  # Save tracking state
        'timestamp': datetime.now().isoformat()
    }
    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)

def save_final_annotations(video_name, left_annotations, right_annotations, 
                          fps, width, height, left_output_path, right_output_path):
    """Save separate JSON files for left and right fencers - UPDATED FOR MULTIPLE ACTIONS"""
    
    # Prepare left fencer data
    left_data = {
        'video_name': video_name,
        'fencer_position': 'left',
        'fps': fps,
        'resolution': [width, height],
        'total_frames': len(left_annotations),
        'action_mapping': ACTION_TO_INDEX,
        'frames': []
    }
    
    # Prepare right fencer data  
    right_data = {
        'video_name': video_name,
        'fencer_position': 'right',
        'fps': fps,
        'resolution': [width, height],
        'total_frames': len(right_annotations),
        'action_mapping': ACTION_TO_INDEX,
        'frames': []
    }
    
    # Convert annotations to frame lists - NOW SAVES MULTIPLE ACTION INDICES
    for frame_idx in sorted(left_annotations.keys(), key=int):
        frame_data = left_annotations[frame_idx]
        # Convert action names to indices
        action_indices = [ACTION_TO_INDEX[action] for action in frame_data['actions']]
        left_data['frames'].append({
            'frame_index': int(frame_idx),
            'timestamp': frame_data['timestamp'],
            'actions': action_indices,  # Now a list of indices
            'keypoints': frame_data['keypoints']
        })
    
    for frame_idx in sorted(right_annotations.keys(), key=int):
        frame_data = right_annotations[frame_idx]
        # Convert action names to indices
        action_indices = [ACTION_TO_INDEX[action] for action in frame_data['actions']]
        right_data['frames'].append({
            'frame_index': int(frame_idx),
            'timestamp': frame_data['timestamp'],
            'actions': action_indices,  # Now a list of indices
            'keypoints': frame_data['keypoints']
        })
    
    # Save files
    with open(left_output_path, 'w') as f:
        json.dump(left_data, f, indent=2)
    
    with open(right_output_path, 'w') as f:
        json.dump(right_data, f, indent=2)
    
    print(f"\nSaved annotations:")
    print(f"  Left fencer: {left_output_path} ({len(left_data['frames'])} frames)")
    print(f"  Right fencer: {right_output_path} ({len(right_data['frames'])} frames)")

def print_controls():
    """Print control instructions - UPDATED"""
    print("\n" + "="*60)
    print("FENCING ACTION ANNOTATION TOOL - WITH TRACKING")
    print("="*60)
    print("\nCONTROLS:")
    print("  L         - Select LEFT fencer")
    print("  R         - Select RIGHT fencer")
    print("  0-9,P,T   - Add action to selected fencer (max 2)")
    print("  C         - Clear all actions for selected fencer")
    print("  ←         - Previous frame (Left arrow)")
    print("  →         - Next frame (Right arrow)")
    print("  S         - Save checkpoint")
    print("  Q         - Quit and save")
    print("\nACTIONS:")
    for key, action in ACTIONS.items():
        # Add spacing for better readability
        print(f"  {key} - {action.capitalize():15s}", end="")
        # Brief description for each action
        if action == 'advance':
            print("(Forward movement)")
        elif action == 'retreat':
            print("(Backward movement)")
        elif action == 'lunge':
            print("(Explosive attack)")
        elif action == 'attack':
            print("(Offensive blade action)")
        elif action == 'remise':
            print("(Renewed attack)")
        elif action == 'counterattack':
            print("(Attack during opponent's attack)")
        elif action == 'parry':
            print("(Defensive blade action)")
        elif action == 'line':
            print("(Extended arm threat)")
        elif action == 'infighting':
            print("(Close-quarter combat)")
        elif action == 'provoke':
            print("(Feints/preparations)")
        elif action == 'idle':
            print("(En garde/waiting)")
        elif action == 'transition':
            print("(Between actions)")
    print("\nNOTE: Each fencer can have up to 2 actions per frame")
    print("      Tracking maintains consistent left/right assignment")
    print("      Use TRANSITION for movements between distinct actions")
    print("="*60 + "\n")

if __name__ == "__main__":
    annotate_video(input_video_path, output_dir)