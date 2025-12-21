import json
import os
import sys

"""
Quick script to check if your annotation JSON files are valid
and in the correct format for training data preparation.
"""

def check_json_file(filepath):
    """Check a single JSON annotation file"""
    print(f"\n{'='*60}")
    print(f"Checking: {filepath}")
    print('='*60)
    
    # Check if file exists
    if not os.path.exists(filepath):
        print(f"❌ ERROR: File does not exist!")
        print(f"   Looked at: {os.path.abspath(filepath)}")
        return False
    
    # Try to load JSON
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        print("✓ JSON file loaded successfully")
    except json.JSONDecodeError as e:
        print(f"❌ ERROR: Invalid JSON format - {e}")
        return False
    except Exception as e:
        print(f"❌ ERROR: Could not read file - {e}")
        return False
    
    # Check required fields
    required_fields = ['video_name', 'fencer_position', 'frames']
    for field in required_fields:
        if field in data:
            print(f"✓ Found required field: '{field}'")
        else:
            print(f"❌ Missing required field: '{field}'")
            return False
    
    # Check frames structure
    frames = data.get('frames', [])
    print(f"\nFrames analysis:")
    print(f"  - Total frames: {len(frames)}")
    
    if len(frames) == 0:
        print("❌ ERROR: No frames found in the file!")
        return False
    
    # Check first and last frames
    first_frame = frames[0]
    last_frame = frames[-1]
    
    print(f"  - First frame index: {first_frame.get('frame_index', 'Missing')}")
    print(f"  - Last frame index: {last_frame.get('frame_index', 'Missing')}")
    
    # Check frame structure
    frame_issues = []
    for i, frame in enumerate(frames[:5]):  # Check first 5 frames
        issues = []
        if 'frame_index' not in frame:
            issues.append("missing frame_index")
        if 'actions' not in frame:
            issues.append("missing actions")
        else:
            if not isinstance(frame['actions'], list):
                issues.append(f"actions is not a list (got {type(frame['actions'])})")
        if 'keypoints' not in frame:
            issues.append("missing keypoints")
        else:
            kp = frame['keypoints']
            if not isinstance(kp, list):
                issues.append(f"keypoints is not a list (got {type(kp)})")
            elif len(kp) != 17:
                issues.append(f"wrong number of keypoints (got {len(kp)}, expected 17)")
            elif len(kp) > 0 and len(kp[0]) != 3:
                issues.append(f"keypoint format wrong (expected [x,y,conf])")
        
        if issues:
            frame_issues.append(f"Frame {i}: {', '.join(issues)}")
    
    if frame_issues:
        print("\n❌ Frame structure issues found:")
        for issue in frame_issues:
            print(f"   - {issue}")
        return False
    else:
        print("\n✓ Frame structure looks correct")
    
    # Check action values
    print(f"\nAction analysis:")
    action_counts = {}
    for frame in frames:
        for action in frame.get('actions', []):
            action_counts[action] = action_counts.get(action, 0) + 1
    
    print(f"  - Unique actions found: {list(action_counts.keys())}")
    print(f"  - Action distribution:")
    for action, count in sorted(action_counts.items()):
        print(f"    - Action {action}: {count} occurrences")
    
    return True

if __name__ == "__main__":
    # Get file paths from command line or use defaults
    if len(sys.argv) > 1:
        files_to_check = sys.argv[1:]
    else:
        # Default paths - update these to match yours
        files_to_check = [
            "/Users/aidenburagohain/Coding/FencingAI/data/pose_data/SageVNguyen-00.01.39.886-00.01.43.435-00.00.02.475-00.00.08.450/SageVNguyen-00.01.39.886-00.01.43.435-00.00.02.475-00.00.08.450_left_fencer.json",
            "/Users/aidenburagohain/Coding/FencingAI/data/pose_data/SageVNguyen-00.01.39.886-00.01.43.435-00.00.02.475-00.00.08.450/SageVNguyen-00.01.39.886-00.01.43.435-00.00.02.475-00.00.08.450_right_fencer.json"
        ]
        print("No files specified. Checking default paths:")
        print(f"  - {files_to_check[0]}")
        print(f"  - {files_to_check[1]}")
        print("\nUsage: python check_annotation_files.py path/to/left.json path/to/right.json")
    
    # Check each file
    all_valid = True
    for filepath in files_to_check:
        if not check_json_file(filepath):
            all_valid = False
    
    # Summary
    print(f"\n{'='*60}")
    if all_valid:
        print("✓ All files are valid and ready for training data preparation!")
    else:
        print("❌ Some issues were found. Please fix them before running prepare_training_data.py")
    print('='*60)