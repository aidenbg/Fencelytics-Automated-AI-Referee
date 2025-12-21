import numpy as np
import json
import os
from collections import Counter
import pickle

"""
FENCING TRAINING DATA CREATOR

This script converts annotated fencing data into LSTM-ready training sequences.
Features:
- Creates sliding windows of 8 frames
- Handles multi-label annotations
- Mirrors poses for data augmentation
- Provides detailed statistics
"""

# Configuration
SEQUENCE_LENGTH = 8   # Number of frames per sequence (reduced for more sequences)
NUM_KEYPOINTS = 17    # COCO pose format
KEYPOINT_DIM = 3      # x, y, confidence
NUM_CLASSES = 11      # Number of action classes

# Action mapping (must match annotation tool)
ACTION_TO_INDEX = {
    'advance': 0,
    'retreat': 1,
    'lunge': 2,
    'attack': 3,
    'remise': 4,
    'counterattack': 5,
    'parry': 6,
    'line': 7,
    'infighting': 8,
    'provoke': 9,
    'idle': 10
}

INDEX_TO_ACTION = {v: k for k, v in ACTION_TO_INDEX.items()}


def load_annotation_file(filepath):
    """Load a single annotation JSON file"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    print(f"Loaded {filepath}:")
    print(f"  - Video: {data.get('video_name', 'Unknown')}")
    print(f"  - Fencer: {data.get('fencer_position', 'Unknown')}")
    print(f"  - Total frames in file: {data.get('total_frames', 'Unknown')}")
    print(f"  - Number of frame entries: {len(data.get('frames', []))}")
    
    # Debug: Check structure of first frame if it exists
    if data.get('frames') and len(data['frames']) > 0:
        first_frame = data['frames'][0]
        print(f"  - First frame structure:")
        print(f"    - frame_index: {first_frame.get('frame_index', 'Missing')}")
        print(f"    - actions: {first_frame.get('actions', 'Missing')}")
        print(f"    - keypoints: {'Present' if 'keypoints' in first_frame else 'Missing'}")
        if 'keypoints' in first_frame:
            print(f"    - keypoints shape: {len(first_frame['keypoints'])} keypoints")
    
    return data


def extract_frame_data(annotation_data):
    """
    Extract frames into a list sorted by frame index.
    Handles missing frames by returning None for gaps.
    """
    frames_dict = {}
    
    # Build dictionary of frame_index -> frame_data
    for frame in annotation_data['frames']:
        frames_dict[frame['frame_index']] = frame
    
    # Get min and max frame indices
    if not frames_dict:
        print("  WARNING: No frames found in annotation data!")
        return [], None, None
    
    min_frame = min(frames_dict.keys())
    max_frame = max(frames_dict.keys())
    
    print(f"  Frame indices range from {min_frame} to {max_frame}")
    print(f"  Total annotated frames: {len(frames_dict)}")
    
    # Create ordered list, None for missing frames
    frames_list = []
    for i in range(min_frame, max_frame + 1):
        if i in frames_dict:
            frames_list.append(frames_dict[i])
        else:
            frames_list.append(None)
    
    return frames_list, min_frame, max_frame


def create_feature_vector(keypoints):
    """
    Convert keypoints to feature vector.
    Input: List of 17 keypoints, each [x, y, confidence]
    Output: Flattened array of shape (51,)
    """
    return np.array(keypoints).flatten()


def mirror_keypoints(keypoints):
    """
    Mirror keypoints horizontally (flip x-coordinates).
    Also swaps left/right body parts.
    
    COCO Keypoint Order (17 keypoints):
    0: nose
    1: left_eye
    2: right_eye
    3: left_ear
    4: right_ear
    5: left_shoulder
    6: right_shoulder
    7: left_elbow
    8: right_elbow
    9: left_wrist
    10: right_wrist
    11: left_hip
    12: right_hip
    13: left_knee
    14: right_knee
    15: left_ankle
    16: right_ankle
    """
    mirrored = np.array(keypoints)
    
    # Flip x coordinates (assuming normalized 0-1)
    mirrored[:, 0] = 1.0 - mirrored[:, 0]
    
    # COCO keypoint indices for swapping left/right
    swap_pairs = [
        (1, 2),   # left_eye <-> right_eye
        (3, 4),   # left_ear <-> right_ear
        (5, 6),   # left_shoulder <-> right_shoulder
        (7, 8),   # left_elbow <-> right_elbow
        (9, 10),  # left_wrist <-> right_wrist
        (11, 12), # left_hip <-> right_hip
        (13, 14), # left_knee <-> right_knee
        (15, 16)  # left_ankle <-> right_ankle
    ]
    
    # Swap left/right keypoints
    mirrored_copy = mirrored.copy()
    for left_idx, right_idx in swap_pairs:
        mirrored[left_idx] = mirrored_copy[right_idx]
        mirrored[right_idx] = mirrored_copy[left_idx]
    
    return mirrored


def mirror_action(action_idx):
    """
    Actions DO NOT change when mirroring!
    A fencer advancing is still advancing when mirrored.
    Only the body orientation changes, not the action itself.
    """
    # Simply return the same action
    return action_idx


def create_multi_label_vector(action_indices):
    """
    Create multi-label binary vector from action indices.
    E.g., [2, 6] -> [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]
    """
    label_vector = np.zeros(NUM_CLASSES, dtype=np.float32)
    for idx in action_indices:
        label_vector[idx] = 1.0
    return label_vector


def create_sequences(frames_list, sequence_length=8, 
                    use_multi_label=True, create_mirrored=True):
    """
    Create training sequences from frame list.
    
    For each frame N, uses frames [max(0, N-7), ..., N] as input.
    Early frames are padded with zeros to maintain consistent sequence length.
    
    Example:
    - Frame 0: [pad, pad, pad, pad, pad, pad, pad, frame0]
    - Frame 1: [pad, pad, pad, pad, pad, pad, frame0, frame1]
    - Frame 7: [frame0, frame1, frame2, frame3, frame4, frame5, frame6, frame7]
    - Frame 8: [frame1, frame2, frame3, frame4, frame5, frame6, frame7, frame8]
    
    Args:
        frames_list: List of frame dictionaries
        sequence_length: Number of frames per sequence (default: 8)
        use_multi_label: If True, create multi-label vectors
        create_mirrored: If True, also create mirrored sequences
    
    Returns:
        sequences: Array of shape (n_sequences, sequence_length, 51)
        labels: Array of shape (n_sequences, num_classes) or (n_sequences,)
        sequence_info: List of dictionaries with sequence metadata
    """
    sequences = []
    labels = []
    sequence_info = []
    
    # Create padding frame (all zeros)
    padding_features = np.zeros(NUM_KEYPOINTS * KEYPOINT_DIM)
    
    # Process each frame
    for current_idx in range(len(frames_list)):
        # Skip if current frame is None (gap in annotation)
        if frames_list[current_idx] is None:
            continue
        
        # Determine start index for this sequence
        start_idx = max(0, current_idx - sequence_length + 1)
        
        # Build sequence with padding if necessary
        sequence_features = []
        valid_frames = []
        
        # Add padding for early frames
        padding_needed = sequence_length - (current_idx - start_idx + 1)
        for _ in range(padding_needed):
            sequence_features.append(padding_features)
        
        # Add actual frames
        skip_sequence = False
        for frame_idx in range(start_idx, current_idx + 1):
            if frames_list[frame_idx] is None:
                # Gap in annotation within sequence - skip this sequence
                skip_sequence = True
                break
            features = create_feature_vector(frames_list[frame_idx]['keypoints'])
            sequence_features.append(features)
            valid_frames.append(frame_idx)
        
        if skip_sequence:
            continue
        
        sequence_features = np.array(sequence_features)
        
        # Get label from current frame (last frame in sequence)
        current_frame = frames_list[current_idx]
        action_indices = current_frame['actions']
        
        if use_multi_label:
            label = create_multi_label_vector(action_indices)
        else:
            label = action_indices[0] if action_indices else ACTION_TO_INDEX['idle']
        
        # Add original sequence
        sequences.append(sequence_features)
        labels.append(label)
        sequence_info.append({
            'current_frame': current_idx,
            'sequence_frames': valid_frames,
            'padded_frames': padding_needed,
            'mirrored': False,
            'actions': [INDEX_TO_ACTION[idx] for idx in action_indices]
        })
        
        # Create mirrored version if requested
        if create_mirrored:
            mirrored_sequence = []
            
            # Mirror padding frames (still zeros)
            for _ in range(padding_needed):
                mirrored_sequence.append(padding_features)
            
            # Mirror actual frames
            for frame_idx in range(start_idx, current_idx + 1):
                frame = frames_list[frame_idx]
                keypoints_array = np.array(frame['keypoints'])
                mirrored_kpts = mirror_keypoints(keypoints_array)
                mirrored_features = create_feature_vector(mirrored_kpts)
                mirrored_sequence.append(mirrored_features)
            
            mirrored_sequence = np.array(mirrored_sequence)
            
            # Actions stay the same when mirroring!
            sequences.append(mirrored_sequence)
            labels.append(label)  # Same label - actions don't change
            sequence_info.append({
                'current_frame': current_idx,
                'sequence_frames': valid_frames,
                'padded_frames': padding_needed,
                'mirrored': True,
                'actions': [INDEX_TO_ACTION[idx] for idx in action_indices]
            })
    
    return np.array(sequences), np.array(labels), sequence_info


def analyze_sequences(labels, sequence_info, use_multi_label=True):
    """Print statistics about the created sequences"""
    print("\n" + "="*60)
    print("SEQUENCE STATISTICS")
    print("="*60)
    
    total_sequences = len(labels)
    original_sequences = sum(1 for info in sequence_info if not info['mirrored'])
    mirrored_sequences = sum(1 for info in sequence_info if info['mirrored'])
    
    print(f"Total sequences: {total_sequences}")
    print(f"  - Original: {original_sequences}")
    print(f"  - Mirrored: {mirrored_sequences}")
    
    if use_multi_label:
        # Count action occurrences in multi-label setting
        action_counts = Counter()
        for label_vector in labels:
            active_actions = np.where(label_vector == 1)[0]
            for action_idx in active_actions:
                action_counts[INDEX_TO_ACTION[action_idx]] += 1
        
        # Count single vs multi-action frames
        single_action = sum(1 for label in labels if np.sum(label) == 1)
        multi_action = sum(1 for label in labels if np.sum(label) > 1)
        
        print(f"\nLabel distribution:")
        print(f"  - Single action sequences: {single_action} ({single_action/total_sequences*100:.1f}%)")
        print(f"  - Multi-action sequences: {multi_action} ({multi_action/total_sequences*100:.1f}%)")
    else:
        # Count action occurrences in single-label setting
        action_counts = Counter(INDEX_TO_ACTION[label] for label in labels)
    
    print(f"\nAction distribution:")
    for action, count in sorted(action_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = count / total_sequences * 100
        print(f"  - {action:15s}: {count:4d} ({percentage:5.1f}%)")
    
    # Analyze sequence coverage - Updated for new structure
    frame_indices = set()
    padded_count = 0
    
    for info in sequence_info:
        if not info['mirrored']:
            # Add all frames that were used in this sequence
            for frame_idx in info['sequence_frames']:
                frame_indices.add(frame_idx)
            # Count sequences with padding
            if info['padded_frames'] > 0:
                padded_count += 1
    
    print(f"\nFrame coverage:")
    print(f"  - Unique frames used: {len(frame_indices)}")
    if frame_indices:
        print(f"  - Frame range: {min(frame_indices)} to {max(frame_indices)}")
    
    # Additional statistics for padded sequences
    print(f"\nPadding statistics (original sequences only):")
    print(f"  - Sequences with padding: {padded_count} ({padded_count/original_sequences*100:.1f}%)")
    print(f"  - Fully populated sequences: {original_sequences - padded_count} ({(original_sequences - padded_count)/original_sequences*100:.1f}%)")
    
    # Show example of padding distribution
    padding_distribution = Counter()
    for info in sequence_info:
        if not info['mirrored']:
            padding_distribution[info['padded_frames']] += 1
    
    if padding_distribution:
        print(f"\nPadding distribution:")
        for pad_count in sorted(padding_distribution.keys()):
            count = padding_distribution[pad_count]
            print(f"  - {pad_count} padded frames: {count} sequences ({count/original_sequences*100:.1f}%)")


def save_training_data(sequences, labels, sequence_info, output_dir, prefix="fencing"):
    """Save training data in multiple formats"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as numpy arrays
    np.save(os.path.join(output_dir, f"{prefix}_sequences.npy"), sequences)
    np.save(os.path.join(output_dir, f"{prefix}_labels.npy"), labels)
    
    # Save sequence info as JSON
    with open(os.path.join(output_dir, f"{prefix}_sequence_info.json"), 'w') as f:
        json.dump(sequence_info, f, indent=2)
    
    # Save as pickle for easy loading
    with open(os.path.join(output_dir, f"{prefix}_data.pkl"), 'wb') as f:
        pickle.dump({
            'sequences': sequences,
            'labels': labels,
            'sequence_info': sequence_info,
            'sequence_length': SEQUENCE_LENGTH,
            'num_keypoints': NUM_KEYPOINTS,
            'num_classes': NUM_CLASSES,
            'action_mapping': ACTION_TO_INDEX
        }, f)
    
    print(f"\nData saved to {output_dir}/")
    print(f"  - {prefix}_sequences.npy: {sequences.shape}")
    print(f"  - {prefix}_labels.npy: {labels.shape}")
    print(f"  - {prefix}_sequence_info.json")
    print(f"  - {prefix}_data.pkl (complete dataset)")


def process_fencer_annotations(left_annotation_path, right_annotation_path=None,
                              output_dir="./training_data", use_multi_label=True):
    """
    Process annotations for one or both fencers.
    
    Args:
        left_annotation_path: Path to left fencer JSON
        right_annotation_path: Path to right fencer JSON (optional)
        output_dir: Directory to save training data
        use_multi_label: Whether to use multi-label classification
    """
    all_sequences = []
    all_labels = []
    all_sequence_info = []
    
    # Debug info
    print(f"\nprocess_fencer_annotations called with:")
    print(f"  left_annotation_path: {left_annotation_path}")
    print(f"  right_annotation_path: {right_annotation_path}")
    
    # Process left fencer
    if left_annotation_path and os.path.exists(left_annotation_path):
        print("\nProcessing LEFT fencer...")
        left_data = load_annotation_file(left_annotation_path)
        frames_list, min_frame, max_frame = extract_frame_data(left_data)
        
        if not frames_list:
            print("  WARNING: No frames extracted from left fencer data!")
        else:
            print(f"  Extracted {len(frames_list)} frames (indices {min_frame} to {max_frame})")
            
            sequences, labels, seq_info = create_sequences(
                frames_list, 
                SEQUENCE_LENGTH, 
                use_multi_label=use_multi_label,
                create_mirrored=True
            )
            
            # Add fencer position to sequence info
            for info in seq_info:
                info['fencer'] = 'left'
            
            all_sequences.append(sequences)
            all_labels.append(labels)
            all_sequence_info.extend(seq_info)
            
            print(f"Created {len(sequences)} sequences from left fencer")
            print(f"  (Using {SEQUENCE_LENGTH}-frame sliding windows)")
    else:
        print(f"\nSkipping LEFT fencer - file not found at: {left_annotation_path}")
    
    # Process right fencer
    if right_annotation_path and os.path.exists(right_annotation_path):
        print("\nProcessing RIGHT fencer...")
        right_data = load_annotation_file(right_annotation_path)
        frames_list, min_frame, max_frame = extract_frame_data(right_data)
        
        if not frames_list:
            print("  WARNING: No frames extracted from right fencer data!")
        else:
            print(f"  Extracted {len(frames_list)} frames (indices {min_frame} to {max_frame})")
            
            sequences, labels, seq_info = create_sequences(
                frames_list, 
                SEQUENCE_LENGTH, 
                use_multi_label=use_multi_label,
                create_mirrored=True
            )
            
            # Add fencer position to sequence info
            for info in seq_info:
                info['fencer'] = 'right'
            
            all_sequences.append(sequences)
            all_labels.append(labels)
            all_sequence_info.extend(seq_info)
            
            print(f"Created {len(sequences)} sequences from right fencer")
            print(f"  (Using {SEQUENCE_LENGTH}-frame sliding windows)")
    else:
        print(f"\nSkipping RIGHT fencer - file not found at: {right_annotation_path}")
    
    # Combine all sequences
    if all_sequences:
        combined_sequences = np.vstack(all_sequences)
        combined_labels = np.vstack(all_labels) if use_multi_label else np.hstack(all_labels)
        
        # Analyze combined data
        analyze_sequences(combined_labels, all_sequence_info, use_multi_label)
        
        # Save combined data
        save_training_data(combined_sequences, combined_labels, all_sequence_info, 
                          output_dir, prefix="combined")
        
        return combined_sequences, combined_labels, all_sequence_info
    else:
        print("\nNo valid annotation files found!")
        print("Please check:")
        print("1. The file paths are correct")
        print("2. The JSON files exist in the specified locations")
        print("3. The JSON files contain valid annotation data")
        return None, None, None


# Example usage
if __name__ == "__main__":
    # Update these paths to your annotation files
    left_json = "/Users/aidenburagohain/Coding/FencingAI/data/pose_data/SageVNguyen-00.01.39.886-00.01.43.435-00.00.02.475-00.00.08.450/SageVNguyen-00.01.39.886-00.01.43.435-00.00.02.475-00.00.08.450_left_fencer.json"
    right_json = "/Users/aidenburagohain/Coding/FencingAI/data/pose_data/SageVNguyen-00.01.39.886-00.01.43.435-00.00.02.475-00.00.08.450/SageVNguyen-00.01.39.886-00.01.43.435-00.00.02.475-00.00.08.450_right_fencer.json"
    output_directory = "/Users/aidenburagohain/Coding/FencingAI/data/pose_data/training_data"
    
    # Debug: Print paths to verify they're correct
    print(f"Looking for left fencer data at: {left_json}")
    print(f"Left file exists: {os.path.exists(left_json)}")
    print(f"Looking for right fencer data at: {right_json}")
    print(f"Right file exists: {os.path.exists(right_json)}")
    
    # Process annotations and create training data
    # Set use_multi_label=False if you want single-label classification
    sequences, labels, info = process_fencer_annotations(
        left_json, 
        right_json,
        output_directory,
        use_multi_label=True  # Set to False for single-label
    )
    
    if sequences is not None:
        print(f"\nFinal training data shape:")
        print(f"  - Sequences: {sequences.shape}")
        print(f"  - Labels: {labels.shape}")
        print(f"\nReady for LSTM training!")
        
        # Example of how to load the saved data
        print("\nTo load the data later:")
        print("```python")
        print("import pickle")
        print(f"with open('{os.path.join(output_directory, 'combined_data.pkl')}', 'rb') as f:")
        print("    data = pickle.load(f)")
        print("sequences = data['sequences']")
        print("labels = data['labels']")
        print("```")
    
    if sequences is not None:
        print(f"\nFinal training data shape:")
        print(f"  - Sequences: {sequences.shape}")
        print(f"  - Labels: {labels.shape}")
        print(f"\nReady for LSTM training!")
        
        # Example of how to load the saved data
        print("\nTo load the data later:")
        print("```python")
        print("import pickle")
        print("with open('./training_data/combined_data.pkl', 'rb') as f:")
        print("    data = pickle.load(f)")
        print("sequences = data['sequences']")
        print("labels = data['labels']")
        print("```")
