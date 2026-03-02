#!/usr/bin/env python3
"""
Convert hand_tracking data to 20D Wuji hand action.

This script reads hand_tracking_left/right from recorded JSON data,
converts them to 20D Wuji hand joint positions using the retargeting pipeline,
and saves the results as action_wuji_qpos_target_left/right.

Pipeline:
    hand_tracking_* (26D dict with 3D positions)
         hand_26d_to_mediapipe_21d()
    21x3 MediaPipe format
         apply_mediapipe_transformations()
    21x3 transformed coordinates
         WujiHandRetargeter.retarget()
    20D Wuji hand joint positions (5 fingers x 4 joints)

Usage:
    python convert_hand_tracking_to_wuji_action.py --input path/to/data.json --output path/to/output.json
    python convert_hand_tracking_to_wuji_action.py --input path/to/data.json  # overwrites input file
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

# Add paths for wuji_retargeting
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # TWIST2/
WUJI_RETARGETING_PATH = PROJECT_ROOT / "wuji_retargeting"
if str(WUJI_RETARGETING_PATH) not in sys.path:
    sys.path.insert(0, str(WUJI_RETARGETING_PATH))

try:
    from wuji_retargeting import WujiHandRetargeter
    from wuji_retargeting.mediapipe import apply_mediapipe_transformations
except ImportError as e:
    print(f"[ERROR] Error importing wuji_retargeting: {e}")
    print("   Please ensure wuji_retargeting is properly installed")
    sys.exit(1)


# ============================================================================
# Constants: 26D hand tracking joint names and mapping to 21D MediaPipe format
# ============================================================================

HAND_JOINT_NAMES_26 = [
    "Wrist", "Palm",
    "ThumbMetacarpal", "ThumbProximal", "ThumbDistal", "ThumbTip",
    "IndexMetacarpal", "IndexProximal", "IndexIntermediate", "IndexDistal", "IndexTip",
    "MiddleMetacarpal", "MiddleProximal", "MiddleIntermediate", "MiddleDistal", "MiddleTip",
    "RingMetacarpal", "RingProximal", "RingIntermediate", "RingDistal", "RingTip",
    "LittleMetacarpal", "LittleProximal", "LittleIntermediate", "LittleDistal", "LittleTip"
]

# 26D -> 21D MediaPipe mapping
# MediaPipe format: [Wrist, Thumb(4), Index(4), Middle(4), Ring(4), Pinky(4)]
# Skip: Palm, IndexDistal, MiddleDistal, RingDistal, LittleDistal
MEDIAPIPE_MAPPING_26_TO_21 = [
    1,   # 0: Wrist (actually Palm in 26D, used as Wrist in MediaPipe)
    2,   # 1: ThumbMetacarpal -> Thumb CMC
    3,   # 2: ThumbProximal -> Thumb MCP
    4,   # 3: ThumbDistal -> Thumb IP
    5,   # 4: ThumbTip -> Thumb Tip
    6,   # 5: IndexMetacarpal -> Index MCP
    7,   # 6: IndexProximal -> Index PIP
    8,   # 7: IndexIntermediate -> Index DIP
    10,  # 8: IndexTip -> Index Tip (skip IndexDistal at 9)
    11,  # 9: MiddleMetacarpal -> Middle MCP
    12,  # 10: MiddleProximal -> Middle PIP
    13,  # 11: MiddleIntermediate -> Middle DIP
    15,  # 12: MiddleTip -> Middle Tip (skip MiddleDistal at 14)
    16,  # 13: RingMetacarpal -> Ring MCP
    17,  # 14: RingProximal -> Ring PIP
    18,  # 15: RingIntermediate -> Ring DIP
    20,  # 16: RingTip -> Ring Tip (skip RingDistal at 19)
    21,  # 17: LittleMetacarpal -> Pinky MCP
    22,  # 18: LittleProximal -> Pinky PIP
    23,  # 19: LittleIntermediate -> Pinky DIP
    25,  # 20: LittleTip -> Pinky Tip (skip LittleDistal at 24)
]


# ============================================================================
# Conversion Functions
# ============================================================================

def hand_26d_to_mediapipe_21d(hand_data_dict: Dict[str, Any], hand_side: str = "left") -> np.ndarray:
    """
    Convert 26D hand tracking data (dict format) to 21D MediaPipe format.
    
    Args:
        hand_data_dict: Dict with joint names as keys, values are [[x,y,z], [qw,qx,qy,qz]]
        hand_side: "left" or "right"
        
    Returns:
        numpy array of shape (21, 3) - MediaPipe format hand landmarks
    """
    hand_side_prefix = "LeftHand" if hand_side.lower() == "left" else "RightHand"
    
    # Extract 26 joint positions
    joint_positions_26 = np.zeros((26, 3), dtype=np.float32)
    for i, joint_name in enumerate(HAND_JOINT_NAMES_26):
        key = hand_side_prefix + joint_name
        if key in hand_data_dict:
            pos = hand_data_dict[key][0]  # [x, y, z]
            joint_positions_26[i] = pos
        else:
            joint_positions_26[i] = [0.0, 0.0, 0.0]
    
    # Map to 21D using indices
    mediapipe_21d = joint_positions_26[MEDIAPIPE_MAPPING_26_TO_21]
    
    # Normalize: set wrist as origin
    wrist_pos = mediapipe_21d[0].copy()
    mediapipe_21d = mediapipe_21d - wrist_pos
    
    return mediapipe_21d


def convert_hand_tracking_to_wuji_action(
    hand_tracking_data: Dict[str, Any],
    hand_side: str,
    retargeter: WujiHandRetargeter
) -> Optional[np.ndarray]:
    """
    Convert hand tracking data to 20D Wuji hand action.
    
    Args:
        hand_tracking_data: Dict from hand_tracking_left/right, including is_active, timestamp, and joints
        hand_side: "left" or "right"
        retargeter: WujiHandRetargeter instance (should be pre-initialized)
        
    Returns:
        numpy array of shape (20,) - Wuji hand joint positions, or None if data is invalid
    """
    if hand_tracking_data is None:
        return None
    
    # Check if data is active
    if not hand_tracking_data.get("is_active", False):
        return None
    
    # Extract hand joint data (exclude metadata)
    hand_dict = {k: v for k, v in hand_tracking_data.items() 
                 if k not in ["is_active", "timestamp"]}
    
    if not hand_dict:
        return None
    
    try:
        # Step 1: 26D dict -> 21x3 MediaPipe format
        mediapipe_21d = hand_26d_to_mediapipe_21d(hand_dict, hand_side)
        
        # Step 2: Apply coordinate transformations
        mediapipe_transformed = apply_mediapipe_transformations(mediapipe_21d, hand_type=hand_side)
        
        # Step 3: Retarget to 20D
        result = retargeter.retarget(mediapipe_transformed)
        wuji_20d = result.robot_qpos  # shape: (20,)
        
        return wuji_20d
        
    except Exception as e:
        print(f"[WARN]  Error converting {hand_side} hand: {e}")
        return None


def process_json_file(
    input_path: str,
    output_path: Optional[str] = None,
    verbose: bool = True
) -> Tuple[int, int, int]:
    """
    Process a JSON file and convert hand_tracking_* to action_wuji_qpos_target_*.
    
    Args:
        input_path: Path to input JSON file
        output_path: Path to output JSON file (if None, overwrites input)
        verbose: Whether to print progress
        
    Returns:
        Tuple of (total_frames, left_converted, right_converted)
    """
    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path
    else:
        output_path = Path(output_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    if verbose:
        print(f" Loading: {input_path}")
    
    # Load JSON
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Check format
    if "data" not in data:
        raise ValueError("JSON file must contain 'data' key with list of frames")
    
    frames = data["data"]
    total_frames = len(frames)
    
    if verbose:
        print(f"[INFO] Total frames: {total_frames}")
        print(f" Initializing retargeters...")
    
    # Initialize retargeters (once for all frames)
    left_retargeter = WujiHandRetargeter(hand_side="left")
    right_retargeter = WujiHandRetargeter(hand_side="right")
    
    left_converted = 0
    right_converted = 0
    
    if verbose:
        print(f"[INFO] Converting hand tracking to Wuji actions...")
    
    iterator = tqdm(frames, desc="Converting") if verbose else frames
    
    for frame in iterator:
        # Process left hand
        hand_tracking_left = frame.get("hand_tracking_left")
        if hand_tracking_left is not None:
            wuji_left = convert_hand_tracking_to_wuji_action(
                hand_tracking_left, "left", left_retargeter
            )
            if wuji_left is not None:
                frame["action_wuji_qpos_target_left"] = wuji_left.tolist()
                left_converted += 1
            else:
                frame["action_wuji_qpos_target_left"] = None
        else:
            frame["action_wuji_qpos_target_left"] = None
        
        # Process right hand
        hand_tracking_right = frame.get("hand_tracking_right")
        if hand_tracking_right is not None:
            wuji_right = convert_hand_tracking_to_wuji_action(
                hand_tracking_right, "right", right_retargeter
            )
            if wuji_right is not None:
                frame["action_wuji_qpos_target_right"] = wuji_right.tolist()
                right_converted += 1
            else:
                frame["action_wuji_qpos_target_right"] = None
        else:
            frame["action_wuji_qpos_target_right"] = None
    
    # Save output
    if verbose:
        print(f" Saving to: {output_path}")
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    
    if verbose:
        print(f"[OK] Done!")
        print(f"   - Left hand converted: {left_converted}/{total_frames} frames")
        print(f"   - Right hand converted: {right_converted}/{total_frames} frames")
    
    return total_frames, left_converted, right_converted


def process_directory(
    input_dir: str,
    output_dir: Optional[str] = None,
    pattern: str = "*.json",
    verbose: bool = True
) -> Dict[str, Tuple[int, int, int]]:
    """
    Process all JSON files in a directory.
    
    Args:
        input_dir: Input directory path
        output_dir: Output directory path (if None, overwrites input files)
        pattern: Glob pattern for JSON files
        verbose: Whether to print progress
        
    Returns:
        Dict mapping filename to (total_frames, left_converted, right_converted)
    """
    input_dir = Path(input_dir)
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    json_files = list(input_dir.glob(pattern))
    
    if verbose:
        print(f" Found {len(json_files)} JSON files in {input_dir}")
    
    results = {}
    for json_file in json_files:
        if output_dir is not None:
            out_file = output_dir / json_file.name
        else:
            out_file = None
        
        try:
            stats = process_json_file(str(json_file), str(out_file) if out_file else None, verbose)
            results[json_file.name] = stats
        except Exception as e:
            print(f"[ERROR] Error processing {json_file.name}: {e}")
            results[json_file.name] = (0, 0, 0)
    
    return results


# ============================================================================
# Main Entry Point
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert hand_tracking data to 20D Wuji hand action",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single file (overwrite in place)
  python convert_hand_tracking_to_wuji_action.py --input data.json

  # Process single file with different output
  python convert_hand_tracking_to_wuji_action.py --input data.json --output converted.json

  # Process all JSON files in a directory
  python convert_hand_tracking_to_wuji_action.py --input_dir ./data/ --output_dir ./converted/

Output format:
  Each frame will have two new fields:
    - action_wuji_qpos_target_left: 20D array (5 fingers x 4 joints) or null
    - action_wuji_qpos_target_right: 20D array (5 fingers x 4 joints) or null
        """
    )
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input", "-i",
        type=str,
        help="Path to input JSON file"
    )
    input_group.add_argument(
        "--input_dir",
        type=str,
        help="Path to directory containing JSON files"
    )
    
    # Output options
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Path to output JSON file (default: overwrite input)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Path to output directory (for directory mode)"
    )
    
    # Other options
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.json",
        help="Glob pattern for JSON files (default: *.json)"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    verbose = not args.quiet
    
    print("=" * 60)
    print("Hand Tracking -> Wuji Action Converter")
    print("=" * 60)
    
    if args.input:
        # Single file mode
        process_json_file(args.input, args.output, verbose)
    else:
        # Directory mode
        results = process_directory(args.input_dir, args.output_dir, args.pattern, verbose)
        
        if verbose:
            print("\n" + "=" * 60)
            print("Summary:")
            print("=" * 60)
            total_files = len(results)
            total_frames = sum(r[0] for r in results.values())
            total_left = sum(r[1] for r in results.values())
            total_right = sum(r[2] for r in results.values())
            print(f"  Files processed: {total_files}")
            print(f"  Total frames: {total_frames}")
            print(f"  Left hand converted: {total_left}")
            print(f"  Right hand converted: {total_right}")


if __name__ == "__main__":
    main()
