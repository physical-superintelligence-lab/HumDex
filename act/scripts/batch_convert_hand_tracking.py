#!/usr/bin/env python3
"""
Batch convert hand_tracking data to 20D Wuji hand action for all episodes in a dataset.

This script processes a dataset folder containing multiple episode_xxxx subfolders,
each with a data.json file, and converts hand_tracking_left/right to 
action_wuji_qpos_target_left/right.

Dataset structure:
    dataset_folder/
     episode_0000/
        data.json
        rgb/
     episode_0001/
        data.json
        rgb/
     ...

Usage:
    python batch_convert_hand_tracking.py --dataset_dir /path/to/dataset
    python batch_convert_hand_tracking.py --dataset_dir /path/to/dataset --workers 4
    python batch_convert_hand_tracking.py --dataset_dir /path/to/dataset --dry_run
"""

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from tqdm import tqdm

# Add paths for wuji_retargeting
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # TWIST2/
WUJI_RETARGETING_PATH = PROJECT_ROOT / "wuji_retargeting"
if str(WUJI_RETARGETING_PATH) not in sys.path:
    sys.path.insert(0, str(WUJI_RETARGETING_PATH))


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

MEDIAPIPE_MAPPING_26_TO_21 = [
    1, 2, 3, 4, 5,      # Wrist (Palm), Thumb (4 joints)
    6, 7, 8, 10,        # Index (skip IndexDistal at 9)
    11, 12, 13, 15,     # Middle (skip MiddleDistal at 14)
    16, 17, 18, 20,     # Ring (skip RingDistal at 19)
    21, 22, 23, 25,     # Pinky (skip LittleDistal at 24)
]


# ============================================================================
# Conversion Functions (copied from convert_hand_tracking_to_wuji_action.py)
# ============================================================================

def hand_26d_to_mediapipe_21d(hand_data_dict: dict, hand_side: str = "left") -> np.ndarray:
    """Convert 26D hand tracking data to 21D MediaPipe format."""
    hand_side_prefix = "LeftHand" if hand_side.lower() == "left" else "RightHand"
    
    joint_positions_26 = np.zeros((26, 3), dtype=np.float32)
    for i, joint_name in enumerate(HAND_JOINT_NAMES_26):
        key = hand_side_prefix + joint_name
        if key in hand_data_dict:
            pos = hand_data_dict[key][0]
            joint_positions_26[i] = pos
    
    mediapipe_21d = joint_positions_26[MEDIAPIPE_MAPPING_26_TO_21]
    wrist_pos = mediapipe_21d[0].copy()
    mediapipe_21d = mediapipe_21d - wrist_pos
    
    return mediapipe_21d


def convert_single_hand(hand_tracking_data: dict, hand_side: str, retargeter) -> Optional[np.ndarray]:
    """Convert a single hand's tracking data to 20D Wuji action."""
    if hand_tracking_data is None:
        return None
    
    if not hand_tracking_data.get("is_active", False):
        return None
    
    hand_dict = {k: v for k, v in hand_tracking_data.items() 
                 if k not in ["is_active", "timestamp"]}
    
    if not hand_dict:
        return None
    
    try:
        from wuji_retargeting.mediapipe import apply_mediapipe_transformations
        
        mediapipe_21d = hand_26d_to_mediapipe_21d(hand_dict, hand_side)
        mediapipe_transformed = apply_mediapipe_transformations(mediapipe_21d, hand_type=hand_side)
        result = retargeter.retarget(mediapipe_transformed)
        return result.robot_qpos
    except Exception as e:
        return None


def process_single_episode(episode_path: Path, dry_run: bool = False) -> Tuple[str, int, int, int, str]:
    """
    Process a single episode's data.json file.
    
    Args:
        episode_path: Path to episode folder (e.g., episode_0000/)
        dry_run: If True, only check files without modifying
        
    Returns:
        Tuple of (episode_name, total_frames, left_converted, right_converted, status)
    """
    episode_name = episode_path.name
    data_json_path = episode_path / "data.json"
    
    if not data_json_path.exists():
        return (episode_name, 0, 0, 0, "no data.json")
    
    try:
        # Load JSON
        with open(data_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if "data" not in data:
            return (episode_name, 0, 0, 0, "invalid format")
        
        frames = data["data"]
        total_frames = len(frames)
        
        if dry_run:
            # Just count how many frames have hand tracking data
            left_count = sum(1 for f in frames if f.get("hand_tracking_left", {}).get("is_active", False))
            right_count = sum(1 for f in frames if f.get("hand_tracking_right", {}).get("is_active", False))
            return (episode_name, total_frames, left_count, right_count, "dry_run")
        
        # Import and initialize retargeters
        from wuji_retargeting import WujiHandRetargeter
        left_retargeter = WujiHandRetargeter(hand_side="left")
        right_retargeter = WujiHandRetargeter(hand_side="right")
        
        left_converted = 0
        right_converted = 0
        
        for frame in frames:
            # Process left hand
            hand_tracking_left = frame.get("hand_tracking_left")
            wuji_left = convert_single_hand(hand_tracking_left, "left", left_retargeter)
            if wuji_left is not None:
                frame["action_wuji_qpos_target_left"] = wuji_left.tolist()
                left_converted += 1
            else:
                frame["action_wuji_qpos_target_left"] = None
            
            # Process right hand
            hand_tracking_right = frame.get("hand_tracking_right")
            wuji_right = convert_single_hand(hand_tracking_right, "right", right_retargeter)
            if wuji_right is not None:
                frame["action_wuji_qpos_target_right"] = wuji_right.tolist()
                right_converted += 1
            else:
                frame["action_wuji_qpos_target_right"] = None
        
        # Save back to file
        with open(data_json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        
        return (episode_name, total_frames, left_converted, right_converted, "success")
        
    except Exception as e:
        return (episode_name, 0, 0, 0, f"error: {str(e)}")


def find_episode_folders(dataset_dir: Path, pattern: str = "episode_*") -> List[Path]:
    """Find all episode folders in dataset directory."""
    episodes = sorted(dataset_dir.glob(pattern))
    # Filter to only directories
    episodes = [ep for ep in episodes if ep.is_dir()]
    return episodes


def batch_process_dataset(
    dataset_dir: str,
    workers: int = 1,
    dry_run: bool = False,
    episode_pattern: str = "episode_*",
    verbose: bool = True
) -> dict:
    """
    Batch process all episodes in a dataset.
    
    Args:
        dataset_dir: Path to dataset folder
        workers: Number of parallel workers (1 = sequential)
        dry_run: If True, only check files without modifying
        episode_pattern: Glob pattern for episode folders
        verbose: Whether to print progress
        
    Returns:
        Dict with processing statistics
    """
    dataset_dir = Path(dataset_dir)
    
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    
    # Find all episode folders
    episodes = find_episode_folders(dataset_dir, episode_pattern)
    
    if verbose:
        print(f" Dataset: {dataset_dir}")
        print(f"[INFO] Found {len(episodes)} episodes")
        if dry_run:
            print(" Dry run mode - no files will be modified")
        print()
    
    if len(episodes) == 0:
        print("[WARN]  No episodes found!")
        return {"total_episodes": 0}
    
    results = []
    
    if workers == 1:
        # Sequential processing
        iterator = tqdm(episodes, desc="Processing episodes") if verbose else episodes
        for episode_path in iterator:
            result = process_single_episode(episode_path, dry_run)
            results.append(result)
    else:
        # Parallel processing
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(process_single_episode, ep, dry_run): ep for ep in episodes}
            
            iterator = tqdm(as_completed(futures), total=len(futures), desc="Processing episodes") if verbose else as_completed(futures)
            for future in iterator:
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    episode_path = futures[future]
                    results.append((episode_path.name, 0, 0, 0, f"error: {str(e)}"))
    
    # Compile statistics
    total_episodes = len(results)
    successful = sum(1 for r in results if r[4] == "success" or r[4] == "dry_run")
    failed = sum(1 for r in results if r[4].startswith("error"))
    no_data = sum(1 for r in results if r[4] == "no data.json")
    
    total_frames = sum(r[1] for r in results)
    total_left = sum(r[2] for r in results)
    total_right = sum(r[3] for r in results)
    
    stats = {
        "total_episodes": total_episodes,
        "successful": successful,
        "failed": failed,
        "no_data": no_data,
        "total_frames": total_frames,
        "total_left_converted": total_left,
        "total_right_converted": total_right,
        "results": results
    }
    
    if verbose:
        print()
        print("=" * 60)
        print("Summary")
        print("=" * 60)
        print(f"  Episodes processed: {successful}/{total_episodes}")
        if failed > 0:
            print(f"  [ERROR] Failed: {failed}")
        if no_data > 0:
            print(f"  [WARN]  No data.json: {no_data}")
        print(f"  Total frames: {total_frames}")
        print(f"  Left hand converted: {total_left}")
        print(f"  Right hand converted: {total_right}")
        
        # Show failed episodes
        failed_episodes = [r for r in results if r[4].startswith("error")]
        if failed_episodes:
            print()
            print("Failed episodes:")
            for ep_name, _, _, _, status in failed_episodes[:10]:  # Show first 10
                print(f"  - {ep_name}: {status}")
            if len(failed_episodes) > 10:
                print(f"  ... and {len(failed_episodes) - 10} more")
    
    return stats


# ============================================================================
# Main Entry Point
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch convert hand_tracking to Wuji action for all episodes in a dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all episodes in dataset (sequential)
  python batch_convert_hand_tracking.py --dataset_dir /path/to/dataset

  # Process with 4 parallel workers
  python batch_convert_hand_tracking.py --dataset_dir /path/to/dataset --workers 4

  # Dry run - check files without modifying
  python batch_convert_hand_tracking.py --dataset_dir /path/to/dataset --dry_run

  # Custom episode pattern
  python batch_convert_hand_tracking.py --dataset_dir /path/to/dataset --pattern "ep_*"

Dataset structure expected:
    dataset_dir/
     episode_0000/
        data.json
        rgb/
     episode_0001/
        data.json
        rgb/
     ...
        """
    )
    
    parser.add_argument(
        "--dataset_dir", "-d",
        type=str,
        required=True,
        help="Path to dataset directory containing episode folders"
    )
    
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1 = sequential)"
    )
    
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Check files without modifying (count hand tracking data)"
    )
    
    parser.add_argument(
        "--pattern", "-p",
        type=str,
        default="episode_*",
        help="Glob pattern for episode folders (default: episode_*)"
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
    
    if verbose:
        print("=" * 60)
        print("Batch Hand Tracking -> Wuji Action Converter")
        print("=" * 60)
    
    start_time = time.time()
    
    stats = batch_process_dataset(
        dataset_dir=args.dataset_dir,
        workers=args.workers,
        dry_run=args.dry_run,
        episode_pattern=args.pattern,
        verbose=verbose
    )
    
    elapsed = time.time() - start_time
    
    if verbose:
        print()
        print(f"  Total time: {elapsed:.2f}s")
        if stats["total_episodes"] > 0:
            print(f"   Average per episode: {elapsed / stats['total_episodes']:.2f}s")


if __name__ == "__main__":
    main()
