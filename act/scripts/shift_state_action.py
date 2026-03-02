#!/usr/bin/env python3
"""
Shift state-action pairs: state[i] = action[i-1], and drop first/last frames.

This script processes episode data to align states with previous actions:
- New state[i] = Old action[i] (from original frame i)
- New action[i] = Old action[i+1] (from original frame i+1)
- Drop first frame (no previous action) and last frame (no next action)
- Reindex images accordingly

State-Action pairs processed:
- state_body  action_body (extracted to 31D: [3:5] + [6:35])
- state_wuji_hand_left  action_wuji_qpos_target_left (20D)
- state_wuji_hand_right  action_wuji_qpos_target_right (20D)

For state_body, we extract 31 dimensions from action_body:
- action_body[3:5] = roll, pitch (2D)
- action_body[6:35] = joint positions (29D)
- Total = 31D

Action dimensions stay unchanged:
- action_body stays 35D
- action_wuji_qpos_target_* stays 20D

Usage:
    python shift_state_action.py --dataset_dir /path/to/dataset
    python shift_state_action.py --dataset_dir /path/to/dataset --dry_run
"""

import argparse
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm


# ============================================================================
# Configuration: State-Action pairs to process
# ============================================================================

# Mapping: state_key -> (action_key, extraction_function)
# extraction_function transforms action to match state dimensions
STATE_ACTION_PAIRS = {
    "state_body": ("action_body", lambda a: a[3:5] + a[6:35] if a else None),  # 31D
    "state_wuji_hand_left": ("action_wuji_qpos_target_left", lambda a: a),      # 20D as-is
    "state_wuji_hand_right": ("action_wuji_qpos_target_right", lambda a: a),    # 20D as-is
}


def extract_action_body_31d(action_body: Optional[List[float]]) -> Optional[List[float]]:
    """
    Extract 31D from action_body (35D).
    
    action_body structure (35D):
        [0:2]  - XY velocity (2D) - NOT included
        [2]    - Z height (1D) - NOT included
        [3:5]  - roll, pitch (2D) - INCLUDED
        [5]    - yaw angular velocity (1D) - NOT included
        [6:35] - joint positions (29D) - INCLUDED
    
    Returns:
        31D array: [3:5] + [6:35] = roll/pitch (2) + joints (29)
    """
    if action_body is None:
        return None
    if len(action_body) < 35:
        return None
    
    # Extract roll/pitch (indices 3, 4) and joints (indices 6-34)
    return action_body[3:5] + action_body[6:35]


# ============================================================================
# Processing Functions
# ============================================================================

def process_single_episode(
    episode_path: Path,
    dry_run: bool = False,
    verbose: bool = False,
    backup: bool = True
) -> Tuple[str, int, int, str]:
    """
    Process a single episode: shift state-action pairs and reindex images.
    
    Args:
        episode_path: Path to episode folder
        dry_run: If True, only report what would be done
        verbose: Print detailed info
        backup: If True, create data_original.json backup before modifying
        
    Returns:
        Tuple of (episode_name, original_frames, new_frames, status)
    """
    episode_name = episode_path.name
    data_json_path = episode_path / "data.json"
    data_backup_path = episode_path / "data_original.json"
    rgb_dir = episode_path / "rgb"
    rgb_backup_dir = episode_path / "rgb_original"
    
    if not data_json_path.exists():
        return (episode_name, 0, 0, "no data.json")
    
    try:
        # Load JSON
        with open(data_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if "data" not in data:
            return (episode_name, 0, 0, "invalid format")
        
        frames = data["data"]
        original_count = len(frames)
        
        if original_count < 3:
            return (episode_name, original_count, 0, "too few frames (need >= 3)")
        
        if dry_run:
            new_count = original_count - 2
            return (episode_name, original_count, new_count, "dry_run")
        
        # ================================================================
        # Step 0: Create backups before modifying
        # ================================================================
        if backup:
            # Backup data.json -> data_original.json
            if not data_backup_path.exists():
                shutil.copy2(data_json_path, data_backup_path)
            
            # Backup rgb/ -> rgb_original/
            if rgb_dir.exists() and not rgb_backup_dir.exists():
                shutil.copytree(rgb_dir, rgb_backup_dir)
        
        # ================================================================
        # Step 1: Process data frames
        # ================================================================
        new_frames = []
        
        # We keep original frames 1 to N-1 (drop 0 and N)
        # For new frame i (0-indexed):
        #   - state = action from original frame i
        #   - action = action from original frame i+1
        #   - Other fields come from original frame i+1 (the "current" observation)
        
        for new_idx in range(original_count - 2):
            old_idx_for_state = new_idx       # action from this frame becomes state
            old_idx_for_action = new_idx + 1  # action from this frame stays as action
            old_idx_for_others = new_idx + 1  # other fields (rgb, timestamps, etc.)
            
            old_frame_state = frames[old_idx_for_state]
            old_frame_action = frames[old_idx_for_action]
            old_frame_others = frames[old_idx_for_others]
            
            # Create new frame - start with a copy of the "others" frame
            new_frame = dict(old_frame_others)
            
            # Update idx
            new_frame["idx"] = new_idx
            
            # Update rgb path (will be reindexed later)
            new_frame["rgb"] = f"rgb/{new_idx:06d}.jpg"
            
            # Process each state-action pair
            for state_key, (action_key, extractor) in STATE_ACTION_PAIRS.items():
                # New state = old action (from previous frame), extract 31D for body
                old_action_for_state = old_frame_state.get(action_key)
                if state_key == "state_body":
                    new_state = extract_action_body_31d(old_action_for_state)
                else:
                    new_state = old_action_for_state
                new_frame[state_key] = new_state
                
                # New action = old action (from current frame), keep original dimensions
                # action_body stays 35D, hand actions stay 20D
                old_action_for_action = old_frame_action.get(action_key)
                new_frame[action_key] = old_action_for_action
            
            new_frames.append(new_frame)
        
        # ================================================================
        # Step 2: Process images (reindex)
        # ================================================================
        if rgb_dir.exists():
            # Get list of image files, sorted
            image_files = sorted(rgb_dir.glob("*.jpg")) + sorted(rgb_dir.glob("*.png"))
            
            if len(image_files) >= original_count:
                # Create temp directory for reindexing
                temp_dir = episode_path / "_temp_rgb"
                temp_dir.mkdir(exist_ok=True)
                
                # Copy images with new indices (skip first and last)
                for new_idx in range(original_count - 2):
                    old_idx = new_idx + 1  # Original image index
                    
                    # Find the old image file
                    old_image_name = f"{old_idx:06d}"
                    old_image_path = None
                    for ext in [".jpg", ".png"]:
                        candidate = rgb_dir / f"{old_image_name}{ext}"
                        if candidate.exists():
                            old_image_path = candidate
                            break
                    
                    if old_image_path is None:
                        continue
                    
                    # Copy to temp with new name
                    new_image_name = f"{new_idx:06d}{old_image_path.suffix}"
                    new_image_path = temp_dir / new_image_name
                    shutil.copy2(old_image_path, new_image_path)
                
                # Remove old rgb directory and rename temp
                shutil.rmtree(rgb_dir)
                temp_dir.rename(rgb_dir)
        
        # ================================================================
        # Step 3: Save updated data.json
        # ================================================================
        data["data"] = new_frames
        
        with open(data_json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        
        new_count = len(new_frames)
        return (episode_name, original_count, new_count, "success")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return (episode_name, 0, 0, f"error: {str(e)}")


def find_episode_folders(dataset_dir: Path, pattern: str = "episode_*") -> List[Path]:
    """Find all episode folders in dataset directory."""
    episodes = sorted(dataset_dir.glob(pattern))
    episodes = [ep for ep in episodes if ep.is_dir()]
    return episodes


def batch_process_dataset(
    dataset_dir: str,
    dry_run: bool = False,
    episode_pattern: str = "episode_*",
    verbose: bool = True,
    backup: bool = True
) -> dict:
    """
    Batch process all episodes in a dataset.
    
    Args:
        dataset_dir: Path to dataset folder
        dry_run: If True, only check files without modifying
        episode_pattern: Glob pattern for episode folders
        verbose: Whether to print progress
        backup: If True, create backups before modifying
        
    Returns:
        Dict with processing statistics
    """
    dataset_dir = Path(dataset_dir)
    
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    
    episodes = find_episode_folders(dataset_dir, episode_pattern)
    
    if verbose:
        print(f" Dataset: {dataset_dir}")
        print(f"[INFO] Found {len(episodes)} episodes")
        if dry_run:
            print(" Dry run mode - no files will be modified")
        if backup and not dry_run:
            print(" Backup enabled - will create data_original.json and rgb_original/")
        print()
    
    if len(episodes) == 0:
        print("[WARN]  No episodes found!")
        return {"total_episodes": 0}
    
    results = []
    
    iterator = tqdm(episodes, desc="Processing episodes") if verbose else episodes
    for episode_path in iterator:
        result = process_single_episode(episode_path, dry_run, verbose, backup)
        results.append(result)
    
    # Compile statistics
    total_episodes = len(results)
    successful = sum(1 for r in results if r[3] == "success" or r[3] == "dry_run")
    failed = sum(1 for r in results if r[3].startswith("error"))
    too_few = sum(1 for r in results if "too few" in r[3])
    no_data = sum(1 for r in results if r[3] == "no data.json")
    
    total_original_frames = sum(r[1] for r in results)
    total_new_frames = sum(r[2] for r in results)
    total_dropped = total_original_frames - total_new_frames
    
    stats = {
        "total_episodes": total_episodes,
        "successful": successful,
        "failed": failed,
        "too_few_frames": too_few,
        "no_data": no_data,
        "total_original_frames": total_original_frames,
        "total_new_frames": total_new_frames,
        "total_dropped_frames": total_dropped,
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
        if too_few > 0:
            print(f"  [WARN]  Too few frames: {too_few}")
        if no_data > 0:
            print(f"  [WARN]  No data.json: {no_data}")
        print()
        print(f"  Original frames: {total_original_frames}")
        print(f"  New frames: {total_new_frames}")
        print(f"  Dropped frames: {total_dropped} (first + last per episode)")
        print()
        print("  Transformations applied:")
        print("    - state_body  action_body[3:5] + action_body[6:35] (31D)")
        print("    - state_wuji_hand_left  action_wuji_qpos_target_left (20D)")
        print("    - state_wuji_hand_right  action_wuji_qpos_target_right (20D)")
        print("    - action_body stays original (35D)")
        print("    - action_wuji_qpos_target_* stays original (20D)")
        
        # Show failed episodes
        failed_episodes = [r for r in results if r[3].startswith("error")]
        if failed_episodes:
            print()
            print("Failed episodes:")
            for ep_name, _, _, status in failed_episodes[:10]:
                print(f"  - {ep_name}: {status}")
            if len(failed_episodes) > 10:
                print(f"  ... and {len(failed_episodes) - 10} more")
    
    return stats


# ============================================================================
# Main Entry Point
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Shift state-action pairs and reindex images for all episodes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Transformation logic:
  - Original frames: 0, 1, 2, ..., N
  - Drop first (0) and last (N) frames
  - New frame i (originally i+1):
      state[i] = action[i]     (action from original frame i)
      action[i] = action[i+1]  (action from original frame i+1)

Dimension changes:
  - state_body: 34D -> 31D (from action_body[3:5] + [6:35])
  - action_body: stays 35D (unchanged)

Examples:
  # Process all episodes (with automatic backup)
  python shift_state_action.py --dataset_dir /path/to/dataset

  # Process multiple datasets in one command
  python shift_state_action.py --dataset_dir /path/to/d1 /path/to/d2 /path/to/d3

  # Dry run - check without modifying
  python shift_state_action.py --dataset_dir /path/to/dataset --dry_run

  # Process without creating backups
  python shift_state_action.py --dataset_dir /path/to/dataset --no_backup

  # Custom episode pattern
  python shift_state_action.py --dataset_dir /path/to/dataset --pattern "ep_*"

Backup files created:
  - data_original.json  (backup of original data.json)
  - rgb_original/       (backup of original rgb/ folder)
        """
    )
    
    parser.add_argument(
        "--dataset_dir", "-d",
        type=str,
        nargs="+",
        required=True,
        help="One or more dataset directories containing episode folders"
    )
    
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Check files without modifying"
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
    
    parser.add_argument(
        "--no_backup",
        action="store_true",
        help="Skip creating backup files (data_original.json, rgb_original/)"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    verbose = not args.quiet
    
    if verbose:
        print("=" * 60)
        print("State-Action Shift Processor")
        print("=" * 60)
        print()
        print("This will:")
        print("  1. Backup data.json -> data_original.json" if not args.no_backup else "  1. (Backup disabled)")
        print("  2. Backup rgb/ -> rgb_original/" if not args.no_backup else "  2. (Backup disabled)")
        print("  3. Drop first and last frame from each episode")
        print("  4. Set state[i] = action[i-1] (shifted, state_body becomes 31D)")
        print("  5. Keep action as original dimensions (action_body stays 35D)")
        print("  6. Reindex RGB images accordingly")
        print()
    
    start_time = time.time()
    
    dataset_dirs = [str(x) for x in (args.dataset_dir or [])]
    all_stats = []
    for d in dataset_dirs:
        stats = batch_process_dataset(
            dataset_dir=d,
            dry_run=args.dry_run,
            episode_pattern=args.pattern,
            verbose=verbose,
            backup=not args.no_backup,
        )
        all_stats.append(stats)
    
    elapsed = time.time() - start_time
    
    if verbose:
        print()
        print(f"  Total time: {elapsed:.2f}s")
        total_eps = sum(int(s.get("total_episodes", 0)) for s in all_stats)
        if total_eps > 0:
            print(f"   Average per episode: {elapsed / total_eps:.2f}s")


if __name__ == "__main__":
    main()
