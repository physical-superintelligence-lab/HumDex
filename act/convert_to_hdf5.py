"""
Convert JSON + JPEG dataset to compressed HDF5 format

This script converts the episodic dataset from:
  - episode_XXXX/data.json + episode_XXXX/rgb/*.jpg

To a single compressed HDF5 file:
  - dataset.hdf5 with structure:
    /episode_0000/state_body
    /episode_0000/state_wuji_hand_left
    /episode_0000/state_wuji_hand_right
    /episode_0000/action_body
    /episode_0000/action_neck
    /episode_0000/action_wuji_qpos_target_left
    /episode_0000/action_wuji_qpos_target_right
    /episode_0000/rgb  (images stored as compressed arrays)
    ... (repeat for each episode)

Benefits:
- Much smaller file size (compression + no JSON overhead)
- Faster I/O (single file vs thousands of files)
- Easier to transfer and backup
"""

import os
import json
import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse
from typing import List, Optional, Tuple, Dict, Any


def convert_episode_to_hdf5(
    episode_dir: str,
    hdf5_group,
    compress: bool = True,
    source_root: Optional[str] = None,
    source_episode_name: Optional[str] = None,
    state_body_31d: bool = False,
    image_format: str = 'jpeg',
):
    """
    Convert a single episode from JSON+JPEG to HDF5 format

    Args:
        episode_dir: Path to episode_XXXX directory
        hdf5_group: HDF5 group to write data to
        compress: Whether to use compression (gzip)
        source_root: Optional source root folder (for merged datasets)
        source_episode_name: Optional original episode folder name (for merged datasets)
        state_body_31d: If True, extract state_body[3:34] to get 31D (roll/pitch + joints)
                       instead of original 34D (ang_vel + roll/pitch + joints)
        image_format: 'jpeg' to store JPEG bytes directly (recommended), 'raw' for raw array with gzip
    """
    # Load JSON data
    json_path = os.path.join(episode_dir, 'data.json')
    with open(json_path, 'r') as f:
        episode_data = json.load(f)

    timesteps = episode_data['data']
    num_timesteps = len(timesteps)

    # Initialize arrays for each data type
    state_body_list = []
    state_wuji_hand_left_list = []
    state_wuji_hand_right_list = []
    action_body_list = []
    action_neck_list = []
    action_wuji_qpos_target_left_list = []
    action_wuji_qpos_target_right_list = []
    rgb_images = []
    missing_right_count = 0

    # Collect all data from timesteps
    for ts in timesteps:
        # States
        raw_state_body = np.array(ts['state_body'], dtype=np.float32)
        if state_body_31d and len(raw_state_body) >= 34:
            # Extract 31D: [3:34] = roll/pitch (2D) + joints (29D)
            # Skip ang_vel [0:3]
            state_body_list.append(raw_state_body[3:34])
        else:
            state_body_list.append(raw_state_body)
        state_wuji_hand_left_list.append(np.array(ts['state_wuji_hand_left'], dtype=np.float32))
        if 'state_wuji_hand_right' in ts:
            state_wuji_hand_right_list.append(np.array(ts['state_wuji_hand_right'], dtype=np.float32))
        else:
            state_wuji_hand_right_list.append(np.zeros((20,), dtype=np.float32))
            missing_right_count += 1

        # Actions
        action_body_list.append(np.array(ts['action_body'], dtype=np.float32))
        action_neck_list.append(np.array(ts['action_neck'], dtype=np.float32))
        action_wuji_qpos_target_left_list.append(np.array(ts['action_wuji_qpos_target_left'], dtype=np.float32))
        if 'action_wuji_qpos_target_right' in ts:
            action_wuji_qpos_target_right_list.append(np.array(ts['action_wuji_qpos_target_right'], dtype=np.float32))
        else:
            action_wuji_qpos_target_right_list.append(np.zeros((20,), dtype=np.float32))
            missing_right_count += 1

        # Load RGB image
        image_path = os.path.join(episode_dir, ts['rgb'])
        if image_format == 'jpeg':
            # Store JPEG bytes directly (recommended for space efficiency)
            with open(image_path, 'rb') as f:
                jpeg_bytes = np.frombuffer(f.read(), dtype=np.uint8)
            rgb_images.append(jpeg_bytes)
        else:
            # Store raw array (legacy format)
            image = Image.open(image_path)
            rgb_images.append(np.array(image, dtype=np.uint8))

    # Convert lists to numpy arrays
    state_body = np.array(state_body_list, dtype=np.float32)  # (T, 31) if state_body_31d else (T, 34)
    state_wuji_hand_left = np.array(state_wuji_hand_left_list, dtype=np.float32)  # (T, 20)
    state_wuji_hand_right = np.array(state_wuji_hand_right_list, dtype=np.float32)  # (T, 20)
    action_body = np.array(action_body_list, dtype=np.float32)  # (T, 35)
    action_neck = np.array(action_neck_list, dtype=np.float32)  # (T, 2)
    action_wuji_qpos_target_left = np.array(action_wuji_qpos_target_left_list, dtype=np.float32)  # (T, 20)
    action_wuji_qpos_target_right = np.array(action_wuji_qpos_target_right_list, dtype=np.float32)  # (T, 20)
    
    if image_format == 'jpeg':
        # rgb_images is list of variable-length JPEG byte arrays
        rgb = np.array(rgb_images, dtype=object)  # (T,) array of byte arrays
    else:
        # rgb_images is list of fixed-size arrays
        rgb = np.array(rgb_images, dtype=np.uint8)  # (T, H, W, 3)

    # Set compression options
    compression_opts = None
    if compress:
        compression = 'gzip'
        compression_opts = 4  # Compression level (1-9, higher = more compression but slower)
    else:
        compression = None

    # Save to HDF5 with compression
    hdf5_group.create_dataset('state_body', data=state_body,
                             compression=compression, compression_opts=compression_opts)
    hdf5_group.create_dataset('state_wuji_hand_left', data=state_wuji_hand_left,
                             compression=compression, compression_opts=compression_opts)
    hdf5_group.create_dataset('state_wuji_hand_right', data=state_wuji_hand_right,
                             compression=compression, compression_opts=compression_opts)
    hdf5_group.create_dataset('action_body', data=action_body,
                             compression=compression, compression_opts=compression_opts)
    hdf5_group.create_dataset('action_neck', data=action_neck,
                             compression=compression, compression_opts=compression_opts)
    hdf5_group.create_dataset('action_wuji_qpos_target_left', data=action_wuji_qpos_target_left,
                             compression=compression, compression_opts=compression_opts)
    hdf5_group.create_dataset('action_wuji_qpos_target_right', data=action_wuji_qpos_target_right,
                             compression=compression, compression_opts=compression_opts)
    
    # Save images (format depends on image_format parameter)
    if image_format == 'jpeg':
        # Store JPEG bytes directly using variable-length dtype
        # No additional compression needed (JPEG is already compressed)
        dt = h5py.vlen_dtype(np.dtype('uint8'))
        hdf5_group.create_dataset('head', data=rgb, dtype=dt)
    else:
        # Store raw RGB array with gzip compression (legacy format)
        hdf5_group.create_dataset('head', data=rgb,
                                 compression=compression, compression_opts=compression_opts)

    # Save metadata as attributes
    hdf5_group.attrs['num_timesteps'] = num_timesteps
    hdf5_group.attrs['episode_dir'] = os.path.basename(episode_dir)
    hdf5_group.attrs['image_format'] = image_format  # 'jpeg' or 'raw'
    # Mark whether right-hand keys existed in the source JSON for this episode
    hdf5_group.attrs['has_right_hand'] = (missing_right_count == 0)
    hdf5_group.attrs['missing_right_hand_count'] = int(missing_right_count)
    if source_root is not None:
        hdf5_group.attrs['source_root'] = str(source_root)
    if source_episode_name is not None:
        hdf5_group.attrs['source_episode'] = str(source_episode_name)
    if 'info' in episode_data:
        hdf5_group.attrs['info'] = json.dumps(episode_data['info'])
    if 'text' in episode_data:
        hdf5_group.attrs['text'] = json.dumps(episode_data['text'])

    return num_timesteps


def _list_episode_dirs(dataset_dir: str) -> List[str]:
    """List episode_XXXX directories under a dataset_dir (sorted)."""
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"dataset_dir not found: {dataset_dir}")
    episode_dirs = sorted([
        os.path.join(dataset_dir, d)
        for d in os.listdir(dataset_dir)
        if d.startswith('episode_') and os.path.isdir(os.path.join(dataset_dir, d))
    ])
    return episode_dirs


def _collect_episode_dirs(dataset_dirs: List[str], num_episodes: Optional[int]) -> List[Tuple[str, str, str]]:
    """Collect episodes across multiple roots.

    Returns list of tuples: (source_root, episode_name, episode_dir_abs)
    """
    all_eps: List[Tuple[str, str, str]] = []
    for root in dataset_dirs:
        root = os.path.abspath(root)
        for ep_dir in _list_episode_dirs(root):
            ep_name = os.path.basename(ep_dir)
            all_eps.append((root, ep_name, ep_dir))

    # Deterministic ordering: by source_root then episode_name
    all_eps.sort(key=lambda x: (x[0], x[1]))

    if num_episodes is not None:
        all_eps = all_eps[:num_episodes]
    return all_eps


def convert_dataset_to_hdf5(dataset_dirs, output_path, num_episodes=None, compress=True, state_body_31d=False, image_format='jpeg'):
    """
    Convert entire dataset from JSON+JPEG to single HDF5 file

    Args:
        dataset_dirs: Directory (str) OR list of directories containing episode_XXXX folders
        output_path: Path to output HDF5 file
        num_episodes: Number of episodes to convert (None = all episodes)
        compress: Whether to use gzip compression (for non-image data and raw images)
        state_body_31d: If True, extract state_body[3:34] (31D) instead of full 34D
        image_format: 'jpeg' to store JPEG bytes directly (recommended), 'raw' for raw array with gzip
    """
    if isinstance(dataset_dirs, str):
        dataset_dirs = [dataset_dirs]
    dataset_dirs = [os.path.abspath(d) for d in dataset_dirs]

    episodes = _collect_episode_dirs(dataset_dirs, num_episodes=num_episodes)

    print(f"Found {len(episodes)} episodes to convert")
    print(f"Output: {output_path}")
    print(f"Compression: {'gzip (level 4)' if compress else 'none'}")
    print(f"Image format: {image_format} {'(JPEG bytes, recommended)' if image_format == 'jpeg' else '(raw array + gzip)'}")
    print(f"state_body: {'31D (roll/pitch + joints)' if state_body_31d else '34D (original)'}")
    print()

    # Create HDF5 file
    with h5py.File(output_path, 'w') as hdf5_file:
        # Add dataset-level metadata
        hdf5_file.attrs['num_episodes'] = len(episodes)
        hdf5_file.attrs['dataset_dirs'] = json.dumps(dataset_dirs)
        hdf5_file.attrs['compression'] = 'gzip' if compress else 'none'
        hdf5_file.attrs['image_format'] = image_format  # 'jpeg' or 'raw'
        hdf5_file.attrs['state_body_31d'] = state_body_31d
        hdf5_file.attrs['state_body_dim'] = 31 if state_body_31d else 34

        total_timesteps = 0
        episode_lengths = []
        episode_source_map: List[Dict[str, Any]] = []

        # Convert each episode
        for new_idx, (src_root, src_ep_name, episode_dir) in enumerate(tqdm(episodes, desc="Converting episodes")):
            # Re-index episodes sequentially to avoid collisions across multiple roots
            episode_name = f'episode_{new_idx:04d}'

            # Create group for this episode
            episode_group = hdf5_file.create_group(episode_name)

            # Convert and save episode data
            num_timesteps = convert_episode_to_hdf5(
                episode_dir,
                episode_group,
                compress=compress,
                source_root=src_root,
                source_episode_name=src_ep_name,
                state_body_31d=state_body_31d,
                image_format=image_format,
            )

            total_timesteps += num_timesteps
            episode_lengths.append(num_timesteps)
            episode_source_map.append({
                "episode_new": episode_name,
                "source_root": src_root,
                "source_episode": src_ep_name,
            })

        # Save summary statistics
        hdf5_file.attrs['total_timesteps'] = total_timesteps
        hdf5_file.attrs['min_episode_length'] = min(episode_lengths)
        hdf5_file.attrs['max_episode_length'] = max(episode_lengths)
        hdf5_file.attrs['mean_episode_length'] = np.mean(episode_lengths)
        hdf5_file.attrs['episode_source_map'] = json.dumps(episode_source_map)

    # Print statistics
    print("\n" + "="*60)
    print("Conversion Complete!")
    print("="*60)

    # Get file sizes
    original_size = get_directory_size([ep[2] for ep in episodes])
    hdf5_size = os.path.getsize(output_path)

    print(f"\nOriginal size: {original_size / 1024**2:.2f} MB")
    print(f"HDF5 size:     {hdf5_size / 1024**2:.2f} MB")
    print(f"Compression ratio: {original_size / hdf5_size:.2f}x")
    print(f"Space saved: {(original_size - hdf5_size) / 1024**2:.2f} MB ({100 * (1 - hdf5_size/original_size):.1f}%)")

    print(f"\nDataset statistics:")
    print(f"  Episodes: {len(episodes)}")
    print(f"  Total timesteps: {total_timesteps}")
    print(f"  Episode length: {min(episode_lengths)} - {max(episode_lengths)} (mean: {np.mean(episode_lengths):.1f})")
    print(f"  state_body dim: {'31D' if state_body_31d else '34D'}")


def get_directory_size(episode_dirs: List[str]):
    """Calculate total size of episode directories (sum)."""
    total_size = 0
    for episode_dir in episode_dirs:
        for dirpath, dirnames, filenames in os.walk(episode_dir):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total_size += os.path.getsize(filepath)
    return total_size


def verify_conversion(dataset_dir, hdf5_path, episode_idx=0, state_body_31d=False, image_format='jpeg'):
    """
    Verify that conversion was successful by comparing original and HDF5 data

    Args:
        dataset_dir: Original dataset directory
        hdf5_path: Path to HDF5 file
        episode_idx: Episode index to verify
        state_body_31d: Whether state_body was converted to 31D
        image_format: Image storage format ('jpeg' or 'raw')
    """
    print(f"\nVerifying episode {episode_idx}...")

    # Load original data
    episode_dir = os.path.join(dataset_dir, f'episode_{episode_idx:04d}')
    json_path = os.path.join(episode_dir, 'data.json')
    with open(json_path, 'r') as f:
        original_data = json.load(f)

    # Load HDF5 data
    with h5py.File(hdf5_path, 'r') as f:
        hdf5_episode = f[f'episode_{episode_idx:04d}']

        # Compare dimensions
        num_timesteps_original = len(original_data['data'])
        num_timesteps_hdf5 = hdf5_episode.attrs['num_timesteps']

        print(f"  Timesteps: {num_timesteps_original} (original) vs {num_timesteps_hdf5} (HDF5)")
        assert num_timesteps_original == num_timesteps_hdf5, "Timestep count mismatch!"

        # Compare a few random timesteps
        import random
        test_indices = random.sample(range(num_timesteps_original), min(3, num_timesteps_original))

        for ts_idx in test_indices:
            ts_original = original_data['data'][ts_idx]

            # Compare state_body
            state_body_original = np.array(ts_original['state_body'], dtype=np.float32)
            if state_body_31d and len(state_body_original) >= 34:
                state_body_original = state_body_original[3:34]  # Extract 31D for comparison
            state_body_hdf5 = hdf5_episode['state_body'][ts_idx]
            assert np.allclose(state_body_original, state_body_hdf5), f"state_body mismatch at timestep {ts_idx}"

            # Compare state_wuji_hand_left
            state_hand_left_original = np.array(ts_original['state_wuji_hand_left'], dtype=np.float32)
            state_hand_left_hdf5 = hdf5_episode['state_wuji_hand_left'][ts_idx]
            assert np.allclose(state_hand_left_original, state_hand_left_hdf5), f"state_wuji_hand_left mismatch at timestep {ts_idx}"

            # Compare state_wuji_hand_right (if present in source; otherwise should be zeros)
            if 'state_wuji_hand_right' in ts_original:
                state_hand_right_original = np.array(ts_original['state_wuji_hand_right'], dtype=np.float32)
            else:
                state_hand_right_original = np.zeros((20,), dtype=np.float32)
            state_hand_right_hdf5 = hdf5_episode['state_wuji_hand_right'][ts_idx]
            assert np.allclose(state_hand_right_original, state_hand_right_hdf5), f"state_wuji_hand_right mismatch at timestep {ts_idx}"

            # Compare action_body
            action_body_original = np.array(ts_original['action_body'], dtype=np.float32)
            action_body_hdf5 = hdf5_episode['action_body'][ts_idx]
            assert np.allclose(action_body_original, action_body_hdf5), f"action_body mismatch at timestep {ts_idx}"

            # Compare action_wuji_qpos_target_left
            action_hand_left_original = np.array(ts_original['action_wuji_qpos_target_left'], dtype=np.float32)
            action_hand_left_hdf5 = hdf5_episode['action_wuji_qpos_target_left'][ts_idx]
            assert np.allclose(action_hand_left_original, action_hand_left_hdf5), f"action_wuji_qpos_target_left mismatch at timestep {ts_idx}"

            # Compare action_wuji_qpos_target_right (if present; otherwise zeros)
            if 'action_wuji_qpos_target_right' in ts_original:
                action_hand_right_original = np.array(ts_original['action_wuji_qpos_target_right'], dtype=np.float32)
            else:
                action_hand_right_original = np.zeros((20,), dtype=np.float32)
            action_hand_right_hdf5 = hdf5_episode['action_wuji_qpos_target_right'][ts_idx]
            assert np.allclose(action_hand_right_original, action_hand_right_hdf5), f"action_wuji_qpos_target_right mismatch at timestep {ts_idx}"

            # Compare RGB image (stored as 'head' in HDF5)
            image_path = os.path.join(episode_dir, ts_original['rgb'])
            image_original = np.array(Image.open(image_path), dtype=np.uint8)
            
            if image_format == 'jpeg':
                # Decode JPEG bytes from HDF5
                from io import BytesIO
                jpeg_bytes = hdf5_episode['head'][ts_idx]
                image_hdf5 = np.array(Image.open(BytesIO(jpeg_bytes)), dtype=np.uint8)
                # Note: JPEG is lossy, so we can't use exact equality
                # Check that images are very similar (allow small differences due to re-encoding)
                diff = np.abs(image_original.astype(np.int16) - image_hdf5.astype(np.int16))
                max_diff = np.max(diff)
                assert max_diff <= 5, f"Head camera image differs too much at timestep {ts_idx} (max diff: {max_diff})"
            else:
                # Raw format should match exactly
                image_hdf5 = hdf5_episode['head'][ts_idx]
                assert np.array_equal(image_original, image_hdf5), f"Head camera image mismatch at timestep {ts_idx}"

        print(f"   Verified {len(test_indices)} random timesteps - all match!")

    print("   Verification passed!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert JSON+JPEG dataset to HDF5 format')
    parser.add_argument('--dataset_dir', type=str, default=None,
                       help='Directory containing episode_XXXX folders (single source)')
    parser.add_argument('--dataset_dirs', type=str, nargs='+', default=None,
                       help='One or more directories, each containing episode_XXXX folders (will be merged + re-indexed)')
    parser.add_argument('--output', type=str, default='dataset.hdf5',
                       help='Output HDF5 file path')
    parser.add_argument('--num_episodes', type=int, default=None,
                       help='Number of episodes to convert (default: all)')
    parser.add_argument('--no_compress', action='store_true',
                       help='Disable compression (not recommended)')
    parser.add_argument('--state_body_31d', action='store_true',
                       help='Extract state_body[3:34] (31D: roll/pitch + joints) instead of full 34D')
    parser.add_argument('--image_format', type=str, default='jpeg', choices=['jpeg', 'raw'],
                       help='Image storage format: "jpeg" (store JPEG bytes, recommended) or "raw" (raw array + gzip)')
    parser.add_argument('--verify', action='store_true',
                       help='Verify conversion after completion')

    args = parser.parse_args()

    if args.dataset_dirs is not None and len(args.dataset_dirs) > 0:
        dataset_dirs = args.dataset_dirs
    elif args.dataset_dir is not None:
        dataset_dirs = args.dataset_dir
    else:
        raise ValueError("Please pass --dataset_dir or --dataset_dirs")

    # Convert dataset
    convert_dataset_to_hdf5(
        dataset_dirs=dataset_dirs,
        output_path=args.output,
        num_episodes=args.num_episodes,
        compress=not args.no_compress,
        state_body_31d=args.state_body_31d,
        image_format=args.image_format
    )

    # Verify if requested
    if args.verify:
        # Verify uses the first provided directory for raw comparison
        first_dir = args.dataset_dir or (args.dataset_dirs[0] if args.dataset_dirs else None)
        if first_dir is not None:
            verify_conversion(first_dir, args.output, episode_idx=0, 
                            state_body_31d=args.state_body_31d, 
                            image_format=args.image_format)
        print("\n All verification checks passed!")
