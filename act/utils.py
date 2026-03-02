import numpy as np
import torch
import os
import h5py
from torch.utils.data import TensorDataset, DataLoader
from PIL import Image
from io import BytesIO

import IPython
e = IPython.embed

# =============================================================================
# Relative Action Space Configuration
# =============================================================================
# Dimension mapping for relative actions:
# state_body (31D) maps to specific dimensions of action_body (35D):
#   state_body[0:2]   action_body[3:5]   (roll, pitch)
#   state_body[2:31]  action_body[6:35]  (29 joint positions)
# 
# action_body dimensions NOT in state_body (kept absolute in relative mode):
#   action_body[0:3]  - XY velocity, Z height
#   action_body[5:6]  - unknown dimension
#
# Hand actions fully match between state and action:
#   Single hand mode (hand_side="left" or "right"):
#     state_wuji_hand (20D)  action_wuji_qpos_target (20D)
#     action[35:55] relative to state_hand[0:20]
#   
#   Both hands mode (hand_side="both"):
#     state_wuji_hand_left (20D)  action_wuji_qpos_target_left (20D)
#     state_wuji_hand_right (20D)  action_wuji_qpos_target_right (20D)
#     action[35:55] relative to state_hand_left[0:20]
#     action[55:75] relative to state_hand_right[20:40]

# Standard normalization threshold for statistical stability
NORM_STD_MIN_THRESHOLD = 1e-2


def convert_actions_to_relative(actions, qpos, state_body_dim=31):
    """
    Convert absolute actions to relative actions.
    
    Only matched dimensions are converted to relative:
    - Body: action_body[3:5] and [6:35] -> relative to state_body
    - Hand(s): action_hand -> relative to state_hand
      * Single hand (20D): action[35:55] relative to state_hand[0:20]
      * Both hands (40D): action[35:55] relative to state_hand_left[0:20],
                          action[55:75] relative to state_hand_right[20:40]
    - Unmatched body dims [0:3] and [5:6] remain absolute
    
    Args:
        actions: (T, action_dim) or (action_dim,) - absolute actions
                 action_dim = 55 (single hand) or 75 (both hands)
        qpos: (state_dim,) - current state [state_body + state_hand(s)]
              state_dim = 51 or 54 (single) or 71 or 74 (both)
        state_body_dim: dimension of state_body (31 or 34)
    
    Returns:
        actions_relative: same shape as input - relative actions
    """
    # Handle both batched (T, D) and single (D,) cases
    is_batched = actions.ndim == 2 if hasattr(actions, 'ndim') else len(actions.shape) == 2
    if not is_batched:
        if torch.is_tensor(actions):
            actions = actions.unsqueeze(0)
        else:
            actions = actions[None, :]
    
    # Clone/copy to avoid modifying input
    if torch.is_tensor(actions):
        actions_relative = actions.clone()
    else:
        actions_relative = np.copy(actions)
    
    # Extract state components
    if torch.is_tensor(qpos):
        state_body = qpos[:state_body_dim]
        state_hand_all = qpos[state_body_dim:]  # 20D (single hand) or 40D (both hands)
    else:
        state_body = qpos[:state_body_dim]
        state_hand_all = qpos[state_body_dim:]
    
    # Body: convert matching dimensions to relative (only for 31D, as we don't support 34D yet)
    if state_body_dim == 31:
        # action_body[3:5] (roll, pitch) - relative to state_body[0:2]
        actions_relative[:, 3:5] = actions[:, 3:5] - state_body[0:2]
        # action_body[6:35] (joints) - relative to state_body[2:31]
        actions_relative[:, 6:35] = actions[:, 6:35] - state_body[2:31]
        # action_body[0:3] and [5:6] remain absolute (unmatched dims)
    
    # Hand: convert all dimensions to relative (action_body is 35D, hand starts at 35)
    hand_action_start = 35
    hand_dim = len(state_hand_all)
    
    if hand_dim == 20:
        # Single hand: action[35:55] relative to state_hand[0:20]
        actions_relative[:, hand_action_start:hand_action_start+20] = \
            actions[:, hand_action_start:hand_action_start+20] - state_hand_all
    elif hand_dim == 40:
        # Both hands: split state_hand_all into left and right
        state_hand_left = state_hand_all[:20]
        state_hand_right = state_hand_all[20:40]
        # action[35:55] relative to state_hand_left
        actions_relative[:, 35:55] = actions[:, 35:55] - state_hand_left
        # action[55:75] relative to state_hand_right
        actions_relative[:, 55:75] = actions[:, 55:75] - state_hand_right
    else:
        raise ValueError(f"Unexpected hand dimension: {hand_dim}. Expected 20 (single) or 40 (both)")
    
    if not is_batched:
        if torch.is_tensor(actions_relative):
            actions_relative = actions_relative.squeeze(0)
        else:
            actions_relative = actions_relative.squeeze(0)
    
    return actions_relative


def convert_actions_to_absolute(actions_relative, qpos, state_body_dim=31):
    """
    Convert relative actions back to absolute actions (inverse of convert_actions_to_relative).
    
    Args:
        actions_relative: (T, action_dim) or (action_dim,) - relative actions
                          action_dim = 55 (single hand) or 75 (both hands)
        qpos: (state_dim,) - current state [state_body + state_hand(s)]
              state_dim = 51 or 54 (single) or 71 or 74 (both)
        state_body_dim: dimension of state_body (31 or 34)
    
    Returns:
        actions_absolute: same shape as input - absolute actions
    """
    # Handle both batched (T, D) and single (D,) cases
    is_batched = actions_relative.ndim == 2 if hasattr(actions_relative, 'ndim') else len(actions_relative.shape) == 2
    if not is_batched:
        if torch.is_tensor(actions_relative):
            actions_relative = actions_relative.unsqueeze(0)
        else:
            actions_relative = actions_relative[None, :]
    
    # Clone/copy
    if torch.is_tensor(actions_relative):
        actions_absolute = actions_relative.clone()
    else:
        actions_absolute = np.copy(actions_relative)
    
    # Extract state components
    if torch.is_tensor(qpos):
        state_body = qpos[:state_body_dim]
        state_hand_all = qpos[state_body_dim:]  # 20D (single) or 40D (both)
    else:
        state_body = qpos[:state_body_dim]
        state_hand_all = qpos[state_body_dim:]
    
    # Body: convert matching dimensions back to absolute
    if state_body_dim == 31:
        actions_absolute[:, 3:5] = actions_relative[:, 3:5] + state_body[0:2]
        actions_absolute[:, 6:35] = actions_relative[:, 6:35] + state_body[2:31]
    
    # Hand: convert back to absolute
    hand_action_start = 35
    hand_dim = len(state_hand_all)
    
    if hand_dim == 20:
        # Single hand
        actions_absolute[:, hand_action_start:hand_action_start+20] = \
            actions_relative[:, hand_action_start:hand_action_start+20] + state_hand_all
    elif hand_dim == 40:
        # Both hands
        state_hand_left = state_hand_all[:20]
        state_hand_right = state_hand_all[20:40]
        actions_absolute[:, 35:55] = actions_relative[:, 35:55] + state_hand_left
        actions_absolute[:, 55:75] = actions_relative[:, 55:75] + state_hand_right
    else:
        raise ValueError(f"Unexpected hand dimension: {hand_dim}. Expected 20 (single) or 40 (both)")
    
    if not is_batched:
        if torch.is_tensor(actions_absolute):
            actions_absolute = actions_absolute.squeeze(0)
        else:
            actions_absolute = actions_absolute.squeeze(0)
    
    return actions_absolute


class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        episode_ids,
        dataset_path,
        camera_names,
        norm_stats,
        use_rgb=True,
        hand_side: str = "left",
        action_horizon: int = 0,
        use_relative_actions: bool = False,
        state_body_dim: int = 31,
    ):
        """
        Args:
            episode_ids: List of (file_path, episode_id) tuples OR list of episode_ids (for single file)
            dataset_path: Path to HDF5 file (single) OR list of paths (multiple files)
            camera_names: List of camera names (kept for compatibility)
            norm_stats: Normalization statistics
            use_rgb: Whether to use RGB images
            hand_side: "left", "right", or "both"
            action_horizon: If >0, clamp action sequences to this length
            use_relative_actions: If True, convert actions to relative (vs current state)
            state_body_dim: Dimension of state_body (31 or 34)
        """
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids  # List of (file_path, episode_id) tuples
        self.dataset_path = dataset_path  # Can be single path or list (kept for reference)
        self.camera_names = camera_names  # Keep for compatibility, but we'll use single camera
        self.norm_stats = norm_stats
        self.use_rgb = use_rgb
        self.hand_side = str(hand_side).lower().strip()
        assert self.hand_side in ["left", "right", "both"], \
            f"hand_side must be 'left', 'right', or 'both', got {hand_side!r}"
        # If >0, always return action_data/is_pad with fixed length = action_horizon.
        # This should typically match ACT's chunk_size/num_queries to avoid huge padding in collate_fn.
        self.action_horizon = int(action_horizon) if int(action_horizon) > 0 else 0
        self.use_relative_actions = use_relative_actions
        self.state_body_dim = state_body_dim
        self.is_sim = None
        self._cached_head_hw3 = None  # (H, W, 3), used for black-image fallback if HDF5 image chunks are corrupted
        self._warned_corrupt_image = False
        self.__getitem__(0) # initialize self.is_sim

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        sample_full_episode = False  # hardcode

        # Support both old format (episode_id) and new format (file_path, episode_id)
        episode_entry = self.episode_ids[index]
        if isinstance(episode_entry, tuple):
            file_path, episode_id = episode_entry
        else:
            # Legacy single-file format
            file_path = self.dataset_path if isinstance(self.dataset_path, str) else self.dataset_path[0]
            episode_id = episode_entry

        # Robust HDF5 reading:
        # - Some datasets may contain corrupted compressed chunks (inflate() failed).
        # - Retry a few times; if still failing, try a different episode entry.
        max_attempts = int(os.getenv("ACT_HDF5_MAX_ATTEMPTS", "20"))
        last_err = None
        for attempt in range(max_attempts):
            try:
                # After the first failure, try a random episode to avoid getting stuck on a corrupted one.
                if attempt > 0:
                    episode_entry = self.episode_ids[np.random.randint(0, len(self.episode_ids))]
                    if isinstance(episode_entry, tuple):
                        file_path, episode_id = episode_entry
                    else:
                        file_path = self.dataset_path if isinstance(self.dataset_path, str) else self.dataset_path[0]
                        episode_id = episode_entry

                with h5py.File(file_path, 'r') as f:
                    episode_key = f'episode_{episode_id:04d}'
                    episode_group = f[episode_key]

                    # Get episode length
                    episode_len = int(episode_group.attrs['num_timesteps'])
                    if episode_len <= 0:
                        raise ValueError(f"Invalid episode_len={episode_len} for {episode_key} in {file_path}")

                    if sample_full_episode:
                        start_ts = 0
                    else:
                        start_ts = int(np.random.choice(episode_len))

                    # Load state at start_ts
                    # qpos for single hand: 51-dim or 54-dim (state_body + state_wuji_hand_{left/right})
                    # qpos for both hands: 71-dim or 74-dim (state_body + state_wuji_hand_left + state_wuji_hand_right)
                    state_body = episode_group['state_body'][start_ts]  # (31 or 34,)
                    
                    if self.hand_side == "both":
                        # Load both hands
                        state_hand_left = episode_group.get('state_wuji_hand_left', None)
                        state_hand_right = episode_group.get('state_wuji_hand_right', None)
                        
                        if state_hand_left is None or state_hand_right is None:
                            raise KeyError(
                                f"Dataset missing hand keys for 'both' mode in {episode_key}. "
                                f"Available keys: {list(episode_group.keys())}"
                            )
                        
                        state_hand_left = state_hand_left[start_ts]  # (20,)
                        state_hand_right = state_hand_right[start_ts]  # (20,)
                        qpos = np.concatenate([state_body, state_hand_left, state_hand_right])  # 71 or 74-dim
                    else:
                        # Load single hand (left or right)
                        hand_state_key = f"state_wuji_hand_{self.hand_side}"
                        if hand_state_key not in episode_group:
                            raise KeyError(
                                f"Dataset missing key {hand_state_key!r} in {episode_key}. "
                                f"Available keys: {list(episode_group.keys())}"
                            )
                        state_hand = episode_group[hand_state_key][start_ts]  # (20,)
                        qpos = np.concatenate([state_body, state_hand])  # 51 or 54-dim

                    # Load image (only if needed)
                    if self.use_rgb:
                        # Check image format (backward compatible: default to 'raw' if not specified)
                        image_format = episode_group.attrs.get('image_format', 'raw')
                        
                        if image_format == 'jpeg':
                            # Decode JPEG bytes to array
                            jpeg_bytes = episode_group['head'][start_ts]  # variable-length uint8 array
                            image = np.array(Image.open(BytesIO(jpeg_bytes)), dtype=np.uint8)  # (H, W, 3)
                        else:
                            # Load raw array directly (legacy format)
                            # Image chunks can be corrupted; if so, fall back to a black image.
                            try:
                                image = episode_group['head'][start_ts]  # (H, W, 3)
                                if isinstance(image, np.ndarray) and image.ndim == 3:
                                    self._cached_head_hw3 = image.shape
                            except OSError as e:
                                # h5py read failed (often: inflate() failed). Use black image to keep training running.
                                if not self._warned_corrupt_image:
                                    print(f"[dataset] [WARN] corrupted image chunk detected; falling back to black image. err={e}")
                                    self._warned_corrupt_image = True
                                hw3 = self._cached_head_hw3
                                if hw3 is None:
                                    try:
                                        # shape access doesn't require decompression
                                        hw3 = tuple(episode_group['head'].shape[1:])
                                    except Exception:
                                        hw3 = (480, 640, 3)
                                    self._cached_head_hw3 = hw3
                                image = np.zeros(hw3, dtype=np.uint8)
                    else:
                        # Avoid reading image data at all; just use cached/default shape.
                        hw3 = self._cached_head_hw3
                        if hw3 is None:
                            try:
                                hw3 = tuple(episode_group['head'].shape[1:])
                            except Exception:
                                hw3 = (480, 640, 3)
                            self._cached_head_hw3 = hw3
                        image = np.zeros(hw3, dtype=np.uint8)

                    # Load actions (full episode)
                    # action for single hand: 55-dim (action_body[35] + action_wuji_qpos_target_{left/right}[20])
                    # action for both hands: 75-dim (action_body[35] + action_wuji_qpos_target_left[20] + action_wuji_qpos_target_right[20])
                    
                    if self.hand_side == "both":
                        # Load both hands' actions
                        hand_action_left_key = "action_wuji_qpos_target_left"
                        hand_action_right_key = "action_wuji_qpos_target_right"
                        
                        if hand_action_left_key not in episode_group or hand_action_right_key not in episode_group:
                            raise KeyError(
                                f"Dataset missing hand action keys for 'both' mode in {episode_key}. "
                                f"Available keys: {list(episode_group.keys())}"
                            )
                    else:
                        # Load single hand (left or right)
                        hand_action_key = f"action_wuji_qpos_target_{self.hand_side}"
                        if hand_action_key not in episode_group:
                            raise KeyError(
                                f"Dataset missing key {hand_action_key!r} in {episode_key}. "
                                f"Available keys: {list(episode_group.keys())}"
                            )
                    
                    #
                    # IMPORTANT: do NOT read the whole episode actions into memory for every sample.
                    # That causes large transient allocations and the process RSS will keep growing.
                    # Instead, only read the slice we need, and write it into a preallocated padded buffer.
                    action_start = max(0, start_ts - 1)  # "hack" alignment: previous action for state
                    action_len = episode_len - action_start
                    # Read only the tail slice [action_start:].
                    action_body_ds = episode_group["action_body"]
                    body_dim = int(action_body_ds.shape[1])
                    
                    if self.hand_side == "both":
                        action_hand_left_ds = episode_group["action_wuji_qpos_target_left"]
                        action_hand_right_ds = episode_group["action_wuji_qpos_target_right"]
                        hand_dim = int(action_hand_left_ds.shape[1]) + int(action_hand_right_ds.shape[1])  # 20 + 20 = 40
                        total_action_dim = body_dim + hand_dim  # 35 + 40 = 75
                        
                        body_tail = action_body_ds[action_start:]  # (action_len, 35)
                        hand_left_tail = action_hand_left_ds[action_start:]  # (action_len, 20)
                        hand_right_tail = action_hand_right_ds[action_start:]  # (action_len, 20)
                        
                        padded_action = np.zeros((episode_len, total_action_dim), dtype=np.float32)
                        # Fill valid part; leave padding zeros for the rest.
                        if action_len > 0:
                            padded_action[:action_len, :body_dim] = np.asarray(body_tail, dtype=np.float32)
                            padded_action[:action_len, body_dim:body_dim+20] = np.asarray(hand_left_tail, dtype=np.float32)
                            padded_action[:action_len, body_dim+20:body_dim+40] = np.asarray(hand_right_tail, dtype=np.float32)
                    else:
                        action_hand_ds = episode_group[hand_action_key]
                        hand_dim = int(action_hand_ds.shape[1])  # 20
                        total_action_dim = body_dim + hand_dim  # 35 + 20 = 55
                        
                        body_tail = action_body_ds[action_start:]  # (action_len, body_dim)
                        hand_tail = action_hand_ds[action_start:]  # (action_len, hand_dim)
                        
                        padded_action = np.zeros((episode_len, total_action_dim), dtype=np.float32)
                        # Fill valid part; leave padding zeros for the rest.
                        if action_len > 0:
                            padded_action[:action_len, :body_dim] = np.asarray(body_tail, dtype=np.float32)
                            padded_action[:action_len, body_dim:] = np.asarray(hand_tail, dtype=np.float32)

                # success - break out of retry loop
                break
            except OSError as e:
                # Typical message: "Can't synchronously read data (inflate() failed)"
                last_err = e
                continue
        else:
            # All attempts failed
            raise OSError(
                f"HDF5 read failed after {max_attempts} attempts. "
                f"last_error={repr(last_err)}"
            )

        # Real robot data (not simulation)
        is_sim = False
        self.is_sim = is_sim

        # Build is_pad to match padded_action
        is_pad = np.zeros(episode_len, dtype=np.float32)
        is_pad[action_len:] = 1.0

        # Optional: clamp action horizon to a fixed window (recommended for ACT training).
        # This prevents huge per-batch allocations when some episodes are very long.
        if self.action_horizon > 0:
            H = int(self.action_horizon)
            if padded_action.shape[0] >= H:
                padded_action = padded_action[:H]
                is_pad = is_pad[:H]
            else:
                pad_len = H - int(padded_action.shape[0])
                # Match ACTPolicy's pad_sequence_to_length behavior as closely as possible:
                # In training, padding beyond the available sequence is effectively treated as zeros AFTER normalization.
                # To achieve that here (since we normalize later), we pad with action_mean in raw space so that:
                #   (action_mean - action_mean) / action_std == 0
                try:
                    action_mean = np.asarray(self.norm_stats["action_mean"], dtype=np.float32).reshape(1, -1)
                except Exception:
                    action_mean = np.zeros((1, padded_action.shape[1]), dtype=np.float32)
                pad_rows = np.repeat(action_mean, repeats=pad_len, axis=0).astype(np.float32, copy=False)
                padded_action = np.concatenate(
                    [padded_action, pad_rows],
                    axis=0,
                )
                is_pad = np.concatenate([is_pad, np.ones((pad_len,), dtype=np.float32)], axis=0)

        # Process single camera image - add camera dimension for compatibility
        # Shape: (1, H, W, C) to match original multi-camera format
        all_cam_images = np.expand_dims(image, axis=0)  # (1, H, W, 3)

        # construct observations
        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # Convert to relative actions if enabled (before normalization)
        if self.use_relative_actions:
            # qpos is the state at start_ts, used as reference for all actions in this chunk
            action_data = convert_actions_to_relative(
                action_data, 
                qpos_data, 
                state_body_dim=self.state_body_dim
            )

        # channel last -> channel first: (1, H, W, C) -> (1, C, H, W)
        image_data = torch.einsum('k h w c -> k c h w', image_data)

        # normalize image and change dtype to float
        image_data = image_data / 255.0
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        return image_data, qpos_data, action_data, is_pad


def get_episode_ids(dataset_path):
    """
    Get all episode IDs from HDF5 file(s)

    Args:
        dataset_path: Path to HDF5 file (str) OR list of paths (multiple files)

    Returns:
        If single file: Sorted list of episode IDs (integers)
        If multiple files: List of (file_path, episode_id) tuples
    """
    # Normalize to list
    if isinstance(dataset_path, str):
        paths = [dataset_path]
    else:
        paths = list(dataset_path)
    
    if len(paths) == 1:
        # Single file: return simple list of episode IDs (backward compatible)
        with h5py.File(paths[0], 'r') as f:
            episode_ids = []
            for key in f.keys():
                if key.startswith('episode_'):
                    episode_id = int(key.split('_')[1])
                    episode_ids.append(episode_id)
            episode_ids.sort()
        return episode_ids
    else:
        # Multiple files: return list of (file_path, episode_id) tuples
        all_episodes = []
        for file_path in paths:
            with h5py.File(file_path, 'r') as f:
                for key in f.keys():
                    if key.startswith('episode_'):
                        episode_id = int(key.split('_')[1])
                        all_episodes.append((file_path, episode_id))
        return all_episodes


def get_num_episodes(dataset_path):
    """
    Automatically detect number of episodes in HDF5 file(s)

    Args:
        dataset_path: Path to HDF5 file (str) OR list of paths

    Returns:
        num_episodes: Total number of episodes across all files
    """
    return len(get_episode_ids(dataset_path))


def _load_episode_data(hdf5_file, episode_id, hand_side):
    """
    Helper function to load qpos and action data from an episode.
    
    Args:
        hdf5_file: Open h5py.File object
        episode_id: Episode ID (integer)
        hand_side: "left", "right", or "both"
    
    Returns:
        qpos: (T, state_dim) array - state_dim is 51/54 (single) or 71/74 (both)
        action: (T, action_dim) array - action_dim is 55 (single) or 75 (both)
    """
    episode_key = f'episode_{episode_id:04d}'
    episode_group = hdf5_file[episode_key]

    # Load state: state_body + state_wuji_hand_{side}(s)
    state_body_all = episode_group['state_body'][:]
    
    if hand_side == "both":
        # Load both hands
        if 'state_wuji_hand_left' not in episode_group or 'state_wuji_hand_right' not in episode_group:
            raise KeyError(
                f"Dataset missing hand state keys for 'both' mode in {episode_key}. "
                f"Available keys: {list(episode_group.keys())}"
            )
        state_hand_left = episode_group['state_wuji_hand_left'][:]
        state_hand_right = episode_group['state_wuji_hand_right'][:]
        qpos = np.concatenate([state_body_all, state_hand_left, state_hand_right], axis=1)
    else:
        # Load single hand
        hand_state_key = f"state_wuji_hand_{hand_side}"
        if hand_state_key not in episode_group:
            raise KeyError(
                f"Dataset missing key {hand_state_key!r} in {episode_key}. "
                f"Available keys: {list(episode_group.keys())}"
            )
        state_hand_all = episode_group[hand_state_key][:]
        qpos = np.concatenate([state_body_all, state_hand_all], axis=1)

    # Load action: action_body + action_wuji_qpos_target_{side}(s)
    action_body_all = episode_group['action_body'][:]
    
    if hand_side == "both":
        # Load both hands' actions
        if 'action_wuji_qpos_target_left' not in episode_group or 'action_wuji_qpos_target_right' not in episode_group:
            raise KeyError(
                f"Dataset missing hand action keys for 'both' mode in {episode_key}. "
                f"Available keys: {list(episode_group.keys())}"
            )
        action_hand_left = episode_group['action_wuji_qpos_target_left'][:]
        action_hand_right = episode_group['action_wuji_qpos_target_right'][:]
        action = np.concatenate([action_body_all, action_hand_left, action_hand_right], axis=1)
    else:
        # Load single hand
        hand_action_key = f"action_wuji_qpos_target_{hand_side}"
        if hand_action_key not in episode_group:
            raise KeyError(
                f"Dataset missing key {hand_action_key!r} in {episode_key}. "
                f"Available keys: {list(episode_group.keys())}"
            )
        action_hand_all = episode_group[hand_action_key][:]
        action = np.concatenate([action_body_all, action_hand_all], axis=1)

    return qpos, action


def reindex_episodes_consecutive(dataset_path, output_path=None):
    """
    Rename episode indices to be consecutive (0, 1, 2, ...) in an HDF5 file.
    
    Args:
        dataset_path: Path to input HDF5 file
        output_path: Path to output HDF5 file. If None, modifies in place.
    
    Returns:
        mapping: Dict mapping old episode IDs to new consecutive IDs
    """
    episode_ids = get_episode_ids(dataset_path)
    
    # Create mapping from old IDs to new consecutive IDs
    mapping = {old_id: new_id for new_id, old_id in enumerate(episode_ids)}
    
    if output_path is None:
        output_path = dataset_path
    
    if output_path == dataset_path:
        # In-place modification: rename to temp keys first to avoid conflicts
        with h5py.File(dataset_path, 'a') as f:
            # Step 1: Rename all to temporary keys
            for old_id in episode_ids:
                old_key = f'episode_{old_id:04d}'
                temp_key = f'_temp_episode_{old_id:04d}'
                f.move(old_key, temp_key)
            
            # Step 2: Rename from temp keys to new consecutive keys
            for old_id, new_id in mapping.items():
                temp_key = f'_temp_episode_{old_id:04d}'
                new_key = f'episode_{new_id:04d}'
                f.move(temp_key, new_key)
    else:
        # Copy to new file with consecutive indices
        with h5py.File(dataset_path, 'r') as src, h5py.File(output_path, 'w') as dst:
            for old_id, new_id in mapping.items():
                old_key = f'episode_{old_id:04d}'
                new_key = f'episode_{new_id:04d}'
                src.copy(old_key, dst, name=new_key)
    
    print(f"Reindexed {len(episode_ids)} episodes to consecutive indices 0-{len(episode_ids)-1}")
    print(f"Mapping: {mapping}")
    
    return mapping


def get_norm_stats(dataset_path, episode_ids, hand_side: str = "left", 
                   use_relative_actions: bool = False, state_body_dim: int = 31):
    """
    Compute normalization statistics from HDF5 file(s)

    Args:
        dataset_path: Path to HDF5 file (str) OR list of paths (multiple files)
        episode_ids: List of episode IDs (for single file) OR list of (file_path, episode_id) tuples
        hand_side: "left", "right", or "both" (select which hand state/action to use)
        use_relative_actions: If True, convert actions to relative before computing stats
        state_body_dim: Dimension of state_body (31 or 34)
    """
    hand_side = str(hand_side).lower().strip()
    assert hand_side in ["left", "right", "both"], \
        f"hand_side must be 'left', 'right', or 'both', got {hand_side!r}"
    all_qpos_data = []
    all_action_data = []

    # Determine if we have multi-file format
    is_multi_file = len(episode_ids) > 0 and isinstance(episode_ids[0], tuple)
    
    if is_multi_file:
        # Multi-file format: episode_ids is list of (file_path, episode_id)
        # Group by file to minimize file open/close
        from collections import defaultdict
        episodes_by_file = defaultdict(list)
        for file_path, episode_id in episode_ids:
            episodes_by_file[file_path].append(episode_id)
        
        for file_path, ep_ids in episodes_by_file.items():
            with h5py.File(file_path, 'r') as f:
                for episode_id in ep_ids:
                    qpos, action = _load_episode_data(f, episode_id, hand_side)
                    all_qpos_data.append(torch.from_numpy(qpos))
                    all_action_data.append(torch.from_numpy(action))
    else:
        # Single-file format: episode_ids is list of integers
        single_path = dataset_path if isinstance(dataset_path, str) else dataset_path[0]
        with h5py.File(single_path, 'r') as f:
            for episode_id in episode_ids:
                qpos, action = _load_episode_data(f, episode_id, hand_side)
                all_qpos_data.append(torch.from_numpy(qpos))
                all_action_data.append(torch.from_numpy(action))

    # Convert actions to relative if needed (before computing statistics)
    if use_relative_actions:
        print(f"Converting actions to relative space for normalization (state_body_dim={state_body_dim})...")
        all_action_data_converted = []
        for qpos, action in zip(all_qpos_data, all_action_data):
            # For each episode, convert each timestep's action to relative
            # action[t] is relative to qpos[t]
            action_relative = []
            for t in range(len(action)):
                # Use qpos[t] as reference for action[t]
                action_t_relative = convert_actions_to_relative(
                    action[t:t+1],  # (1, action_dim)
                    qpos[t],         # (state_dim,)
                    state_body_dim=state_body_dim
                )
                action_relative.append(action_t_relative)
            action_relative = torch.cat(action_relative, dim=0)
            all_action_data_converted.append(action_relative)
        all_action_data = all_action_data_converted
    
    # CHANGE: Concatenate all episodes along timestep dimension instead of stacking
    # This handles variable episode lengths by treating all timesteps equally
    # Original shape after stack: (num_episodes, episode_len, dim) - FAILS with variable lengths
    # New shape after concat: (total_timesteps, dim) - WORKS with variable lengths
    # For single hand: qpos (51 or 54), action (55)
    # For both hands: qpos (71 or 74), action (75)
    all_qpos_data = torch.cat(all_qpos_data, dim=0)  # (total_timesteps, state_dim)
    all_action_data = torch.cat(all_action_data, dim=0)  # (total_timesteps, action_dim)

    # CHANGE: Compute mean/std along dimension 0 only (no dimension 1 since we concatenated)
    # normalize action data
    action_mean = all_action_data.mean(dim=0, keepdim=True)  # (1, action_dim) - 55 or 75
    action_std = all_action_data.std(dim=0, keepdim=True)  # (1, action_dim)
    action_std = torch.clip(action_std, NORM_STD_MIN_THRESHOLD, np.inf)  # Use configurable threshold

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=0, keepdim=True)  # (1, state_dim) - 51, 54, 71, or 74
    qpos_std = all_qpos_data.std(dim=0, keepdim=True)  # (1, state_dim)
    qpos_std = torch.clip(qpos_std, NORM_STD_MIN_THRESHOLD, np.inf)  # Use configurable threshold

    # Compute dimensions
    state_dim = all_qpos_data.shape[1]
    action_dim = all_action_data.shape[1]
    
    # Normalize paths for metadata
    if isinstance(dataset_path, str):
        dataset_paths_list = [os.path.abspath(dataset_path)]
    else:
        dataset_paths_list = [os.path.abspath(p) for p in dataset_path]
    
    stats = {
        "action_mean": action_mean.numpy().squeeze(),
        "action_std": action_std.numpy().squeeze(),
        "qpos_mean": qpos_mean.numpy().squeeze(),
        "qpos_std": qpos_std.numpy().squeeze(),
        "example_qpos": qpos[0],  # First timestep from last episode
        # Metadata for traceability
        "state_dim": state_dim,
        "action_dim": action_dim,
        "hand_side": hand_side,
        "dataset_paths": dataset_paths_list,
        "num_episodes": len(episode_ids),
        "total_timesteps": int(all_qpos_data.shape[0]),
        # Relative action configuration
        "use_relative_actions": use_relative_actions,
        "state_body_dim": state_body_dim,
    }

    return stats


def custom_collate_fn(batch):
    """
    Custom collate function to handle variable-length episodes in a batch.
    Pads all episodes in the batch to the maximum episode length in that batch.

    Args:
        batch: List of tuples (image_data, qpos_data, action_data, is_pad)
               where action_data and is_pad have variable lengths

    Returns:
        Batched tensors with padding
    """
    # Separate the batch into components
    images, qpos, actions, is_pads = zip(*batch)

    # Stack images and qpos (these have fixed sizes per sample)
    images_batch = torch.stack(images, dim=0)  # (batch, 1, C, H, W)
    qpos_batch = torch.stack(qpos, dim=0)  # (batch, 54)

    # Find max episode length in this batch
    max_len = max(int(action.shape[0]) for action in actions)

    # Pad actions and is_pad to max_len
    actions_padded = []
    is_pads_padded = []

    for action, is_pad in zip(actions, is_pads):
        current_len = action.shape[0]
        if current_len < max_len:
            # Pad with zeros
            pad_len = max_len - current_len
            action_padded = torch.cat([action, torch.zeros(pad_len, action.shape[1])], dim=0)
            is_pad_padded = torch.cat([is_pad, torch.ones(pad_len, dtype=torch.bool)], dim=0)
        else:
            action_padded = action
            is_pad_padded = is_pad

        actions_padded.append(action_padded)
        is_pads_padded.append(is_pad_padded)

    # Stack padded tensors
    actions_batch = torch.stack(actions_padded, dim=0)  # (batch, max_len, 55)
    is_pads_batch = torch.stack(is_pads_padded, dim=0)  # (batch, max_len)

    return images_batch, qpos_batch, actions_batch, is_pads_batch


def load_data(
    dataset_path,
    num_episodes,
    camera_names,
    batch_size_train,
    batch_size_val,
    use_rgb=True,
    hand_side: str = "left",
    split_save_path: str = None,
    *,
    val_robot_only: bool = False,
    use_relative_actions: bool = False,
    state_body_dim: int = 31,
):
    """
    Load data from HDF5 file(s) and create dataloaders

    Args:
        dataset_path: Path to HDF5 file (str) OR list of paths (multiple files to merge)
        num_episodes: Number of episodes in the dataset (kept for compatibility, can be None to auto-detect)
        camera_names: List of camera names (kept for compatibility)
        batch_size_train: Training batch size
        batch_size_val: Validation batch size
        use_rgb: If True, load real images. If False, use black placeholder images (state-only training)
        hand_side: "left", "right", or "both"
        split_save_path: Optional path to save train/val split for reproducibility
        val_robot_only: If True, use only robot episodes for validation
        use_relative_actions: If True, use relative action space
        state_body_dim: Dimension of state_body (31 or 34)

    Returns:
        train_dataloader, val_dataloader, norm_stats, is_sim
    """
    # Normalize dataset_path to list for consistent handling
    if isinstance(dataset_path, str):
        dataset_paths = [dataset_path]
    else:
        dataset_paths = list(dataset_path)
    
    print(f'\nData from {len(dataset_paths)} HDF5 file(s): (use_rgb={use_rgb})')
    for p in dataset_paths:
        print(f'  - {p}')
    print()

    # Get actual episode IDs (handles non-consecutive indices and multiple files)
    episode_ids = get_episode_ids(dataset_paths)
    num_episodes_actual = len(episode_ids)
    
    # Check if multi-file format
    is_multi_file = len(episode_ids) > 0 and isinstance(episode_ids[0], tuple)
    
    print(f'Total episodes: {num_episodes_actual}')
    if is_multi_file:
        # Count episodes per file
        from collections import Counter
        file_counts = Counter(ep[0] for ep in episode_ids)
        for fp, cnt in file_counts.items():
            print(f'  - {os.path.basename(fp)}: {cnt} episodes')
    print()
    
    # obtain train/val split using actual episode IDs
    # Default behavior: random 80/20 split on all episodes.
    # Optional: val_robot_only=True -> use ONLY robot episodes for validation (to make val_loss reflect robot data).
    train_ratio = 0.8
    shuffled_indices = np.random.permutation(num_episodes_actual)
    train_episode_ids = [episode_ids[i] for i in shuffled_indices[:int(train_ratio * num_episodes_actual)]]
    val_episode_ids = [episode_ids[i] for i in shuffled_indices[int(train_ratio * num_episodes_actual):]]

    if bool(val_robot_only) and is_multi_file:
        # Heuristic: treat files whose basename contains "robot" as robot datasets.
        robot_files = {p for p in dataset_paths if "robot" in os.path.basename(str(p)).lower()}
        robot_episode_ids = [ep for ep in episode_ids if isinstance(ep, tuple) and ep[0] in robot_files]
        if len(robot_episode_ids) > 0:
            # Use all robot episodes for validation; train on the rest (avoid overlap).
            val_episode_ids = list(robot_episode_ids)
            val_set = set(val_episode_ids)
            train_episode_ids = [ep for ep in episode_ids if ep not in val_set]
        else:
            print("[load_data] [WARN] val_robot_only requested but no robot dataset file matched (basename contains 'robot'). Using default val split.")
    
    # optionally save split to json for reproducibility
    if split_save_path is not None:
        import json
        split_save_path = os.path.abspath(split_save_path)
        split_dir = os.path.dirname(split_save_path)
        if split_dir:
            os.makedirs(split_dir, exist_ok=True)
        
        # Convert episode_ids to serializable format
        if is_multi_file:
            episode_ids_json = [[fp, int(eid)] for fp, eid in episode_ids]
            train_ids_json = [[fp, int(eid)] for fp, eid in train_episode_ids]
            val_ids_json = [[fp, int(eid)] for fp, eid in val_episode_ids]
        else:
            episode_ids_json = [int(x) for x in episode_ids]
            train_ids_json = [int(x) for x in train_episode_ids]
            val_ids_json = [int(x) for x in val_episode_ids]
        
        payload = {
            "dataset_paths": [os.path.abspath(p) for p in dataset_paths],
            "num_episodes_actual": int(num_episodes_actual),
            "is_multi_file": is_multi_file,
            "episode_ids": episode_ids_json,
            "train_episode_ids": train_ids_json,
            "val_episode_ids": val_ids_json,
            "train_ratio": float(train_ratio),
            "hand_side": str(hand_side),
            "use_rgb": bool(use_rgb),
            "val_robot_only": bool(val_robot_only),
        }
        with open(split_save_path, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)


    # obtain normalization stats for qpos and action using all episode IDs
    norm_stats = get_norm_stats(
        dataset_paths, 
        episode_ids, 
        hand_side=hand_side,
        use_relative_actions=use_relative_actions,
        state_body_dim=state_body_dim,
    )

    # construct dataset and dataloader
    # If ACT_CHUNK_SIZE is set, clamp action horizon at dataset level to avoid huge padding in batches.
    # Recommended: set this to the same value as --chunk_size (ACT num_queries).
    action_horizon = int(os.getenv("ACT_CHUNK_SIZE", "0"))
    train_dataset = EpisodicDataset(
        train_episode_ids,
        dataset_paths,
        camera_names,
        norm_stats,
        use_rgb=use_rgb,
        hand_side=hand_side,
        action_horizon=action_horizon,
        use_relative_actions=use_relative_actions,
        state_body_dim=state_body_dim,
    )
    val_dataset = EpisodicDataset(
        val_episode_ids,
        dataset_paths,
        camera_names,
        norm_stats,
        use_rgb=use_rgb,
        hand_side=hand_side,
        action_horizon=action_horizon,
        use_relative_actions=use_relative_actions,
        state_body_dim=state_body_dim,
    )

    # Use custom collate function to handle variable-length episodes
    # NOTE: h5py + multi-worker DataLoader can be fragile on some systems.
    # Default to 0 workers for stability; override via env ACT_DATALOADER_WORKERS.
    num_workers = int(os.getenv("ACT_DATALOADER_WORKERS", "0"))

    dl_common = dict(
        batch_size=batch_size_train,
        shuffle=True,
        pin_memory=False,
        num_workers=num_workers,
        collate_fn=custom_collate_fn,
    )
    if num_workers > 0:
        dl_common["prefetch_factor"] = 1

    train_dataloader = DataLoader(train_dataset, **dl_common)

    dl_val = dict(dl_common)
    dl_val["batch_size"] = batch_size_val
    val_dataloader = DataLoader(val_dataset, **dl_val)

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim


### helper functions

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result

def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
