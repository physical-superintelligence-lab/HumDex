#!/usr/bin/env python3
"""
Visualization module for humanoid body and hand actions.

Provides:
    - HumanoidVisualizer: Visualize 35D body actions with low-level RL policy
    - HandVisualizer: Visualize 20D dexterous hand actions

Usage in training/evaluation:
    from visualizers import HumanoidVisualizer, HandVisualizer, save_video
    
    # Body visualization
    body_viz = HumanoidVisualizer(xml_path, policy_path)
    frames = body_viz.visualize(actions)  # (T, 35) actions -> list of frames
    save_video(frames, "body_viz.mp4", fps=30)
    
    # Hand visualization
    hand_viz = HandVisualizer(xml_path, hand_side="left")
    frames = hand_viz.visualize(actions)  # (T, 20) actions -> list of frames
    save_video(frames, "hand_viz.mp4", fps=30)
"""

import numpy as np
import mujoco
import torch
from collections import deque
from pathlib import Path
from typing import List, Optional, Union

try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False


# =============================================================================
# Utility Functions
# =============================================================================

def save_video(frames: List[np.ndarray], path: str, fps: int = 30) -> bool:
    """Save frames to video file.
    
    Args:
        frames: List of (H, W, 3) uint8 arrays
        path: Output video path
        fps: Frames per second
        
    Returns:
        True if successful, False otherwise
    """
    if not HAS_IMAGEIO:
        print("[WARN] Cannot save video: imageio not installed")
        print("   Install with: pip install imageio imageio-ffmpeg")
        return False
    
    if not frames:
        print("[WARN] Cannot save video: empty frames")
        return False
    try:
        # imageio/ffmpeg expects fixed-size frames; validate upfront to avoid hard crashes.
        h0, w0 = int(frames[0].shape[0]), int(frames[0].shape[1])
        for idx, fr in enumerate(frames):
            a = np.asarray(fr)
            if a.ndim != 3 or a.shape[2] != 3:
                raise ValueError(f"Frame[{idx}] has invalid shape {a.shape}, expected (H,W,3)")
            if int(a.shape[0]) != h0 or int(a.shape[1]) != w0:
                raise ValueError(f"Frame[{idx}] shape {a.shape} mismatches first frame {(h0, w0, 3)}")
            if a.dtype != np.uint8:
                raise ValueError(f"Frame[{idx}] dtype {a.dtype} is not uint8")
        imageio.mimsave(path, frames, fps=fps)
        print(f"[OK] Video saved: {path} ({len(frames)} frames @ {fps} FPS)")
        return True
    except Exception as e:
        print(f"[WARN] Video save failed: {path}: {e}")
        return False


def save_frame(frame: np.ndarray, path: str) -> bool:
    """Save single frame to image file.
    
    Args:
        frame: (H, W, 3) uint8 array
        path: Output image path
        
    Returns:
        True if successful, False otherwise
    """
    if not HAS_IMAGEIO:
        print("[WARN] Cannot save image: imageio not installed")
        return False
    
    imageio.imwrite(path, frame)
    return True


def quat_to_euler(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion (w, x, y, z) to roll-pitch-yaw."""
    w, x, y, z = quat
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
    return np.array([roll, pitch, yaw])


# =============================================================================
# ONNX Policy Wrapper
# =============================================================================

class OnnxPolicyWrapper:
    """ONNX Runtime policy wrapper for low-level RL policy."""
    
    def __init__(self, session, input_name: str, output_index: int = 0):
        self.session = session
        self.input_name = input_name
        self.output_index = output_index

    def __call__(self, obs_tensor) -> torch.Tensor:
        if isinstance(obs_tensor, torch.Tensor):
            obs_np = obs_tensor.detach().cpu().numpy()
        else:
            obs_np = np.asarray(obs_tensor, dtype=np.float32)
        outputs = self.session.run(None, {self.input_name: obs_np})
        result = outputs[self.output_index]
        return torch.from_numpy(result.astype(np.float32))


def load_onnx_policy(policy_path: str, device: str = 'cpu') -> OnnxPolicyWrapper:
    """Load ONNX policy from file.
    
    Args:
        policy_path: Path to .onnx policy file
        device: 'cpu' or 'cuda'
        
    Returns:
        OnnxPolicyWrapper instance
    """
    if not HAS_ONNX:
        raise ImportError("onnxruntime is required. Install with: pip install onnxruntime")
    
    providers = ['CPUExecutionProvider']
    if device.startswith('cuda'):
        if 'CUDAExecutionProvider' in ort.get_available_providers():
            providers.insert(0, 'CUDAExecutionProvider')
    
    session = ort.InferenceSession(policy_path, providers=providers)
    input_name = session.get_inputs()[0].name
    print(f"[OK] ONNX policy loaded: {session.get_providers()}")
    return OnnxPolicyWrapper(session, input_name)


# =============================================================================
# HumanoidVisualizer - Body Actions (35D)
# =============================================================================

class HumanoidVisualizer:
    """Visualize 35D humanoid body actions using low-level RL policy.
    
    Takes high-level 35D action commands and uses an ONNX RL policy to
    generate low-level joint torques for MuJoCo simulation.
    
    Example:
        viz = HumanoidVisualizer(
            xml_path="assets/g1/g1_sim2sim_29dof.xml",
            policy_path="assets/ckpts/twist2_1017_20k.onnx"
        )
        frames = viz.visualize(actions)  # actions: (T, 35)
        save_video(frames, "output.mp4")
    """
    
    def __init__(
        self,
        xml_path: str,
        policy_path: str,
        device: str = 'cpu',
        width: int = 640,
        height: int = 480
    ):
        """Initialize humanoid visualizer.
        
        Args:
            xml_path: Path to MuJoCo XML model
            policy_path: Path to ONNX low-level RL policy
            device: 'cpu' or 'cuda' for policy inference
            width: Rendering width
            height: Rendering height
        """
        # Load MuJoCo model
        print(f"Loading MuJoCo model: {xml_path}")
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.model.opt.timestep = 0.001
        self.data = mujoco.MjData(self.model)

        # Load policy
        print(f"Loading RL policy: {policy_path}")
        self.policy = load_onnx_policy(policy_path, device)
        self.device = device

        # Robot configuration (29 DOF)
        self.default_dof_pos = np.array([
            -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,   # left leg (6)
            -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,   # right leg (6)
            0.0, 0.0, 0.0,                     # torso (3)
            0.0, 0.4, 0.0, 1.2, 0.0, 0.0, 0.0,   # left arm (7)
            0.0, -0.4, 0.0, 1.2, 0.0, 0.0, 0.0,  # right arm (7)
        ])

        self.mujoco_default_dof_pos = np.concatenate([
            np.array([0, 0, 0.793]),           # base position
            np.array([1, 0, 0, 0]),            # base quaternion
            np.array([
                -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,   # left leg (6)
                -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,   # right leg (6)
                0.0, 0.0, 0.0,                     # torso (3)
                0.0, 0.2, 0.0, 1.2, 0.0, 0.0, 0.0,   # left arm (7)
                0.0, -0.2, 0.0, 1.2, 0.0, 0.0, 0.0,  # right arm (7)
            ])
        ])

        self.stiffness = np.array([
            100, 100, 100, 150, 40, 40,
            100, 100, 100, 150, 40, 40,
            150, 150, 150,
            40, 40, 40, 40, 4.0, 4.0, 4.0,
            40, 40, 40, 40, 4.0, 4.0, 4.0,
        ])
        self.damping = np.array([
            2, 2, 2, 4, 2, 2,
            2, 2, 2, 4, 2, 2,
            4, 4, 4,
            5, 5, 5, 5, 0.2, 0.2, 0.2,
            5, 5, 5, 5, 0.2, 0.2, 0.2,
        ])
        self.torque_limits = np.array([
            100, 100, 100, 150, 40, 40,
            100, 100, 100, 150, 40, 40,
            150, 150, 150,
            40, 40, 40, 40, 4.0, 4.0, 4.0,
            40, 40, 40, 40, 4.0, 4.0, 4.0,
        ])
        self.action_scale = np.array([
            0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
            0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
            0.5, 0.5, 0.5,
            0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
            0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
        ])

        self.ankle_idx = [4, 5, 10, 11]
        self.last_action = np.zeros(29, dtype=np.float32)

        # Observation configuration
        self.n_mimic_obs = 35
        self.n_proprio = 92
        self.n_obs_single = 127
        self.history_len = 10
        self.total_obs_size = self.n_obs_single * (self.history_len + 1) + self.n_mimic_obs

        self.proprio_history_buf = deque(maxlen=self.history_len)
        for _ in range(self.history_len):
            self.proprio_history_buf.append(np.zeros(self.n_obs_single, dtype=np.float32))

        # Reset simulation
        self._reset()

        # Setup renderer
        print(f"Creating offscreen renderer ({width}x{height})")
        self.renderer = mujoco.Renderer(self.model, height=height, width=width)

        self.sim_dt = 0.001
        self.control_decimation = 20  # 50Hz control

    def _reset(self):
        """Reset simulation to default pose."""
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.data.qpos[:] = self.mujoco_default_dof_pos
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)
        
        # Reset history
        self.last_action = np.zeros(29, dtype=np.float32)
        self.proprio_history_buf.clear()
        for _ in range(self.history_len):
            self.proprio_history_buf.append(np.zeros(self.n_obs_single, dtype=np.float32))

    def step(self, action_35d: np.ndarray) -> np.ndarray:
        """Execute one control step (50Hz).
        
        Args:
            action_35d: (35,) high-level action command
            
        Returns:
            frame: (H, W, 3) rendered frame
        """
        # Build observation
        dof_pos = self.data.qpos[7:36].copy()
        dof_vel = self.data.qvel[6:35].copy()
        quat = self.data.qpos[3:7].copy()
        ang_vel = self.data.qvel[3:6].copy()

        rpy = quat_to_euler(quat)
        obs_dof_vel = dof_vel.copy()
        obs_dof_vel[self.ankle_idx] = 0.0

        obs_proprio = np.concatenate([
            ang_vel * 0.25,
            rpy[:2],
            (dof_pos - self.default_dof_pos),
            obs_dof_vel * 0.05,
            self.last_action
        ])

        obs_full = np.concatenate([action_35d, obs_proprio])
        obs_hist = np.array(self.proprio_history_buf).flatten()
        self.proprio_history_buf.append(obs_full)
        future_obs = action_35d.copy()
        obs_buf = np.concatenate([obs_full, obs_hist, future_obs])

        # Run policy
        obs_tensor = torch.from_numpy(obs_buf).float().unsqueeze(0)
        with torch.no_grad():
            raw_action = self.policy(obs_tensor).cpu().numpy().squeeze()

        self.last_action = raw_action.copy()
        raw_action = np.clip(raw_action, -10.0, 10.0)
        pd_target = raw_action * self.action_scale + self.default_dof_pos

        # Simulate with PD control
        for _ in range(self.control_decimation):
            dof_pos = self.data.qpos[7:36].copy()
            dof_vel = self.data.qvel[6:35].copy()
            torque = (pd_target - dof_pos) * self.stiffness - dof_vel * self.damping
            torque = np.clip(torque, -self.torque_limits, self.torque_limits)
            self.data.ctrl[:] = torque
            mujoco.mj_step(self.model, self.data)

        # Render
        self.renderer.update_scene(self.data)
        return self.renderer.render()

    def visualize(
        self,
        actions: np.ndarray,
        output_video: Optional[str] = None,
        fps: int = 30,
        warmup_steps: int = 100,
        reset: bool = True,
        verbose: bool = True
    ) -> List[np.ndarray]:
        """Visualize sequence of 35D actions.
        
        Args:
            actions: (T, 35) array of high-level actions
            output_video: Path to save video (optional)
            fps: Video FPS
            warmup_steps: Steps to stabilize at first action before recording
            reset: Whether to reset simulation before visualization
            verbose: Print progress messages
            
        Returns:
            frames: List of (H, W, 3) rendered frames
        """
        if reset:
            self._reset()
            
        if actions.ndim == 1:
            actions = actions.reshape(1, -1)
        if actions.shape[1] != 35:
            raise ValueError(f"Expected (T, 35) actions, got {actions.shape}")

        num_actions = len(actions)
        if verbose:
            print(f"Visualizing {num_actions} actions...")

        # Warm-up phase
        if warmup_steps > 0:
            if verbose:
                print(f"Warm-up: stabilizing at start position ({warmup_steps} steps)...")
            first_action = actions[0]
            for i in range(warmup_steps):
                self.step(first_action)
            if verbose:
                print("[OK] Robot stabilized")

        # Main visualization loop
        frames = []
        for i, action in enumerate(actions):
            frame = self.step(action)
            frames.append(frame)
            if verbose and (i + 1) % 100 == 0:
                print(f"  {i+1}/{num_actions} frames")

        # Save video if requested
        if output_video:
            save_video(frames, output_video, fps=fps)

        if verbose:
            print("[OK] Visualization complete!")

        return frames


# =============================================================================
# HandVisualizer - Hand Actions (20D)
# =============================================================================

class HandVisualizer:
    """Visualize 20D dexterous hand actions.
    
    Directly controls hand joint positions in MuJoCo simulation.
    
    Example:
        viz = HandVisualizer(
            xml_path="wuji_retargeting/example/utils/mujoco-sim/model/left.xml",
            hand_side="left"
        )
        frames = viz.visualize(actions)  # actions: (T, 20)
        save_video(frames, "hand.mp4")
    """
    
    def __init__(
        self,
        xml_path: str,
        hand_side: str = "left",
        width: int = 640,
        height: int = 480
    ):
        """Initialize hand visualizer.
        
        Args:
            xml_path: Path to hand MuJoCo XML model
            hand_side: "left" or "right"
            width: Rendering width
            height: Rendering height
        """
        self.hand_side = hand_side
        self.width = width
        self.height = height

        # Load MuJoCo model
        print(f"Loading Wuji {hand_side} hand model: {xml_path}")
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)

        # Initialize to neutral position
        for i in range(self.model.nu):
            if self.model.actuator_ctrllimited[i]:
                ctrl_range = self.model.actuator_ctrlrange[i]
                self.data.ctrl[i] = (ctrl_range[0] + ctrl_range[1]) / 2
            else:
                self.data.ctrl[i] = 0.0

        # Stabilize model
        print("Stabilizing hand model...")
        for _ in range(100):
            mujoco.mj_step(self.model, self.data)

        # Setup renderer
        print(f"Creating offscreen renderer ({width}x{height})")
        self.renderer = mujoco.Renderer(self.model, height=height, width=width)

        # Camera setup
        self.camera = mujoco.MjvCamera()
        self.camera.azimuth = 180
        self.camera.elevation = -20
        self.camera.distance = 0.5
        self.camera.lookat[:] = [0, 0, 0.05]

        self.sim_dt = self.model.opt.timestep
        print(f"Hand model loaded (timestep: {self.sim_dt}s)")

    def _reset(self):
        """Reset hand to neutral position."""
        mujoco.mj_resetData(self.model, self.data)
        for i in range(self.model.nu):
            if self.model.actuator_ctrllimited[i]:
                ctrl_range = self.model.actuator_ctrlrange[i]
                self.data.ctrl[i] = (ctrl_range[0] + ctrl_range[1]) / 2
            else:
                self.data.ctrl[i] = 0.0
        for _ in range(100):
            mujoco.mj_step(self.model, self.data)

    def step(self, action_20d: np.ndarray) -> np.ndarray:
        """Execute one control step with 20D hand action.
        
        Args:
            action_20d: (20,) or (5, 4) joint position targets
            
        Returns:
            frame: (H, W, 3) rendered frame
        """
        # Handle different input shapes
        if isinstance(action_20d, np.ndarray):
            if action_20d.shape == (5, 4):
                action_20d = action_20d.flatten()
            elif action_20d.shape != (20,):
                raise ValueError(f"Expected (20,) or (5, 4) action, got {action_20d.shape}")

        action_flat = np.array(action_20d).flatten()
        if len(action_flat) != 20:
            raise ValueError(f"Expected 20D action, got {len(action_flat)}D")

        # Set control
        min_len = min(len(action_flat), self.model.nu)
        self.data.ctrl[:min_len] = action_flat[:min_len]

        # Step simulation
        mujoco.mj_step(self.model, self.data)

        # Render
        self.renderer.update_scene(self.data, camera=self.camera)
        return self.renderer.render()

    def visualize(
        self,
        actions: np.ndarray,
        output_video: Optional[str] = None,
        fps: int = 30,
        warmup_steps: int = 50,
        reset: bool = True,
        verbose: bool = True
    ) -> List[np.ndarray]:
        """Visualize sequence of 20D hand actions.
        
        Args:
            actions: (T, 20) or (T, 5, 4) array of actions
            output_video: Path to save video (optional)
            fps: Video FPS
            warmup_steps: Steps to move from neutral to first action before recording
            reset: Whether to reset before visualization
            verbose: Print progress messages
            
        Returns:
            frames: List of (H, W, 3) rendered frames
        """
        if reset:
            self._reset()
            
        # Handle different input shapes
        if actions.ndim == 1:
            actions = actions.reshape(1, -1)
        elif actions.ndim == 3:
            actions = actions.reshape(len(actions), -1)
        elif actions.ndim == 2:
            if actions.shape[1] != 20:
                raise ValueError(f"Expected (T, 20) actions, got {actions.shape}")

        num_frames = len(actions)
        if verbose:
            print(f"Visualizing {num_frames} hand actions...")

        # Warm-up phase: move to first action position
        if warmup_steps > 0:
            first_action = actions[0]
            for _ in range(warmup_steps):
                self.step(first_action)

        frames = []
        for i, action in enumerate(actions):
            frame = self.step(action)
            frames.append(frame)
            if verbose and (i + 1) % 100 == 0:
                print(f"  {i+1}/{num_frames} frames")

        # Save video if requested
        if output_video:
            save_video(frames, output_video, fps=fps)

        if verbose:
            print("[OK] Hand visualization complete!")

        return frames


# =============================================================================
# Convenience function to get default paths
# =============================================================================

def get_default_paths() -> dict:
    """Get default asset paths relative to this module."""
    module_dir = Path(__file__).parent
    return {
        "body_xml": str(module_dir / "assets/g1/g1_sim2sim_29dof.xml"),
        "body_policy": str(module_dir / "assets/ckpts/twist2_1017_20k.onnx"),
        "left_hand_xml": str(module_dir / "wuji_retargeting/example/utils/mujoco-sim/model/left.xml"),
        "right_hand_xml": str(module_dir / "wuji_retargeting/example/utils/mujoco-sim/model/right.xml"),
    }

