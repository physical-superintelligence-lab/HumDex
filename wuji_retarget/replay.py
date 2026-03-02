#!/usr/bin/env python3
import argparse
import numpy as np
import geort

from geort.utils.config_utils import get_config
from geort.env.hand import HandKinematicModel


def find_key(pack_files, prefer_key: str):
    if prefer_key in pack_files:
        return prefer_key
    # 
    for k in ["fingertips_rel_wrist", "human_points", "keypoints_rel_wrist"]:
        if k in pack_files:
            return k
    raise KeyError(f"Cannot find keypoints key. Available keys: {list(pack_files)}")


def main():
    parser = argparse.ArgumentParser("Replay one training sample for GeoRT retarget model")
    parser.add_argument("--checkpoint_tag", type=str, default="geort_filter_wuji_2")
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--npz_path", type=str, default="/home/jiajunxu/projects/humanoid_tele/GeoRT/data/wuji_right.npz")
    parser.add_argument("--key", type=str, default="fingertips_rel_wrist", help="key name for human fingertip points")
    parser.add_argument("--frame_idx", type=int, default=0, help="which frame (sample) to replay")
    parser.add_argument("--hand_name", type=str, default="wuji_right")
    parser.add_argument("--no_render", action="store_true", help="disable Sapien rendering (only print inference result)")
    args = parser.parse_args()

    # 1) load model
    print(f"[INFO] Loading model: tag={args.checkpoint_tag}, epoch={args.epoch}")
    model = geort.load_model(args.checkpoint_tag, epoch=args.epoch)

    # 2) load one training sample (human points)
    pack = np.load(args.npz_path, allow_pickle=True)
    key = find_key(pack.files, args.key)
    data = pack[key]  # expect [T, N, 3]
    if data.ndim != 3 or data.shape[-1] != 3:
        raise ValueError(f"[ERROR] Expect {key} shape [T,N,3], got {data.shape}")
    
    T, N, _ = data.shape
    frame_idx = int(args.frame_idx)
    if frame_idx < 0:
        frame_idx = T + frame_idx
    if not (0 <= frame_idx < T):
        raise IndexError(f"[ERROR] frame_idx {args.frame_idx} out of range. T={T}")

    human_points = data[frame_idx].astype(np.float32)  # [N,3]
    print(f"[INFO] Loaded sample: npz={args.npz_path}, key={key}, T={T}, N={N}, frame_idx={frame_idx}")

    # 3) inference
    qpos = model.forward(human_points)  # [DOF]
    qpos = np.asarray(qpos, dtype=np.float32).reshape(-1)

    # 4) print for verification
    np.set_printoptions(precision=6, suppress=True)
    print("\n========== REPLAY SAMPLE ==========")
    print(f"[human_points] shape={human_points.shape}")
    print(human_points)
    print("-----------------------------------")
    print(f"[qpos_pred] shape={qpos.shape}")
    print(qpos)
    print("===================================\n")

    # 5) optional: render the pose once
    if not args.no_render:
        print("[INFO] Building HandKinematicModel for rendering...")
        config = get_config(args.hand_name)
        hand = HandKinematicModel.build_from_config(config, render=True)
        viewer_env = hand.get_viewer_env()

        # set target once
        hand.set_qpos_target(qpos)
        print("[INFO] Rendering... (close the viewer window or Ctrl+C to exit)")
        try:
            while True:
                viewer_env.update()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
