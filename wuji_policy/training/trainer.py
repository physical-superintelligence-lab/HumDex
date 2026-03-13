# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import sapien.core as sapien
from torch.utils.data import DataLoader, Dataset
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from training.utils.hand_utils import get_entity_by_name, get_active_joints, get_active_joint_indices
from training.utils.path import get_human_data
from training.utils.config_utils import get_config, save_json
from training.model import FKModel, IKModel
from training.env.hand import HandKinematicModel
from training.formatter import HandFormatter
from training.dataset import RobotKinematicsDataset
from datetime import datetime
from tqdm import tqdm
import os
from pathlib import Path
import math


def merge_dict_list(dl):
    keys = dl[0].keys()

    result = {k: [] for k in keys}
    for data in dl:
        for k in keys:
            result[k].append(data[k])

    result = {k: np.array(v) for k, v in result.items()}
    return result


def format_loss(value):
    return f"{value:.4e}" if math.fabs(value) < 1e-3 else f"{value:.4f}"


def get_float_list_from_np(np_vector):
    float_list = np_vector.tolist()
    float_list = [float(x) for x in float_list]
    return float_list


def generate_current_timestring():
    """Utility Function. Generate a current timestring in the format 'YYYY-MM-DD_HH-MM-SS'."""
    return datetime.now().strftime('%Y-%m-%d_%H-%M-%S')


class SupervisedRetargetDataset(Dataset):
    """
    Supervised dataset: (human_fingertips_rel_wrist -> robot_joint_qpos)

    Expect NPZ keys:
      - fingertips_rel_wrist: [T, 5, 3]
      - qpos (or alternatives): [T, DOF]
    """

    def __init__(self, npz_path: str, qpos_key: str = None, n: int = 20000):
        super().__init__()
        pack = np.load(npz_path, allow_pickle=True)

        if "fingertips_rel_wrist" not in pack.files:
            raise KeyError(
                f"[SupervisedRetargetDataset] 'fingertips_rel_wrist' not found in {npz_path}. "
                f"Available keys: {pack.files}"
            )

        # pick qpos key
        if qpos_key is None:
            candidates = ["qpos", "robot_qpos", "joint", "joint_angle", "joint_angles"]
            qpos_key = None
            for k in candidates:
                if k in pack.files:
                    qpos_key = k
                    break
            if qpos_key is None:
                raise KeyError(
                    f"[SupervisedRetargetDataset] No supervision joint key found. "
                    f"Tried {candidates}. Available keys: {pack.files}"
                )
        else:
            if qpos_key not in pack.files:
                raise KeyError(
                    f"[SupervisedRetargetDataset] qpos_key='{qpos_key}' not found. Available keys: {pack.files}"
                )

        human_points_raw = pack["fingertips_rel_wrist"]  # [T, 5, 3] (or [T, 5, 6/7] but we use :3)
        if human_points_raw.ndim != 3 or human_points_raw.shape[1] != 5 or human_points_raw.shape[2] < 3:
            raise ValueError(
                f"[SupervisedRetargetDataset] Expect fingertips_rel_wrist shape [T,5,>=3], got {human_points_raw.shape}"
            )

        qpos = pack[qpos_key]  # [T, DOF]
        if qpos.ndim != 2:
            raise ValueError(f"[SupervisedRetargetDataset] Expect {qpos_key} shape [T,DOF], got {qpos.shape}")

        T = human_points_raw.shape[0]
        if qpos.shape[0] != T:
            raise ValueError(
                f"[SupervisedRetargetDataset] Time length mismatch: fingertips T={T} vs {qpos_key} T={qpos.shape[0]}"
            )

        self.human_points = human_points_raw[:, :, :3].astype(np.float32)  # [T, 5, 3]
        self.qpos = qpos.astype(np.float32)  # [T, DOF]
        self.T = T
        self.n = int(n)
        self.qpos_key = qpos_key

    def __len__(self):
        # sample with replacement for large epochs
        return self.n

    def __getitem__(self, idx):
        t = np.random.randint(0, self.T)
        point = self.human_points[t]  # [5,3]
        qpos = self.qpos[t]          # [DOF]
        return torch.from_numpy(point), torch.from_numpy(qpos)


class GeoRTTrainer:
    def __init__(self, config):
        self.config = config
        self.hand = HandKinematicModel.build_from_config(self.config)

    def get_robot_kinematics_dataset(self):
        """Utility getter function. Return the robot kinematics dataset"""
        dataset_path = self.get_robot_kinematics_dataset_path(postfix=True)
        if not os.path.exists(dataset_path):
            _ = self.generate_robot_kinematics_dataset(n_total=100000, save=True)
            dataset_path = self.get_robot_kinematics_dataset_path(postfix=True)

        keypoint_names = self.get_keypoint_info()["link"]
        kinematics_dataset = RobotKinematicsDataset(dataset_path, keypoint_names=keypoint_names)
        return kinematics_dataset

    def get_robot_kinematics_dataset_path(self, postfix=False):
        """Utility getter function. Return the path to the robot kinematics dataset."""
        data_name = self.config["name"]
        out = f"data/{data_name}"
        if postfix:
            out += ".npz"
        return out

    def get_keypoint_info(self):
        keypoint_links = []
        keypoint_offsets = []
        keypoint_joints = []
        keypoint_human_ids = []

        joint_order = self.config["joint_order"]

        for info in self.config["fingertip_link"]:
            keypoint_links.append(info["link"])
            keypoint_offsets.append(info["center_offset"])
            keypoint_human_ids.append(info["human_hand_id"])

            keypoint_joint = []
            for joint in info["joint"]:
                keypoint_joint.append(joint_order.index(joint))
            keypoint_joints.append(keypoint_joint)

        return {
            "link": keypoint_links,
            "offset": keypoint_offsets,
            "joint": keypoint_joints,
            "human_id": keypoint_human_ids,
        }

    def generate_robot_kinematics_dataset(self, n_total=100000, save=True):
        """
        Generate a (joint position, keypoint position) dataset.
        - joint order: config["joint_order"]
        - keypoint order: config["fingertip_link"]
        """
        info = self.get_keypoint_info()
        self.hand.initialize_keypoint(keypoint_link_names=info["link"], keypoint_offsets=info["offset"])

        joint_range_low, joint_range_high = self.hand.get_joint_limit()
        joint_range_low = np.array(joint_range_low)
        joint_range_high = np.array(joint_range_high)

        all_data_qpos = []
        all_data_keypoint = []

        for _ in tqdm(range(n_total)):
            qpos = (
                np.random.uniform(0, 1, len(joint_range_low))
                * (joint_range_high - joint_range_low)
                + joint_range_low
            )
            keypoint = self.hand.keypoint_from_qpos(qpos)
            all_data_qpos.append(qpos)
            all_data_keypoint.append(keypoint)

        all_data_keypoint = merge_dict_list(all_data_keypoint)
        dataset = {"qpos": all_data_qpos, "keypoint": all_data_keypoint}

        if save:
            os.makedirs("data", exist_ok=True)
            np.savez(self.get_robot_kinematics_dataset_path(), **dataset)

        return dataset

    def get_fk_checkpoint_path(self):
        name = self.config["name"]
        os.makedirs("checkpoint", exist_ok=True)
        return f"checkpoint/fk_model_{name}.pth"

    def get_robot_neural_fk_model(self, force_train=False):
        """
        Return a forward kinematics model.
        If the fk model does not exist, train one first.
        """
        joint_lower_limit, joint_upper_limit = self.hand.get_joint_limit()
        qpos_normalizer = HandFormatter(joint_lower_limit, joint_upper_limit)

        fk_model = FKModel(keypoint_joints=self.get_keypoint_info()["joint"]).cuda()
        fk_checkpoint_path = self.get_fk_checkpoint_path()

        if os.path.exists(fk_checkpoint_path) and not force_train:
            fk_model.load_state_dict(torch.load(fk_checkpoint_path, weights_only=True))
        else:
            print("Train Neural Forward Kinematics (FK) from Scratch")

            fk_dataset = self.get_robot_kinematics_dataset()
            fk_dataloader = DataLoader(fk_dataset, batch_size=256, shuffle=True)
            fk_optim = optim.Adam(fk_model.parameters(), lr=5e-4)

            criterion_fk = nn.MSELoss()
            for epoch in range(500):
                all_fk_error = 0
                for batch_idx, batch in enumerate(fk_dataloader):
                    keypoint = batch["keypoint"].cuda().float()
                    qpos = batch["qpos"].cuda().float()
                    qpos = qpos_normalizer.normalize_torch(qpos)  # FK is trained on normalized qpos

                    predicted_keypoint = fk_model(qpos)

                    fk_optim.zero_grad()
                    loss = criterion_fk(predicted_keypoint, keypoint)
                    loss.backward()
                    fk_optim.step()

                    all_fk_error += loss.item()

                avg_fk_error = all_fk_error / (batch_idx + 1)
                print(f"Neural FK Training Epoch: {epoch}; Training Loss: {avg_fk_error}")

            torch.save(fk_model.state_dict(), fk_checkpoint_path)

        fk_model.eval()
        return fk_model

    def train(self, human_data_path, **kwargs):
        """
        Supervised trainer:
          input:  human fingertip points (rel wrist)
          target: robot joint qpos (supervision in npz)
          loss:   MSE(ik_model(point), normalize(qpos_gt))
        """
        # IK model
        ik_model = IKModel(keypoint_joints=self.get_keypoint_info()["joint"]).cuda()
        os.makedirs("./checkpoint", exist_ok=True)

        ik_optim = optim.AdamW(ik_model.parameters(), lr=kwargs.get("lr", 1e-4))

        # Workspace
        exp_tag = kwargs.get("tag", "")
        n_epoch = kwargs.get("epoch", 500)
        save_every = kwargs.get("save_every", 10)

        hand_model_name = self.config["name"]

        ckpt_root = Path(kwargs.get("ckpt_root", "/home/jiajunxu/projects/humanoid_tele/GeoRT/checkpoint"))
        run_name = f"{hand_model_name}_{generate_current_timestring()}"
        if exp_tag != "":
            run_name += f"_{exp_tag}"

        save_dir = ckpt_root / run_name
        last_save_dir = ckpt_root / f"{hand_model_name}_last"
        save_dir.mkdir(parents=True, exist_ok=True)
        last_save_dir.mkdir(parents=True, exist_ok=True)

        # Save config (including joint info)
        joint_lower_limit, joint_upper_limit = self.hand.get_joint_limit()
        export_config = self.config.copy()
        export_config["joint"] = {
            "lower": get_float_list_from_np(joint_lower_limit),
            "upper": get_float_list_from_np(joint_upper_limit),
        }
        save_json(export_config, save_dir / "config.json")
        save_json(export_config, last_save_dir / "config.json")

        # Normalizer: match FK training convention (normalized qpos)
        qpos_normalizer = HandFormatter(joint_lower_limit, joint_upper_limit)

        # Dataset (supervised)
        qpos_key = kwargs.get("qpos_key", None)
        n_samples = kwargs.get("n_samples", 20000)
        batch_size = kwargs.get("batch_size", 2048)

        ds = SupervisedRetargetDataset(human_data_path, qpos_key=qpos_key, n=n_samples)
        print(f"[Supervised] Using qpos key: {ds.qpos_key}")

        dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

        # Loss
        criterion = nn.MSELoss()

        # Training
        for epoch in range(n_epoch):
            running = 0.0
            for batch_idx, (point, qpos_gt) in enumerate(dl):
                # point: [B,5,3] -> expected by IKModel: [B,N,3]
                point = point.cuda().float()
                qpos_gt = qpos_gt.cuda().float()  # [B,DOF]

                # normalize supervision target to match model output space
                qpos_gt_norm = qpos_normalizer.normalize_torch(qpos_gt)

                joint_pred = ik_model(point)  # [B,DOF] (assumed normalized space)

                loss = criterion(joint_pred, qpos_gt_norm)

                ik_optim.zero_grad()
                loss.backward()
                ik_optim.step()

                running += loss.item()

                if batch_idx % 50 == 0:
                    avg = running / max(1, (batch_idx + 1))
                    print(f"Epoch {epoch} | MSE Loss: {format_loss(avg)}")

            # checkpointing
            if (epoch % save_every == 0) or (epoch == n_epoch - 1):
                torch.save(ik_model.state_dict(), save_dir / f"epoch_{epoch}.pth")
                torch.save(ik_model.state_dict(), last_save_dir / f"epoch_{epoch}.pth")

            torch.save(ik_model.state_dict(), save_dir / "last.pth")
            torch.save(ik_model.state_dict(), last_save_dir / "last.pth")

        return


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-hand", type=str, default="allegro")
    parser.add_argument("-human_data", type=str, default="human")
    parser.add_argument("-ckpt_tag", type=str, default="")

    # supervised dataset options
    parser.add_argument("--qpos_key", type=str, default=None, help="npz key for supervised joint qpos (e.g., qpos)")
    parser.add_argument("--n_samples", type=int, default=20000)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epoch", type=int, default=500)

    # checkpoint interval
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--ckpt_root", type=str, default="/home/jiajunxu/projects/humanoid_tele/GeoRT/checkpoint")

    args = parser.parse_args()

    config = get_config(args.hand)
    trainer = GeoRTTrainer(config)

    human_data_path = get_human_data(args.human_data)
    print("Training with human data:", human_data_path.as_posix())

    trainer.train(
        human_data_path.as_posix(),
        tag=args.ckpt_tag,
        qpos_key=args.qpos_key,
        n_samples=args.n_samples,
        batch_size=args.batch_size,
        lr=args.lr,
        epoch=args.epoch,
        save_every=args.save_every,
        ckpt_root=args.ckpt_root,
    )
