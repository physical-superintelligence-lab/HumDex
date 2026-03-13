# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random
import numpy as np
import open3d as o3d


def upsample_array(x, K=50000):
    n = x.shape[0]
    if n <= 0:
        raise ValueError("upsample_array got empty input.")
    ind = np.random.randint(0, n, K)
    return x[ind]


class MultiPointDataset:
    def __init__(self, points):
        # points: [Num_Fingers, Num_Samples, 3]
        self.points = np.array(points, dtype=np.float32)

    @staticmethod
    def from_points(points, n, resample_to=50000, resample_resolution=0.001):
        """
        points: [Num_Fingers, Num_Samples, 3]
        NOTE: 参数 n 在原实现中没有被使用（MultiPointDataset 的 len 由 resample_to 决定）
        """
        num_fingers = points.shape[0]
        all_points = []

        # Resampling to reduce spatial imbalance.
        for finger_id in range(num_fingers):
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points[finger_id])
            downpcd = pcd.voxel_down_sample(voxel_size=resample_resolution)
            resampled_points = np.asarray(downpcd.points)

            if resampled_points.shape[0] == 0:
                # 退化情况：如果 voxel_down_sample 后没有点，回退到原始点
                resampled_points = np.asarray(pcd.points)

            all_points.append(upsample_array(resampled_points, K=resample_to))

        return MultiPointDataset(np.array(all_points).astype(np.float32))

    def __len__(self):
        return self.points.shape[1]

    def __getitem__(self, idx):
        # return: [Num_Fingers, 3]
        return self.points[:, idx]


class RobotKinematicsDataset:
    """
    兼容两种数据格式：

    A) GeoRT 原生格式（npz keys: 'qpos', 'keypoint' (dict)）
    B) 你的 wuji npz 格式（npz keys:
        'action_wuji_qpos_target', 'fingertips_rel_wrist', 'fingertip_names', ...）
    """

    def __init__(self, qpos_keypoint_file, keypoint_names):
        npz = np.load(qpos_keypoint_file, allow_pickle=True)
        self.npz_files = list(npz.files)

        self.keypoint_names = list(keypoint_names)

        # ---------- Case A: 原生 GeoRT ----------
        if "qpos" in npz.files and "keypoint" in npz.files:
            self.mode = "geort_native"
            self.qpos = npz["qpos"]  # [N, dof]
            # keypoint 是 dict: {name: [N, 3 or ...]}
            self.keypoints_dict = npz["keypoint"].item()
            self.available_keypoint_names = list(self.keypoints_dict.keys())
            self.n = len(self.qpos)
            return

        # ---------- Case B: 你的 wuji npz ----------
        required = ["action_wuji_qpos_target", "fingertips_rel_wrist"]
        for k in required:
            if k not in npz.files:
                raise KeyError(
                    f"Your npz does not contain required key '{k}'. "
                    f"Available keys: {npz.files}"
                )

        self.mode = "wuji_npz"

        # qpos: [T, dof]
        self.qpos = npz["action_wuji_qpos_target"].astype(np.float32)

        # keypoints: [T, 5, 3]
        self.keypoints_array = npz["fingertips_rel_wrist"].astype(np.float32)

        if self.keypoints_array.ndim != 3 or self.keypoints_array.shape[-1] != 3:
            raise ValueError(
                f"Expected fingertips_rel_wrist to have shape [T, F, 3], "
                f"but got {self.keypoints_array.shape}"
            )

        self.n = self.qpos.shape[0]
        if self.keypoints_array.shape[0] != self.n:
            raise ValueError(
                f"Mismatch in time dimension: qpos has {self.n} frames, "
                f"but fingertips_rel_wrist has {self.keypoints_array.shape[0]} frames"
            )

        # 名称：npz 里通常有 fingertip_names: [F]
        if "fingertip_names" in npz.files:
            self.fingertip_names = [str(x) for x in npz["fingertip_names"].tolist()]
        else:
            # 如果没有，就用默认顺序名
            F = self.keypoints_array.shape[1]
            self.fingertip_names = [f"finger_{i}" for i in range(F)]

        # 建立 name -> index 映射
        self.name_to_index = {name: i for i, name in enumerate(self.fingertip_names)}

        return

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        qpos = self.qpos[idx].astype(np.float32)

        if self.mode == "geort_native":
            keypoint_data = []
            for name in self.keypoint_names:
                if name not in self.keypoints_dict:
                    raise KeyError(
                        f"Requested keypoint '{name}' not found in native dataset. "
                        f"Available: {list(self.keypoints_dict.keys())}"
                    )
                keypoint_data.append(self.keypoints_dict[name][idx][:3])
            keypoint = np.array(keypoint_data, dtype=np.float32)
            return {"qpos": qpos, "keypoint": keypoint}

        # mode == "wuji_npz"
        keypoint_data = []

        # 优先按名字匹配；如果匹配不上，则按顺序回退（保证能跑，但建议你对齐命名）
        for j, name in enumerate(self.keypoint_names):
            if name in self.name_to_index:
                fi = self.name_to_index[name]
            else:
                # 回退：按顺序选第 j 个 finger
                fi = j
                if fi >= self.keypoints_array.shape[1]:
                    raise KeyError(
                        f"Requested keypoint '{name}' cannot be mapped. "
                        f"Dataset fingertip_names={self.fingertip_names}, "
                        f"and fallback index {fi} is out of range."
                    )
            keypoint_data.append(self.keypoints_array[idx, fi, :3])

        keypoint = np.array(keypoint_data, dtype=np.float32)
        return {"qpos": qpos, "keypoint": keypoint}

    def export_robot_pointcloud(self, keypoint_names):
        """
        训练代码里会用它做 Chamfer 的 target 点云。
        在原版里，它返回 shape: [N_keypoints, N_samples, 3]
        """
        if self.mode == "geort_native":
            all_keypoint_data = []
            for keypoint_name in keypoint_names:
                all_keypoint_data.append(self.keypoints_dict[keypoint_name])
            return np.array(all_keypoint_data)

        # mode == "wuji_npz"
        # 这里返回 [K, T, 3]，其中 K = len(keypoint_names)
        all_keypoint_data = []
        for j, name in enumerate(keypoint_names):
            if name in self.name_to_index:
                fi = self.name_to_index[name]
            else:
                fi = j
                if fi >= self.keypoints_array.shape[1]:
                    raise KeyError(
                        f"Keypoint '{name}' cannot be mapped for export_robot_pointcloud. "
                        f"Dataset fingertip_names={self.fingertip_names}"
                    )
            all_keypoint_data.append(self.keypoints_array[:, fi, :])
        return np.array(all_keypoint_data, dtype=np.float32)


if __name__ == "__main__":
    # 示例：读取你的 wuji 数据
    ds = RobotKinematicsDataset(
        qpos_keypoint_file="/home/jiajunxu/projects/humanoid_tele/data/your_wuji_file.npz",
        keypoint_names=["thumb", "index", "middle", "ring", "pinky"],  # 你要按你的 fingertip_names 改
    )
    print("mode:", ds.mode)
    print("npz files:", ds.npz_files)
    print("len:", len(ds))
    print("item0:", ds[0]["qpos"].shape, ds[0]["keypoint"].shape)
