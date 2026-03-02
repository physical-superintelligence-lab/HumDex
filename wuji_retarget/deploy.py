import time
import numpy as np
import geort

from geort.utils.config_utils import get_config
from geort.env.hand import HandKinematicModel


class VirdynMocap:
    def __init__(self,
                 npz_path: str,
                 key: str = "fingertips_rel_wrist",
                 loop: bool = True,
                 sleep_dt: float = 0.01):
        self.npz_path = npz_path
        self.key = key
        self.loop = loop
        self.sleep_dt = sleep_dt

        pack = np.load(npz_path, allow_pickle=True)
        if key not in pack.files:
            raise KeyError(f"[VirdynMocap] key '{key}' not found in {npz_path}. "
                           f"Available keys: {pack.files}")

        data = pack[key]  # expected: [T,5,3]
        if data.ndim != 3 or data.shape[-1] != 3:
            raise ValueError(f"[VirdynMocap] Expect shape [T, N, 3], but got {data.shape}")

        self.data = data.astype(np.float32)
        self.T = self.data.shape[0]
        self.N = self.data.shape[1]
        self.idx = 0

        print(f"[VirdynMocap] Loaded {npz_path}, key={key}, shape={self.data.shape}")

    def get(self) -> np.ndarray:
        """
        return: [N,3] float32
        """
        x = self.data[self.idx]  # [N,3]
        self.idx += 1
        if self.idx >= self.T:
            if self.loop:
                self.idx = 0
            else:
                self.idx = self.T - 1

        # 
        if self.sleep_dt is not None and self.sleep_dt > 0:
            time.sleep(self.sleep_dt)

        return x


class WujiHand:
    """
    Wuji (sim ):
    -  GeoRT  HandKinematicModel  wuji_right  urdf
    - command(qpos)  PD drive target
    """
    def __init__(self, hand_name: str = "wuji_right", render: bool = True):
        config = get_config(hand_name)
        self.model = HandKinematicModel.build_from_config(config, render=render)
        self.viewer_env = self.model.get_viewer_env() if render else None

        self.n_dof = self.model.get_n_dof()
        self.lower, self.upper = self.model.get_joint_limit()

        print(f"[WujiHand] Loaded hand={hand_name}, dof={self.n_dof}")

    def command(self, qpos: np.ndarray):
        """
        qpos: [DOF] unnormalized joint angles, in *user joint_order* order (GeoRT config order)
        """
        qpos = np.asarray(qpos, dtype=np.float32).reshape(-1)
        if qpos.shape[0] != self.n_dof:
            raise ValueError(f"[WujiHand] qpos dim mismatch: expect {self.n_dof}, got {qpos.shape[0]}")

        self.model.set_qpos_target(qpos)

        if self.viewer_env is not None:
            self.viewer_env.update()


if __name__ == "__main__":
    checkpoint_tag = "geort_filter_wuji_2"
    model = geort.load_model(checkpoint_tag, epoch=1)

    npz_path = "/home/jiajunxu/projects/humanoid_tele/GeoRT/data/wuji_right.npz"

    mocap = VirdynMocap(npz_path=npz_path, key="fingertips_rel_wrist", loop=True, sleep_dt=0.01)

    robot = WujiHand(hand_name="wuji_right", render=True)

    # ====== 4) inference loop ======
    while True:
        human_points = mocap.get()  # [N,3]
        qpos = model.forward(human_points)  # [DOF], unnormalized
        # breakpoint()
        robot.command(qpos)
