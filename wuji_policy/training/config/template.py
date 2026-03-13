# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import sapien.core as sapien
from sapien.utils import Viewer
from training.utils.config_utils import get_config
from training.utils.hand_utils import get_entity_by_name

class HandKinematicModel:
    def __init__(
        self,
        scene=None,
        render=False,
        hand=None,
        hand_urdf="",
        base_link="base_link",
        joint_names=None,
        kp=400.0,
        kd=10.0,
        force_limit=10.0,
    ):
        if joint_names is None:
            joint_names = []

        self.engine = None
        if scene is None:
            engine = sapien.Engine()
            if render:
                renderer = sapien.VulkanRenderer()
                engine.set_renderer(renderer)
                print("Enable Render Mode.")
            else:
                renderer = None

            scene_config = sapien.SceneConfig()
            scene_config.default_dynamic_friction = 1.0
            scene_config.default_static_friction = 1.0
            scene_config.default_restitution = 0.0
            scene_config.contact_offset = 0.02
            scene_config.enable_pcm = False
            scene_config.solver_iterations = 25
            scene_config.solver_velocity_iterations = 1
            scene = engine.create_scene(scene_config)

            self.engine = engine
        else:
            renderer = None

        self.scene = scene
        self.renderer = renderer

        if hand is not None:
            self.hand = hand
        else:
            loader = scene.create_urdf_loader()
            self.hand = loader.load(hand_urdf)

            # Keep your original root pose (you said don't change this)
            self.hand.set_root_pose(sapien.Pose([0, 0, 0.35], [0.695, 0, -0.718, 0]))

        self.pmodel = self.hand.create_pinocchio_model()

        # base link
        self.base_link = get_entity_by_name(self.hand.get_links(), base_link)
        self.base_link_idx = self.hand.get_links().index(self.base_link)

        # Active joints in SAPIEN (sim order)
        self.sim_active_joints = self.hand.get_active_joints()
        self.name2joint = {j.get_name(): j for j in self.sim_active_joints}

        # Validate that all joints in config exist in active joints
        self.joint_names = list(joint_names)
        missing = [n for n in self.joint_names if n not in self.name2joint]
        if len(missing) > 0:
            raise RuntimeError(
                f"These joints are in config joint_order but NOT active in SAPIEN: {missing}"
            )

        # Drive property for ALL active joints (stable baseline)
        for j in self.sim_active_joints:
            j.set_drive_property(kp, kd, force_limit=force_limit)

        # Limits (in YOUR joint_names order)
        self.joint_lower_limit = []
        self.joint_upper_limit = []
        for n in self.joint_names:
            lim = self.name2joint[n].get_limits()  # shape (1,2) for revolute
            self.joint_lower_limit.append(lim[0][0])
            self.joint_upper_limit.append(lim[0][1])
        self.joint_lower_limit = np.array(self.joint_lower_limit, dtype=np.float32)
        self.joint_upper_limit = np.array(self.joint_upper_limit, dtype=np.float32)

        # Initialize qpos to mid for joints in joint_names
        init_q_user = (self.joint_lower_limit + self.joint_upper_limit) / 2.0
        self.set_targets_by_name({n: float(init_q_user[i]) for i, n in enumerate(self.joint_names)})

        # Print joint ordering info for sanity
        print("=== active joints in sim order (hand.get_active_joints) ===")
        for i, j in enumerate(self.sim_active_joints):
            print(i, j.get_name())

        print("=== config joint_order (user order) ===")
        for i, n in enumerate(self.joint_names):
            print(i, n, "limits=", self.joint_lower_limit[i], self.joint_upper_limit[i])

    def set_targets_by_name(self, name2q):
        """Write drive targets by joint name (no ordering ambiguity)."""
        for n, q in name2q.items():
            if n in self.name2joint:
                self.name2joint[n].set_drive_target(float(q))

    def set_thumb_only_target(self, thumb_joint_names, thumb_targets, non_thumb_mode="mid"):
        """
        thumb_joint_names: list[str]
        thumb_targets: np.ndarray shape (len(thumb_joint_names),)
        non_thumb_mode: "mid" (default) keeps all other joints at mid each call.
        """
        # baseline for all joints in config order
        if non_thumb_mode == "mid":
            baseline = (self.joint_lower_limit + self.joint_upper_limit) / 2.0
        else:
            raise ValueError("Only non_thumb_mode='mid' is supported in this script.")

        target_map = {n: float(baseline[i]) for i, n in enumerate(self.joint_names)}

        # overwrite thumb joints
        for n, v in zip(thumb_joint_names, thumb_targets):
            if n not in self.name2joint:
                raise RuntimeError(f"Thumb joint {n} not found in sim active joints.")
            if n in self.joint_names:
                idx = self.joint_names.index(n)
                v = float(
                    np.clip(
                        v,
                        self.joint_lower_limit[idx] + 1e-3,
                        self.joint_upper_limit[idx] - 1e-3,
                    )
                )
            target_map[n] = float(v)

        # write targets by name
        self.set_targets_by_name(target_map)

    @staticmethod
    def build_from_config(config, **kwargs):
        render = kwargs.get("render", False)
        urdf_path = config["urdf_path"]
        base_link = config["base_link"]
        joint_order = config["joint_order"]
        return HandKinematicModel(
            hand_urdf=urdf_path,
            render=render,
            base_link=base_link,
            joint_names=joint_order,
            kp=kwargs.get("kp", 400.0),
            kd=kwargs.get("kd", 10.0),
            force_limit=kwargs.get("force_limit", 10.0),
        )

    def get_viewer_env(self):
        return HandViewerEnv(self)

    def get_scene(self):
        return self.scene

    def get_renderer(self):
        return self.renderer

    def get_joint_limit_by_name(self, joint_name):
        idx = self.joint_names.index(joint_name)
        return float(self.joint_lower_limit[idx]), float(self.joint_upper_limit[idx])


class HandViewerEnv:
    def __init__(self, model: HandKinematicModel):
        scene = model.get_scene()
        scene.set_timestep(1 / 100.0)
        scene.set_ambient_light([0.5, 0.5, 0.5])
        scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5], shadow=True)
        scene.add_ground(altitude=0)

        viewer = Viewer(model.get_renderer())
        viewer.set_scene(scene)

        viewer.window.set_camera_position([0.1550926, -0.1623763, 0.7064089])
        viewer.window.set_camera_rotation([0.8716827, 0.3260138, 0.12817779, 0.3427167])
        viewer.window.set_camera_parameters(near=0.05, far=100, fovy=1)

        self.model = model
        self.scene = scene
        self.viewer = viewer

    def update(self):
        self.scene.step()
        self.scene.update_render()
        self.viewer.render()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--hand", type=str, default="allegro")
    args = parser.parse_args()

    # Load Hand Model
    config = get_config(args.hand)
    model = HandKinematicModel.build_from_config(config, render=True)
    viewer_env = model.get_viewer_env()

    # ONLY move thumb joints (as you specified)
    thumb_joints = ["finger1_joint1", "finger1_joint2", "finger1_joint3", "finger1_joint4"]

    # Sanity: ensure these exist
    for n in thumb_joints:
        if n not in model.name2joint:
            raise RuntimeError(f"Thumb joint not found in sim active joints: {n}")

    steps = 0
    while True:
        viewer_env.update()
        steps += 1

        if steps % 30 == 0:
            # baseline thumb mid
            thumb_mid = np.array(
                [(model.get_joint_limit_by_name(n)[0] + model.get_joint_limit_by_name(n)[1]) / 2.0 for n in thumb_joints],
                dtype=np.float32,
            )
            thumb_targets = thumb_mid.copy()

            # each time, only move ONE of the 4 thumb joints to its upper limit
            k = (steps // 60) % len(thumb_joints)
            jname = thumb_joints[k]
            lo, hi = model.get_joint_limit_by_name(jname)
            thumb_targets[k] = hi

            model.set_thumb_only_target(thumb_joints, thumb_targets, non_thumb_mode="mid")
