# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import sapien.core as sapien
from sapien.utils import Viewer
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from training.utils.config_utils import get_config, save_json
from training.utils.hand_utils import get_entity_by_name, get_active_joints, get_active_joint_indices
from datetime import datetime
from tqdm import tqdm 
import os
from pathlib import Path 
import math

class HandKinematicModel:
    def __init__(self, 
                 scene=None, 
                 render=False, 
                 hand=None, 
                 hand_urdf='', 
                 n_hand_dof=16, 
                 base_link='base_link', 
                 joint_names=[],
                 # Ideally, these two guys (PD controller args) shouldn't be here. 
                 # -- There should be a controller class. I leave them here for code simplicity (maybe truth: or because I am lazy).
                 # If you see your hand model doing something weird (in the simulation viewer below), tune them.
                 kp=400.0, 
                 kd=10):
        
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
            scene_config.default_restitution = 0.00
            scene_config.contact_offset = 0.02
            scene_config.enable_pcm = False
            scene_config.solver_iterations = 25
            scene_config.solver_velocity_iterations = 1
            scene = engine.create_scene(scene_config)  
            self.engine = engine 
        

        self.scene = scene 
        self.renderer = renderer 

        if hand is not None:
            self.hand = hand

        else:
            loader = scene.create_urdf_loader()
            self.hand = loader.load(hand_urdf)
            self.hand.set_root_pose(sapien.Pose([0, 0, 0.35], [0.695, 0, -0.718, 0]))


        self.pmodel = self.hand.create_pinocchio_model()

        # Setup hand base link.
        self.base_link = get_entity_by_name(self.hand.get_links(), base_link)
        self.base_link_idx = self.hand.get_links().index(self.base_link)

        # Setup hand dofs.
        self.all_joints = get_active_joints(self.hand, joint_names)
        all_limits = [joint.get_limits() for joint in self.all_joints]



        # print("=== active joints in sim order (hand.get_active_joints) ===")
        # sim_active = self.hand.get_active_joints()
        # for i, j in enumerate(sim_active):
        #     print(i, j.get_name())

        # print("=== self.all_joints order (from get_active_joints filtered by joint_names) ===")
        # for i, j in enumerate(self.all_joints):
        #     sim_i = [x.get_name() for x in sim_active].index(j.get_name())
        #     print(i, j.get_name(), "sim_index=", sim_i)





        self.joint_names = joint_names
        self.user_idx_to_sim_idx = get_active_joint_indices(self.hand, joint_names)
        print("User-to-Sim Joint", self.user_idx_to_sim_idx)
        self.sim_idx_to_user_idx = [self.user_idx_to_sim_idx.index(i) for i in range(len(self.user_idx_to_sim_idx))]
        print("Sim-to-User Joint", self.sim_idx_to_user_idx)

        self.joint_lower_limit = np.array([l[0][0] for l in all_limits])  # this is in user specified "joint_name" order
        self.joint_upper_limit = np.array([l[0][1] for l in all_limits])  # this is in user specified "joint_name" order
        print(self.joint_lower_limit, self.joint_upper_limit)

        init_qpos = self.convert_user_order_to_sim_order((self.joint_lower_limit + self.joint_upper_limit) / 2)
        self.hand.set_qpos(init_qpos)
        self.hand.set_qvel(0.0 * init_qpos)
        self.qpos_target = init_qpos

        for i, joint in enumerate(self.all_joints):
            print(i, self.joint_names[i], joint, self.joint_lower_limit[i], self.joint_upper_limit[i])
            joint.set_drive_property(kp, kd, force_limit=10)

    def __del__(self):
        del self.engine 
        if hasattr(self, "scene"):
            del self.scene

    def get_n_dof(self):
        '''
            number of dof.
        '''
        return len(self.joint_lower_limit)

    def get_joint_limit(self):
        '''
            Get the hand joint limit.
        '''
        return self.joint_lower_limit, self.joint_upper_limit

    def initialize_keypoint(self, keypoint_link_names, keypoint_offsets):
        '''
            Setup keypoints to track.
        '''
        keypoint_links = [get_entity_by_name(self.hand.get_links(), link) for link in keypoint_link_names]
        print(keypoint_links)

        keypoint_links_id_dict = {link_name: (self.hand.get_links().index(keypoint_links[i]), i) for i, link_name in enumerate(keypoint_link_names)}
        self.keypoint_links = keypoint_links
        self.keypoint_links_id_dict = keypoint_links_id_dict
        self.keypoint_offsets = np.array(keypoint_offsets)

    def convert_user_order_to_sim_order(self, qpos):
        return qpos[self.sim_idx_to_user_idx]

    def keypoint_from_qpos(self, qpos, ret_vec=False):
        '''
            Get keypoints from hand qpos. qpos is specified using the user order.
        '''
        qpos = self.convert_user_order_to_sim_order(qpos)
        self.pmodel.compute_forward_kinematics(qpos)
        base_pose = self.pmodel.get_link_pose(self.base_link_idx)

        result = {} 
        vec_result = []

        for m, (link_idx, i) in self.keypoint_links_id_dict.items():
            pose = self.pmodel.get_link_pose(link_idx)
            new_pose = sapien.Pose(p=pose.p + (pose.to_transformation_matrix()[:3, :3] @ self.keypoint_offsets[i].reshape(3, 1)).reshape(-1), q=pose.q)

            x = (base_pose.inv() * new_pose).p # convert to hand base frame.
            vec_result.append(x)
            result[m] = x

        if ret_vec:
            return np.array(vec_result)
        return result

    @staticmethod
    def build_from_config(config, **kwargs):
        '''
            Build a kinematic model from user config.
        '''
        render = kwargs.get("render", False)
        urdf_path = config["urdf_path"]
        n_hand_dof = len(config["joint_order"])
        base_link = config["base_link"]
        joint_order = config["joint_order"]

        model = HandKinematicModel(hand_urdf=urdf_path, render=render, n_hand_dof=n_hand_dof,base_link=base_link, joint_names=joint_order)
        return model 

    def get_viewer_env(self):
        return HandViewerEnv(self)

    def get_scene(self):
        return self.scene

    def get_renderer(self):
        return self.renderer

    def set_qpos_target(self, qpos):
        '''
            This function is only used during visualization
        '''
        qpos = np.clip(qpos, self.joint_lower_limit + 1e-3, self.joint_upper_limit - 1e-3)
        qpos = self.convert_user_order_to_sim_order(qpos)
        self.qpos_target = qpos 

        for i in range(len(qpos)):
            self.all_joints[i].set_drive_target(self.qpos_target[i])

class HandViewerEnv:
    def __init__(self, model):
        scene = model.get_scene()
        scene.set_timestep(1 / 100.0) 
        scene.set_ambient_light([0.5, 0.5, 0.5])
        scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5], shadow=True)
        scene.add_ground(altitude=0) 

        viewer = Viewer(model.get_renderer())
        viewer.set_scene(scene) 


        viewer.window.set_camera_position([0.1550926,-0.1623763, 0.7064089])
        viewer.window.set_camera_rotation([0.8716827, 0.3260138, 0.12817779, 0.3427167])
        viewer.window.set_camera_parameters(near=0.05, far=100, fovy=1)
        
        self.model = model
        self.scene = scene 
        self.viewer = viewer 

    def update(self):
        self.scene.step()
        self.scene.update_render()  
        self.viewer.render()

if __name__ == '__main__':
    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument('--hand', type=str, default='allegro')

    args = parser.parse_args()

    # Load Hand Model
    config = get_config(args.hand)
    model = HandKinematicModel.build_from_config(config, render=True)
    viewer_env = model.get_viewer_env()
   
    # Control Loop
    n_dof = model.get_n_dof()
    dof_lower, dof_upper = model.get_joint_limit()

    steps = 0
    while True:
        viewer_env.update()

        steps += 1
        if steps % 30 == 0:
            # targets = np.random.uniform(0, 1, n_dof) * (dof_upper - dof_lower - 1e-7) + dof_lower + 1e-7

            mid = (dof_lower + dof_upper) / 2
            targets = mid.copy()
            k = steps // 60 % n_dof
            targets[k] = dof_upper[k]  # 或 dof_lower[k]
            model.set_qpos_target(targets)