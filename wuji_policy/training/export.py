# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import os 
from pathlib import Path
from training.formatter import HandFormatter
from training.model import IKModel
from training.utils.path import to_package_root, get_checkpoint_root
from training.utils.config_utils import load_json, parse_config_keypoint_info, parse_config_joint_limit


class GeoRTRetargetingModel:
    '''
        Used by external programs.
    '''
    def __init__(self, model_path, config_path):
        config = load_json(config_path)
        keypoint_info = parse_config_keypoint_info(config)
        joint_lower_limit, joint_upper_limit = parse_config_joint_limit(config)
        print(keypoint_info["joint"])
        self.human_ids = keypoint_info["human_id"]
        self.model = IKModel(keypoint_joints=keypoint_info["joint"]).cuda()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.qpos_normalizer = HandFormatter(joint_lower_limit, joint_upper_limit) # GeoRT will do normalization.

    def forward(self, keypoints):
        # keypoints: [N, 3]
        keypoints = keypoints[self.human_ids] # extract.
        joint_normalized = self.model.forward(torch.from_numpy(keypoints).unsqueeze(0).reshape(1, -1, 3).float().cuda())
        joint_raw = self.qpos_normalizer.unnormalize(joint_normalized.detach().cpu().numpy())
        return joint_raw[0]


def load_model(tag='', epoch=0):
    '''
        Loading API.
    '''
    checkpoint_root = get_checkpoint_root()
    all_checkpoints = os.listdir(checkpoint_root)
    
    checkpoint_name = ''
    for checkpoint in all_checkpoints:
        if tag in checkpoint:
            checkpoint_name = checkpoint
            break 

    checkpoint_root = Path(checkpoint_root) / checkpoint_name

    # breakpoint()
    if epoch > 0:
        model_path = checkpoint_root / f"epoch_{epoch}.pth"
    else:
        model_path = checkpoint_root / f"last.pth"
    
    config_path = checkpoint_root / "config.json"
    return GeoRTRetargetingModel(model_path=model_path, config_path=config_path)

if __name__ == '__main__':
    # load the model in one line.
    load_model(tag="allegro_last")