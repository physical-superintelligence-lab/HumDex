# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os 
import numpy as np
from pathlib import Path 
from training.utils.path import get_package_root, get_human_data_output_path
from training.utils.config_utils import get_config
from training.export import load_model

def save_human_data(human_data, tag):
    save_path = get_human_data_output_path(tag)
    np.save(save_path, human_data)
    return save_path

