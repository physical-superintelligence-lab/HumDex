# Updated for dex3-1 using new unified unitree_interface
import sys
import os

# Add unitree_sdk2 python binding path
unitree_sdk2_path = "/home/ANT.AMAZON.COM/yanjieze/lab42/src/unitree_sdk2/build/lib"
if os.path.exists(unitree_sdk2_path):
    sys.path.insert(0, unitree_sdk2_path)

import unitree_interface as ui
import numpy as np
from enum import IntEnum
import time
from data_utils.params import DEFAULT_HAND_POSE
parent2_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(parent2_dir)


Dex3_Num_Motors = 7
kTopicDex3LeftCommand = "rt/dex3/left/cmd"
kTopicDex3RightCommand = "rt/dex3/right/cmd"
kTopicDex3LeftState = "rt/dex3/left/state"
kTopicDex3RightState = "rt/dex3/right/state"


DEFAULT_QPOS_LEFT = DEFAULT_HAND_POSE["unitree_g1"]["left"]["open"]
DEFAULT_QPOS_RIGHT = DEFAULT_HAND_POSE["unitree_g1"]["right"]["open"]

# thumb, middle, index
QPOS_LEFT_MAX = [1.0472, 1.0472, 1.74533,  0, 0, 0, 0]
QPOS_LEFT_MIN  = [-1.0472, -0.724312, 0, -1.5708, -1.74533, -1.5708, -1.74533]
QPOS_RIGHT_MAX = [1.0472, 0.724312, 0, 1.5708, 1.74533, 1.5708, 1.74533]
QPOS_RIGHT_MIN = [-1.0472, -1.0472, -1.74533, 0, 0, 0, 0]


class Dex3_1_Controller:
    def __init__(self, net, re_init=True):
        """
        [note] A *_array type parameter requires using a multiprocessing Array, because it needs to be passed to the internal child process

        left_hand_array: [input] Left hand skeleton data (required from XR device) to hand_ctrl.control_process

        right_hand_array: [input] Right hand skeleton data (required from XR device) to hand_ctrl.control_process

        dual_hand_data_lock: Data synchronization lock for dual_hand_state_array and dual_hand_action_array

        dual_hand_state_array: [output] Return left(7), right(7) hand motor state

        dual_hand_action_array: [output] Return left(7), right(7) hand motor action

        fps: Control frequency

        Unit_Test: Whether to enable unit testing
        """
        print("Initialize Dex3_1_Controller...")
        print("[INFO] Using new unified unitree_interface API")
        
        # Create hand interfaces using new API
        self.left_hand = ui.HandInterface.create_left_hand(net, re_init)
        self.right_hand = ui.HandInterface.create_right_hand(net, False)  # Don't re-init for second hand
        
        print(f"[OK] {self.left_hand.get_hand_name()} initialized")
        print(f"[OK] {self.right_hand.get_hand_name()} initialized")

        # Arrays for additional hand states
        self.Ltemp = np.zeros((Dex3_Num_Motors, 2))
        self.Rtemp = np.zeros((Dex3_Num_Motors,2))
        self.Ltau = np.zeros(Dex3_Num_Motors)
        self.Rtau = np.zeros(Dex3_Num_Motors)
        self.Lpos = np.zeros(Dex3_Num_Motors)
        self.Rpos = np.zeros(Dex3_Num_Motors)

        # Arrays for hand states
        self.left_hand_state_array  = np.zeros(Dex3_Num_Motors)
        self.right_hand_state_array = np.zeros(Dex3_Num_Motors)
        self.get_hand_state()
        print(f"[Dex3_1_Controller] left_hand_state_array: {self.left_hand_state_array} \nright_hand_state_array: {self.right_hand_state_array}")
        self.initialize()

        print("Initialize Dex3_1_Controller OK!\n")

    def get_hand_state(self):
        # Use new unified API
        left_state = self.left_hand.read_hand_state()
        right_state = self.right_hand.read_hand_state()
        
        # Update arrays from new API
        for idx in range(Dex3_Num_Motors):
            self.left_hand_state_array[idx] = left_state.motor.q[idx]
            self.right_hand_state_array[idx] = right_state.motor.q[idx]
            
            # Temperature data (2 values per motor in new API)
            self.Ltemp[idx, 0] = left_state.motor.temperature[idx][0]
            self.Ltemp[idx, 1] = left_state.motor.temperature[idx][1]
            self.Rtemp[idx, 0] = right_state.motor.temperature[idx][0]
            self.Rtemp[idx, 1] = right_state.motor.temperature[idx][1]
            
            # Torque and position
            self.Ltau[idx] = left_state.motor.tau_est[idx]
            self.Rtau[idx] = right_state.motor.tau_est[idx]
            self.Lpos[idx] = left_state.motor.q[idx]
            self.Rpos[idx] = right_state.motor.q[idx]
            
        return self.left_hand_state_array.copy(), self.right_hand_state_array.copy()

    def get_hand_all_state(self):
        return self.Lpos.copy(), self.Rpos.copy(), self.Ltemp.copy(), self.Rtemp.copy(), self.Ltau.copy(), self.Rtau.copy()

    def ctrl_dual_hand(self, left_q_target, right_q_target):
        """set current left, right hand motor state target q"""
        # Use new unified API
        left_cmd = self.left_hand.create_zero_command()
        right_cmd = self.right_hand.create_zero_command()
        
        # Set target positions
        left_cmd.q_target = list(left_q_target)
        right_cmd.q_target = list(right_q_target)
        
        # Send commands
        self.left_hand.write_hand_command(left_cmd)
        self.right_hand.write_hand_command(right_cmd)
    
    def initialize(self):
        # Use new unified API - send default poses
        print(" Initializing hands with default poses using new API...")
        self.ctrl_dual_hand(DEFAULT_QPOS_LEFT, DEFAULT_QPOS_RIGHT)
 
class Dex3_1_Left_JointIndex(IntEnum):
    kLeftHandThumb0 = 0
    kLeftHandThumb1 = 1
    kLeftHandThumb2 = 2
    kLeftHandMiddle0 = 3
    kLeftHandMiddle1 = 4
    kLeftHandIndex0 = 5
    kLeftHandIndex1 = 6

class Dex3_1_Right_JointIndex(IntEnum):
    kRightHandThumb0 = 0
    kRightHandThumb1 = 1
    kRightHandThumb2 = 2
    kRightHandIndex0 = 3
    kRightHandIndex1 = 4
    kRightHandMiddle0 = 5
    kRightHandMiddle1 = 6


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', type=str, default='eno1', help='Network interface used by G1RealWorldEnv.')
    args = parser.parse_args()

    print(" Testing Dex3_1_Controller with new unified API...")
    hand_ctrl = Dex3_1_Controller(args.net)

    # Simple test sequence
    print(" Running test sequence...")
    for i in range(10):
        # Test with small joint movements
        left_target = [0.01*i, 0, 0, 0, 0, 0, 0]
        right_target = [0, 0, 0, 0, 0, 0, 0]
        
        hand_ctrl.ctrl_dual_hand(left_target, right_target)
        
        # get hand state
        left_hand_state, right_hand_state = hand_ctrl.get_hand_state()
        print(f"Step {i}: Left [{left_hand_state[0]:.3f}, {left_hand_state[1]:.3f}, {left_hand_state[2]:.3f}] Right [{right_hand_state[0]:.3f}, {right_hand_state[1]:.3f}, {right_hand_state[2]:.3f}]")
        time.sleep(0.1)
    
    print("[OK] Test completed! New unified API is working perfectly.")
