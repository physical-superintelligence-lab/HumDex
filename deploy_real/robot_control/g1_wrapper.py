import numpy as np
import time
import unitree_interface
from rich import print

ControllerMapping = {
    "A": 0x0100,
    "B": 0x0200,
    "X": 0x0400,
    "Y": 0x0800,
    "R1": 0x0001,
    "L1": 0x0002,
    "start": 0x0004,
    "select": 0x0008,
    "R2": 0x0010,
    "L2": 0x0020,
    "F1": 0x0040,
    "F2": 0x0080,
    "up": 0x1000,
    "right": 0x2000,
    "down": 0x4000,
    "left": 0x8000
    }

class G1RealWorldEnv:
    def __init__(self, net, config):
        
        self.config = config
        self.robot = unitree_interface.create_robot(net, unitree_interface.RobotType.G1, unitree_interface.MessageType.HG)
        self.running = True
        
        self.torque_limits = np.array([
                88, 139, 88, 139, 50, 50,
                88, 139, 88, 139, 50, 50,
                88, 50, 50,
                25, 25, 25, 25,
                25, 25, 25, 25,
            ])
        
        # Initializing process variables
        self.qj = np.zeros(config.num_actions, dtype=np.float32)
        self.dqj = np.zeros(config.num_actions, dtype=np.float32)
        self.tau_est = np.zeros(config.num_actions, dtype=np.float32)
        self.temperature = np.zeros((config.num_actions, 2), dtype=np.float32)
        self.voltage = np.zeros(config.num_actions, dtype=np.float32)
        self.action = np.zeros(config.num_actions, dtype=np.float32)
        self.target_dof_pos = config.default_angles.copy()
        self.counter = 0
        
        # Get robot configuration
        self.robot_config = self.robot.get_config()
        self.num_motors = self.robot.get_num_motors()
        print(f"Robot: {self.robot_config.name}")
        print(f"Motors: {self.num_motors}")
        print(f"Message type: {self.robot_config.message_type}")

        # Set control mode to PR (Pitch/Roll)
        self.robot.set_control_mode(unitree_interface.ControlMode.PR)
        control_mode = self.robot.get_control_mode()
        print(f"Control mode set to: {'PR' if control_mode == unitree_interface.ControlMode.PR else 'AB'}")

        # read the low state
        self.low_state = self.robot.read_low_state()
        self.low_cmd = self.robot.create_zero_command()
        print(f"Current robot state: {self.low_state}")
        print("[green]Successfully connected to the robot[/green]")

        self.controller_mapping = ControllerMapping
       
    
    def read_robot_state(self) -> unitree_interface.LowState:
        """Read current robot state"""
        return self.robot.read_low_state()

    def read_controller_input(self) -> unitree_interface.WirelessController:
        """Read wireless controller input"""
        controller = self.robot.read_wireless_controller()
        return controller

    def send_cmd(self, cmd: unitree_interface.MotorCommand):
        self.robot.write_low_command(cmd)


    def move_to_default_pos(self):
        print("[green]Waiting for the start signal to move to default pos...[/green]")
        while self.read_controller_input().keys != self.controller_mapping["start"]:
            time.sleep(self.config.control_dt)
        print("[green]Moving to default pos.[/green]")
        # move time 2s
        total_time = 2
        num_step = int(total_time / self.config.control_dt)
        
        # record the current pos
        qpos = self.get_robot_state()[0]
        tgt_qpos = self.config.default_angles.copy()
        # move to default pos
        for i in range(num_step):
            alpha = i / num_step
            interp_qpos = qpos * (1 - alpha) + tgt_qpos * alpha
            self.send_robot_action(interp_qpos)
            time.sleep(self.config.control_dt)

    def default_pos_state(self):
        print("Enter default pos state. Waiting for the Button A signal...")
        while self.read_controller_input().keys != self.controller_mapping["A"]:
            # keep the default pos
            default_pos = self.config.default_angles.copy()
            self.send_robot_action(default_pos)
            time.sleep(self.config.control_dt)
    
    def get_robot_state(self):
        self.counter += 1
        low_state = self.read_robot_state()
        # Get the current joint position and velocity
        for i in range(len(self.config.joint2motor_idx)):
            self.qj[i] = low_state.motor.q[self.config.joint2motor_idx[i]]
            self.dqj[i] = low_state.motor.dq[self.config.joint2motor_idx[i]]
            self.tau_est[i] = low_state.motor.tau_est[self.config.joint2motor_idx[i]]
            self.voltage[i] = low_state.motor.voltage[self.config.joint2motor_idx[i]]
            self.temperature[i] = low_state.motor.temperature[self.config.joint2motor_idx[i]]
            
        # imu_state quaternion: w, x, y, z
        quat = low_state.imu.quat.copy()
        ang_vel = np.array(low_state.imu.omega, dtype=np.float32)
        accel = np.array(low_state.imu.accel, dtype=np.float32) # [m/s^2] 
        dof_pos = self.qj.copy()
        dof_vel = self.dqj.copy()
        dof_temp = self.temperature.copy()
        dof_tau = self.tau_est.copy()
        dof_vol = self.voltage.copy()

        return (dof_pos, dof_vel, quat, ang_vel, dof_temp, dof_tau, dof_vol)
    

    def send_robot_action(self, target_dof_pos, kp_scale=1.0, kd_scale=1.0):
        
        # Read current state to get current positions for uncontrolled joints
        current_state = self.read_robot_state()
        cmd = self.robot.create_zero_command()
        
        cmd.q_target = target_dof_pos.copy()
        cmd.dq_target = np.zeros_like(target_dof_pos)
        kps = [self.config.kps[i] * kp_scale for i in range(len(self.config.kps))]
        kds = [self.config.kds[i] * kd_scale for i in range(len(self.config.kds))]
        cmd.kp = kps
        cmd.kd = kds
        cmd.tau_ff = np.zeros_like(target_dof_pos)
        self.send_cmd(cmd)
        return
        


    def close(self):
        exit()

if __name__ == "__main__":
    # example usage
    from config import Config
    config = Config("configs/g1.yaml")
    env = G1RealWorldEnv(net="enp0s31f6", config=config)
    # measure fps
    start_time = time.time()
    while True:
        state = env.get_robot_state()
        controller = env.read_controller_input()
        # for i in range(len(state[0])):
        #     print(f"dof {i}: {state[0][i]:.2f}", end=" ")
        # print("--------------------------------")
      
        env.move_to_default_pos()

        if controller.keys:
            print(f"keys: {controller.keys}")
            print(f"Controller: L_stick=[{controller.lx:.2f}, {controller.ly:.2f}]")
            if controller.keys == env.controller_mapping["A"]:
               print("A button pressed")
            elif controller.keys == env.controller_mapping["B"]:
                print("B button pressed")
            elif controller.keys == env.controller_mapping["X"]:
                print("X button pressed")
            elif controller.keys == env.controller_mapping["Y"]:
                print("Y button pressed")
            else:
                print("No button pressed")
        end_time = time.time()
        fps = 1 / (end_time - start_time)
        print(f"FPS: {fps:.2f}")
        start_time = end_time
        time.sleep(0.1)