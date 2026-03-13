# Mocap Instructions
In this folder, we provide manus gloves mocap code and a very simple vision based mocap system based on mediapipe for code debugging and quick-start purpose only. DO NOT use Mediapipe (+GeoRT) for real teleop! Mediapipe is very unreliable under wrist movement. The hand scale is constantly changing throughout deployment and this will feed out-of-distribution keypoints to GeoRT!!

## Our Vision-based Mocap Recommendations:
Please use a mocap glove. If you insist on using a vision-based mocap system, consider:
1. Use a more advanced 3D hand detector (e.g. Hamer).
2. Probably mount the camera on your wrist in a fixed, standard way through training and deployment.


## Manus Gloves Mocap.
GeoRT works best with the glove-based mocap. We provide an example based on the Manus gloves.

### Installation
We need to get Manus gloves and its mocap server installed on a separate windows laptop. Make sure that windows laptop and the host is in the same LAN an ping each other. Then, follows the readme in the manus_client folder to setup. This will build a ROS2 node that can 

After this step, run the following for the right hand in one terminal and proceed to the next section.
```
ros2 run manus_client manus_right
```

This will create a pipe: windows --> ROS2_CPP_BROADCAST.
### Deployment

In one terminal, run
```
python ./training/mocap/manus_mocap_core.py
```
this will start a ROS2 process that fetches data from Manus server on the windows laptop and broadcasts it on the localhost (this process is just a connector to decouple ROS from code, and the pipeline now becomes windows --> ROS2_CPP_BROADCAST --> localhost:PORT BROADCAST)

Then, in another terminal, run
```
python ./training/mocap/manus_evaluation.py -hand YOUR_ROBOT_HAND_IN_CONFIG -ckpt_tag YOUR_CKPT
```
This will fetch the data on the localhost:PORT. After launching this, you will be able to do a virtual manus teleop in the pop-up viewer as shown in the Readme teaser.


