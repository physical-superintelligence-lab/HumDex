# VDMocap & VDHand Setup Guide

This document describes how to prepare VDMocap/VDHand for teleoperation in this repository.

## 1. Purchase and Wearing

### 1.1 Purchase Link

VDMocap/VDHand product page:

- https://vdsuit.com/h-pd-34.html

### 1.2 Wearing Guide

Please wear sensors and gloves according to the official wearing tutorial video.

- Wearing video: [YouTube](https://youtu.be/1urOMXukJVc)


## 2. Windows Software Installation and Setup

Use a **Windows PC** and connect the official hardware through USB.

### 2.1 Install Wi-Fi Driver

Install the official Wi-Fi driver on the Windows PC first.

Download it from [link](https://drive.google.com/file/d/1SCIezEEy2k8YOBB9VjTMO7oKLXqWLQaV/view?usp=sharing), then extract the package and run the installer.

### 2.2 Install DreamsCapStudio

Install DreamsCapStudio after the driver setup is complete.

Download it from [link](https://drive.google.com/file/d/1480iz0yccpRhxriPuIlsLNuUNTlIVk-L/view?usp=sharing), then extract the package and launch `DreamsCapStudio`.

### 2.3 Power On, Connect, and Calibrate

1. Power on the devices.
2. Open `DreamsCapStudio`.
3. Click **Connect**.
4. Run **Calibration**.


<img src="../assets/vdmocap/software.jpg" alt="DreamsCapStudio connect and calibration" width="720" />

### 2.4 Enable Data Broadcast (IP/Port)

1. In `DreamsCapStudio`, click the top-right broadcast icon.
2. Configure **IP** and **Port**.
3. Click **Open** to start broadcast.

Network requirement:

- Ensure the Linux machine and Windows machine are reachable on the same network
  - same Wi-Fi
  - wired LAN with reachable IPs


<img src="../assets/vdmocap/data_share.jpg" alt="DreamsCapStudio data share settings" width="720" />


## 3. Map Settings to `teleop.yaml`

After setting broadcast IP/Port in DreamsCapStudio, sync them into:

- `deploy_real/config/teleop.yaml` (`network.mocap` section)