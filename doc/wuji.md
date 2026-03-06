# Wuji Hand Setup Guide

This document describes how to prepare Wuji Hand for G1 teleoperation in this repository.

## 1. Print the G1-Wuji Adapter and Mount the Hand

First, 3D print the G1-Wuji adapter from [link](https://drive.google.com/file/d/1PKtpvaxZI7zmqRgvxg64AasXwEP1iWsf/view?usp=sharing), then mount the Wuji hand on the G1 robot.


## 2. Install Upgrader and Flash Firmware

Connect power to the hand, connect the hand to your PC with a USB cable, then follow the official Upgrader guide at [link](https://docs.wuji.tech/docs/zh/wuji-hand/latest/wuji-hand-upgrader-user-guide/) to install Upgrader and flash firmware.


## 3. Configure udev for Non-root USB Access

Run the following commands on Linux to allow non-root USB access:

```bash
echo 'SUBSYSTEM=="usb", ATTR{idVendor}=="0483", MODE="0666"' | \
sudo tee /etc/udev/rules.d/95-wujihand.rules && \
sudo udevadm control --reload-rules && \
sudo udevadm trigger
```


## 4. Query Serial Number and Update Startup Script

Query the device serial number with:

```bash
lsusb -v -d 0483:2000 | grep iSerial
```

Then set `hand_side` and `serial_number` in `wuji_hand_real.sh` before starting the controller.

For more Wuji Hand details, see the official documentation at [link](https://docs.wuji.tech/docs/zh/wuji-hand/latest/).
