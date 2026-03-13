# MANUS Setup Guide

This sends the useful Manus data from the SDK to Python nodes, specifically `RawSkeletonInfo` is the fingertip data from MANUS and `Ergonomics` data is the approximate human skeleton joint angle data estimated by MANUS.

## 1. SDK Prerequisites

You'll need a hardware license key from Manus to use the [Ubuntu Standalone SDK](https://docs.manus-meta.com/2.4.0/Plugins/SDK/Linux/)

- Install [cppzmq](https://github.com/zeromq/cppzmq/tree/master). (This is a dependency for our version of the Manus SDK)
  ```bash
  sudo apt-get install libzmq3-dev
  download and unzip the lib, cd to directory
  mkdir build
  cd build
  cmake ..
  sudo make -j4 install
  ```
- Follow [Manus' instructions](https://docs.manus-meta.com/2.4.0/Plugins/SDK/Linux/) and install "Core Integrated" dependencies without Docker.
- Instead of compiling the default sdk you download from them, compile [our version of the SDK](../assets/MANUS_Core_SDK/) but use the same instructions. (Note -lzmq has been added to the makefile that we provide) `Make (Option 1)` works well.

## 2. Run MANUS SDK

Now to run our version of the MANUS SDK:

- Run the [C++ SDK](https://docs.manus-meta.com/2.4.0/Plugins/SDK/) with `./SDKClient_Linux.out`
- Pick standalone `1 Core Integrated` by pressing 1.
- The data should stream successfully via ZMQ. 