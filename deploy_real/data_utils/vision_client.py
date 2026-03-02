#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import zmq
import numpy as np
import time
import cv2
import io
from multiprocessing import shared_memory
from collections import deque
from rich import print
import struct
import pickle

class VisionClient:
    def __init__(
        self,
        server_address="127.0.0.1",
        port=5555,
        
        img_shape=None,
        img_shm_name=None,
        depth_shape=None,
        depth_shm_name=None,
        
        unit_test=False,
        image_show=False,
        depth_show=False,
    ):
        """
        A ZeroMQ subscriber client for receiving JPEG-compressed RGB and depth data from a single camera.

        Parameters:
        -----------
        server_address : str
            The IP address or hostname where the server is running.
        port : int
            The ZeroMQ PUB socket port used by the server.
        img_shape : tuple or None
            Shape (H, W, C) for the RGB image shared memory array (optional).
            Default resolution is 480x640 (Height x Width).
        img_shm_name : str or None
            Shared memory name for RGB image (optional).
        depth_shape : tuple or None
            Shape (H, W) for depth image shared memory array (optional).
            Default resolution is 480x640.
        depth_shm_name : str or None
            Shared memory name for depth image (optional).
        unit_test : bool
            If True, enables a simple FPS/performance print-out.
        image_show : bool
            If True, displays the RGB image in a pop-up window.
        depth_show : bool
            If True, uses OpenCV to visualize the depth image in real-time.
            
        Note:
        ----
        This client expects JPEG-compressed image data with the format:
        [4-byte width][4-byte height][4-byte JPEG length][JPEG data]
        """
        self.server_address = server_address
        self.port = port
        self.running = True
        
        self.image_show = image_show
        self.depth_show = depth_show
        
        # Optional shared memory for RGB image (single camera)
        self.img_shape = img_shape
        self.img_shm_name = img_shm_name
        self.img_shm_enabled = False
        if (self.img_shape is not None) and (self.img_shm_name is not None):
            self.img_shm = shared_memory.SharedMemory(name=self.img_shm_name)
            self.img_array = np.ndarray(self.img_shape, dtype=np.uint8, buffer=self.img_shm.buf)
            self.img_shm_enabled = True

        # Optional shared memory for depth image (single camera)
        self.depth_shape = depth_shape
        self.depth_shm_name = depth_shm_name
        self.depth_shm_enabled = False
        if (self.depth_shape is not None) and (self.depth_shm_name is not None):
            self.depth_shm = shared_memory.SharedMemory(name=self.depth_shm_name)
            self.depth_array = np.ndarray(self.depth_shape, dtype=np.float32, buffer=self.depth_shm.buf)
            self.depth_shm_enabled = True

        # Simple performance metrics if needed
        self.unit_test = unit_test
        if self.unit_test:
            self._init_performance_metrics()

    def _init_performance_metrics(self):
        """Initialize a simple set of performance metrics (FPS, etc.)."""
        self.frame_count = 0
        self.start_time = time.time()
        self.time_window = 1.0
        self.frame_times = deque()

    def _update_performance_metrics(self, print_info, verbose=False):
        """Update and print performance metrics (like FPS)."""
        if not self.unit_test:
            return

        now = time.time()
        self.frame_times.append(now)
        while self.frame_times and self.frame_times[0] < now - self.time_window:
            self.frame_times.popleft()
        self.frame_count += 1

        # Print every 30 frames
        if self.frame_count % 30 == 0:
            real_time_fps = len(self.frame_times) / self.time_window
            elapsed = now - self.start_time
            if verbose:
                print(f"[VisionClient] FPS: {real_time_fps:.2f}, Frames: {self.frame_count}, Elapsed: {elapsed:.2f}s")
                print(print_info, "\n")

    def handle_color_image(self, color_img):
        """
        Handle the RGB image from single camera.
        - Optionally display it
        - Optionally copy to shared memory
        """
        if color_img is None:
            return

        # If shared memory is enabled for RGB images, copy it over
        if self.img_shm_enabled and self.img_shape is not None:
            if color_img.shape == self.img_shape:
                np.copyto(self.img_array, color_img)
            else:
                # If shape doesn't match exactly, you might need a crop/resizing.
                h, w = self.img_shape[0], self.img_shape[1]
                np.copyto(self.img_array, color_img[:h, :w])

        # If you want to display the image
        if self.image_show:
            # Convert RGB to BGR for OpenCV display (similar to example_vision_client.py)
            display_img = color_img.copy()
            if len(display_img.shape) == 3 and display_img.shape[2] == 3:
                # Assume RGB format and convert to BGR for OpenCV
                # display_img = cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR)
                pass
            
            cv2.imshow("VisionClient - RGB", display_img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.running = False

    def handle_depth_image(self, depth_img):
        """
        Handle the depth image from single camera.
        - Optionally copy to a shared memory region
        - Optionally visualize with OpenCV if depth_show is True.
        """
        if depth_img is None:
            return

        # If shared memory is enabled for depth images, copy it
        if self.depth_shm_enabled and self.depth_shape is not None:
            if depth_img.shape == self.depth_shape:
                np.copyto(self.depth_array, depth_img)
            # else: adapt cropping/resizing as needed

        # Real-time depth visualization using simplified approach from example_vision_client.py
        if self.depth_show:
            # Handle different depth formats
            temp = depth_img.copy()
            
            # If the image has 3 channels, extract first channel
            if temp.ndim == 3:
                if temp.shape[2] == 1:
                    temp = np.squeeze(temp, axis=2)
                else:
                    temp = temp[:, :, 0]
            
            # Use simplified depth visualization approach (similar to example_vision_client.py)
            max_depth = 5000  # Maximum depth value for normalization
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(temp, alpha=255.0/max_depth), 
                cv2.COLORMAP_JET
            )
            
            cv2.imshow("VisionClient - Depth", depth_colormap)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.running = False
    
    def _close(self):
        """Close the ZeroMQ socket and any windows."""
        self.socket.close()
        self.context.term()
        if self.image_show or self.depth_show:
            cv2.destroyAllWindows()
        print("[VisionClient] Closed.")

    def receive_process(self):
        """
        Main loop for single camera:
        - Connect to the server (PUB socket)
        - Continuously receive JPEG-compressed data
        - Decode JPEG and handle RGB and depth images
        """
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.connect(f"tcp://{self.server_address}:{self.port}")
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")

        print(f"[VisionClient] Subscribed to tcp://{self.server_address}:{self.port}. Waiting for data...")

        verbose = False
        try:
            while self.running:
                try:
                    start_time = time.time()

                    # Receive message.
                    message = self.socket.recv(zmq.NOBLOCK)
                
                    if len(message) < 12:  # At least 12 bytes are required for header.
                        continue
                    
                    # Parse header: [width][height][jpeg_length]
                    width = struct.unpack('i', message[0:4])[0]
                    height = struct.unpack('i', message[4:8])[0]
                    jpeg_length = struct.unpack('i', message[8:12])[0]
                    
                    # Validate JPEG payload length.
                    actual_jpeg_size = len(message) - 12
                    
                    if actual_jpeg_size != jpeg_length:
                        if verbose:
                            print(f"[WARN] JPEG size mismatch: expected {jpeg_length}, got {actual_jpeg_size}")
                        continue
                    
                    # Extract and decode JPEG payload.
                    jpeg_data = message[12:]
                    
                    try:
                        # Decode JPEG bytes with OpenCV.
                        image = cv2.imdecode(np.frombuffer(jpeg_data, dtype=np.uint8), cv2.IMREAD_COLOR)
                        
                        if image is None:
                            if verbose:
                                print("[WARN] Failed to decode JPEG image")
                            continue
                            
                        # Validate decoded image size.
                        if image.shape[0] != height or image.shape[1] != width:
                            if verbose:
                                print(f"[WARN] Decoded image size {image.shape} doesn't match expected {height}x{width}")
                            # Continue anyway since JPEG decoding can introduce slight size differences.
                        
                    except Exception as e:
                        if verbose:
                            print(f"[WARN] JPEG decode error: {e}")
                        continue
                    
                    # # Check what data we have
                    # have_color_data = 'rgb' in data and data['rgb'] is not None
                    # have_depth_data = 'depth' in data and data['depth'] is not None
                
                    have_color_data = True
                    have_depth_data = False
                    
                    if verbose:
                        print(f"==> JPEG decoded image.shape: {image.shape}, original JPEG size: {jpeg_length} bytes")
                    
                    # Process data
                    if have_color_data:
                        self.handle_color_image(image)
                    # if have_depth_data:
                    #     self.handle_depth_image(image_data)
                  
                    end_time = time.time()
                    loop_fps = 1.0 / (end_time - start_time)
                    
                    print_info = f"rgb: {have_color_data} | depth: {have_depth_data} | fps: {loop_fps:.2f} | jpeg_size: {jpeg_length}B"
                    self._update_performance_metrics(print_info)

                except zmq.Again:
                    # No new message, keep polling.
                    time.sleep(0.001)
                    continue

        except KeyboardInterrupt:
            print("[VisionClient] Interrupted by user.")
        except Exception as e:
            print(f"[VisionClient] [ERROR] {e}")
        finally:
            self._close()


if __name__ == "__main__":
    # Example usage for single camera (640x480):
    from multiprocessing import shared_memory, Array, Lock
    import threading
    
    # Create shared memory for single camera - 640x480 image
    num_cameras = 2
    image_shape = (360, 640 * num_cameras, 3)  # Height, Width, Channels for OpenCV format
    image_shared_memory = shared_memory.SharedMemory(create=True, size=int(np.prod(image_shape) * np.uint8().itemsize * num_cameras))
    image_array = np.ndarray(image_shape, dtype=np.uint8, buffer=image_shared_memory.buf)
    
    # Create shared memory for depth - same resolution as RGB
    depth_shape = (360, 640)
    depth_shared_memory = shared_memory.SharedMemory(create=True, size=int(np.prod(depth_shape) * np.float32().itemsize))
    depth_array = np.ndarray(depth_shape, dtype=np.float32, buffer=depth_shared_memory.buf)
    
    # Display settings for single camera
    image_show = True
    depth_show = True

    client = VisionClient(
        server_address="192.168.123.164",  # or "127.0.0.1"
        port=5556,
        img_shape=image_shape,
        img_shm_name=image_shared_memory.name,
        depth_shape=depth_shape,
        depth_shm_name=depth_shared_memory.name,
        image_show=image_show,
        depth_show=depth_show,
        unit_test=True
    )
    
    vision_thread = threading.Thread(target=client.receive_process, daemon=True)
    vision_thread.start()
    
    # Main loop
    try:
        frame_count = 0
        while True:
            time.sleep(0.01)
            frame_count += 1
            
            # Access the shared memory arrays if needed
            # color_data = image_array.copy()
            # depth_data = depth_array.copy()
            
    except KeyboardInterrupt:
        print("[VisionClient] Interrupted by user.")
    except Exception as e:
        print(f"[VisionClient] [ERROR] {e}")
    finally:
        # Clean up shared memory
        image_shared_memory.unlink()
        depth_shared_memory.unlink()
        
        image_shared_memory.close()
        depth_shared_memory.close()

    