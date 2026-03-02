#!/usr/bin/env python3
"""
RealSense -> JPEG -> ZeroMQ PUB server.

This server publishes messages compatible with deploy_real/data_utils/vision_client.py (VisionClient):
Message format (bytes):
    [int32 width][int32 height][int32 jpeg_length][jpeg_bytes]

Typical usage on g1:
  source ~/miniconda3/bin/activate twist2
  cd ~/TWIST2/deploy_real
  python server_realsense_zmq_pub.py --bind 0.0.0.0 --port 5555 --width 640 --height 480 --fps 30
"""

import argparse
import struct
import time

import cv2
import numpy as np
import zmq
from rich import print


def parse_args():
    p = argparse.ArgumentParser(description="Publish RealSense color frames as JPEG over ZeroMQ (PUB).")
    p.add_argument("--bind", default="0.0.0.0", help="Bind address, e.g. 0.0.0.0")
    p.add_argument("--port", default=5555, type=int, help="Bind port, e.g. 5555")
    p.add_argument("--width", default=640, type=int, help="Color width")
    p.add_argument("--height", default=480, type=int, help="Color height")
    p.add_argument("--fps", default=30, type=int, help="Camera FPS")
    p.add_argument("--jpeg_quality", default=80, type=int, help="JPEG quality (1-100)")
    p.add_argument("--print_every", default=60, type=int, help="Print stats every N frames")
    p.add_argument("--rs_serial", default="", help="Optional RealSense serial to select device")
    return p.parse_args()


def main():
    args = parse_args()

    try:
        import pyrealsense2 as rs  # type: ignore
    except Exception as e:
        raise SystemExit(
            "[ERROR] pyrealsense2 is not installed.\n"
            "If you publish RealSense ZMQ stream directly on g1, install pyrealsense2 first.\n"
            "If you switch to another camera publisher, keep output format compatible with VisionClient."
        ) from e

    # ZMQ PUB
    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUB)
    sock.bind(f"tcp://{args.bind}:{args.port}")

    # RealSense
    pipeline = rs.pipeline()
    config = rs.config()
    if args.rs_serial:
        config.enable_device(args.rs_serial)
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    profile = pipeline.start(config)

    device = profile.get_device()
    dev_name = device.get_info(rs.camera_info.name) if device else "unknown"
    dev_sn = device.get_info(rs.camera_info.serial_number) if device else "unknown"

    print("=" * 70)
    print("RealSense ZMQ PUB Server")
    print("=" * 70)
    print(f"Device: {dev_name}  SN: {dev_sn}")
    print(f"Bind: tcp://{args.bind}:{args.port}")
    print(f"Stream: {args.width}x{args.height}@{args.fps}  JPEG Q={args.jpeg_quality}")
    print("Press Ctrl+C to quit")
    print("=" * 70)

    frame_idx = 0
    t0 = time.time()
    last_print = t0

    try:
        while True:
            frames = pipeline.wait_for_frames(timeout_ms=2000)
            color = frames.get_color_frame()
            if not color:
                continue

            img = np.asanyarray(color.get_data())  # BGR, shape (H, W, 3)
            h, w = img.shape[:2]

            ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpeg_quality)])
            if not ok:
                continue
            jpeg_bytes = enc.tobytes()

            header = struct.pack("iii", int(w), int(h), int(len(jpeg_bytes)))
            msg = header + jpeg_bytes

            # PUB send succeeds even without subscribers; dropped messages are expected.
            sock.send(msg)

            frame_idx += 1
            if args.print_every > 0 and frame_idx % args.print_every == 0:
                now = time.time()
                fps = args.print_every / (now - last_print)
                last_print = now
                mbps = (len(msg) * fps) / (1024 * 1024)
                print(f"[ZMQ PUB] frames={frame_idx}  fps={fps:.1f}  jpeg={len(jpeg_bytes)/1024:.1f}KB  ~{mbps:.1f}MB/s")

    except KeyboardInterrupt:
        print("\n[STOP] Interrupted, shutting down...")
    finally:
        try:
            pipeline.stop()
        except Exception:
            pass
        try:
            sock.close(0)
            ctx.term()
        except Exception:
            pass


if __name__ == "__main__":
    main()


