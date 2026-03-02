#!/usr/bin/env python3
"""
CLI script to visualize predicted 20D hand actions.

Usage:
    python visualize_hand_actions.py \
        --actions predicted_hand.npy \
        --hand_side left \
        --output hand_visualization.mp4
"""

import argparse
import numpy as np
from pathlib import Path

from visualizers import HandVisualizer, get_default_paths


def main():
    defaults = get_default_paths()
    
    parser = argparse.ArgumentParser(
        description='Visualize predicted 20D hand actions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Visualize left hand
    python visualize_hand_actions.py \\
        --actions predicted_left_hand.npy \\
        --hand_side left \\
        --output left_hand_viz.mp4

    # Visualize right hand
    python visualize_hand_actions.py \\
        --actions predicted_right_hand.npy \\
        --hand_side right \\
        --output right_hand_viz.mp4

    # Custom XML path
    python visualize_hand_actions.py \\
        --actions predicted_hand.npy \\
        --hand_side left \\
        --xml /path/to/left.xml \\
        --output viz.mp4
        """
    )

    parser.add_argument('--actions', type=str, required=True,
                        help='Path to .npy file with (T, 20) or (T, 5, 4) actions')
    parser.add_argument('--hand_side', type=str, required=True, choices=['left', 'right'],
                        help='Hand side: left or right')
    parser.add_argument('--xml', type=str, default=None,
                        help='Path to hand MuJoCo XML (auto-detected if not provided)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output video path')
    parser.add_argument('--fps', type=int, default=30,
                        help='Video FPS (default: 30)')
    parser.add_argument('--width', type=int, default=640,
                        help='Rendering width (default: 640)')
    parser.add_argument('--height', type=int, default=480,
                        help='Rendering height (default: 480)')

    args = parser.parse_args()

    # Load actions
    print(f"Loading hand actions: {args.actions}")
    actions = np.load(args.actions)
    print(f"  Shape: {actions.shape}")

    # Validate action dimensions
    if actions.ndim == 1:
        if len(actions) != 20:
            raise ValueError(f"Expected 20D action, got {len(actions)}D")
        actions = actions.reshape(1, -1)
    elif actions.ndim == 2:
        if actions.shape[1] != 20:
            raise ValueError(f"Expected (T, 20) actions, got {actions.shape}")
    elif actions.ndim == 3:
        if actions.shape[1:] != (5, 4):
            raise ValueError(f"Expected (T, 5, 4) actions, got {actions.shape}")
        actions = actions.reshape(len(actions), -1)
    else:
        raise ValueError(f"Invalid action shape: {actions.shape}")

    # Find hand XML file
    if args.xml is not None:
        hand_xml = args.xml
    else:
        key = f"{args.hand_side}_hand_xml"
        hand_xml = defaults[key]

    # Verify XML exists
    if not Path(hand_xml).exists():
        print(f"[ERROR] Error: XML file not found: {hand_xml}")
        print("\nPlease specify --xml <path> or ensure the file exists")
        return 1

    # Create visualizer
    viz = HandVisualizer(
        xml_path=hand_xml,
        hand_side=args.hand_side,
        width=args.width,
        height=args.height
    )

    # Visualize
    frames = viz.visualize(
        actions,
        output_video=args.output,
        fps=args.fps
    )
    
    return 0


if __name__ == "__main__":
    exit(main())
