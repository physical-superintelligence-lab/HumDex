import argparse
import os
import subprocess
from pathlib import Path
from typing import List, Tuple


def _natural_sorted(paths: List[Path]) -> List[Path]:
    # Filenames are typically zero-padded (000000.jpg) so lexicographic sort is fine.
    return sorted(paths, key=lambda p: p.name)


def _find_episode_rgb_dirs(root: Path) -> List[Tuple[str, Path]]:
    """
    Returns list of (episode_name, rgb_dir) found under:
      root/episode_XXXX/rgb
    """
    out: List[Tuple[str, Path]] = []
    if not root.exists():
        return out
    for ep_dir in sorted(root.glob("episode_*")):
        if not ep_dir.is_dir():
            continue
        rgb_dir = ep_dir / "rgb"
        if rgb_dir.is_dir():
            out.append((ep_dir.name, rgb_dir))
    return out


def _write_video_from_images(image_paths: List[Path], out_path: Path, fps: float) -> None:
    if len(image_paths) == 0:
        raise ValueError("No images to write.")

    # Prefer OpenCV if available; otherwise fallback to ffmpeg (often available on servers).
    try:
        import cv2  # type: ignore
    except ModuleNotFoundError:
        _write_video_with_ffmpeg_concat(image_paths=image_paths, out_path=out_path, fps=fps)
        return

    first = cv2.imread(str(image_paths[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise ValueError(f"Failed to read first image: {image_paths[0]}")
    h, w = first.shape[:2]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # mp4v is widely available; users can re-encode later if needed.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, float(fps), (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for: {out_path}")

    try:
        for p in image_paths:
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"Failed to read image: {p}")
            if img.shape[0] != h or img.shape[1] != w:
                img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
            writer.write(img)
    finally:
        writer.release()


def _write_video_with_ffmpeg_concat(image_paths: List[Path], out_path: Path, fps: float) -> None:
    """
    Write MP4 using ffmpeg concat demuxer so we can handle missing frames and mixed filenames robustly.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    list_path = out_path.with_suffix(out_path.suffix + ".ffmpeg_list.txt")

    # ffmpeg concat format: each line "file '/abs/path/to/img.jpg'"
    with open(list_path, "w") as f:
        for p in image_paths:
            f.write(f"file '{str(p)}'\n")

    cmd = [
        "ffmpeg",
        "-y",  # overwrite
        "-hide_banner",
        "-loglevel",
        "error",
        "-r",
        str(float(fps)),  # input read rate for concat
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_path),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(out_path),
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {out_path}:\n{res.stderr}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert episode_XXXX/rgb image sequences to MP4 videos for one or more roots."
    )
    parser.add_argument(
        "--roots",
        type=str,
        nargs="+",
        required=True,
        help="One or more humdex_demonstration/<timestamp> directories containing episode_XXXX/rgb/",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Output video FPS (default: 30).",
    )
    parser.add_argument(
        "--glob",
        type=str,
        default="*.jpg",
        help="Image glob under rgb/ (default: *.jpg). You can use '*.png' or '*.*'.",
    )
    parser.add_argument(
        "--output_subdir",
        type=str,
        default="videos",
        help="Where to store videos under each root (default: videos).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing videos if present.",
    )
    args = parser.parse_args()

    roots = [Path(r).expanduser().resolve() for r in args.roots]
    total_eps = 0
    for root in roots:
        episodes = _find_episode_rgb_dirs(root)
        if len(episodes) == 0:
            print(f"[WARN] No episode_*/rgb found under: {root}")
            continue

        out_dir = root / args.output_subdir
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nRoot: {root}")
        for ep_name, rgb_dir in episodes:
            img_paths = _natural_sorted(list(rgb_dir.glob(args.glob)))
            if len(img_paths) == 0 and args.glob != "*.*":
                # fallback: include all common image suffixes
                img_paths = _natural_sorted(
                    [p for p in rgb_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]
                )
            if len(img_paths) == 0:
                print(f"  - {ep_name}: [SKIP] no images in {rgb_dir}")
                continue

            out_path = out_dir / f"{ep_name}.mp4"
            if out_path.exists() and not args.overwrite:
                print(f"  - {ep_name}: exists -> {out_path} (use --overwrite to regenerate)")
                total_eps += 1
                continue

            print(f"  - {ep_name}: {len(img_paths)} frames -> {out_path}")
            _write_video_from_images(img_paths, out_path, fps=args.fps)
            total_eps += 1

    print(f"\nDone. Produced/kept videos for {total_eps} episodes.")


if __name__ == "__main__":
    main()


