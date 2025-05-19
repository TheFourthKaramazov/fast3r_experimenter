

import os
import sys
import argparse
import numpy as np
import cv2


def extract_frames(video_path: str, num_frames: int) -> None:
    """Save `num_frames` evenly spaced frames from `video_path`."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open {video_path}", file=sys.stderr)
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        print(f"[ERROR] {video_path} has no frames.", file=sys.stderr)
        return

    # Choose frame indices (linspace gives us evenly-spaced integers)
    indices = np.linspace(0, total_frames - 1,
                          min(num_frames, total_frames), dtype=int)

    # Output directory:  "<video-name>_frames" (spaces → underscores)
    base = os.path.splitext(os.path.basename(video_path))[0]
    out_dir = os.path.join(os.path.dirname(video_path),
                           f"{base.replace(' ', '_')}_frames")
    os.makedirs(out_dir, exist_ok=True)

    for i, frame_idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ret, frame = cap.read()
        if not ret:
            print(f"[WARN] Could not read frame {frame_idx} of {video_path}")
            continue
        out_path = os.path.join(out_dir, f"frame_{i:04d}.png")
        cv2.imwrite(out_path, frame)

    cap.release()
    print(f"[OK] Saved {len(indices)} frames to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract frames from videos")
    parser.add_argument("videos", nargs="+", help="Input .mp4 files")
    parser.add_argument("--num-frames", "-n", type=int, default=124,
                        help="Number of frames to save (default 124)")
    args = parser.parse_args()

    for vid in args.videos:
        extract_frames(os.path.expanduser(vid), args.num_frames)


if __name__ == "__main__":
    main()
