"""
replay_extractor.py – Record & Extract Gameplay Frames

Features:
  - Record live capture card feed to video file
  - Extract frames from video at configurable FPS (for YOLO training data)
  - Preview replay files

Usage:
  python replay_extractor.py --record --device-index 1 --output replays/my_game.mp4
  python replay_extractor.py --extract --file replays/my_game.mp4 --output-dir frames/ --fps 2
"""

import cv2
import os
import time
import argparse
from video_input import VideoSource, SourceType


def record_gameplay(device_index: int, output_path: str,
                    width: int = 1920, height: int = 1080, fps: float = 30.0):
    """Record live gameplay from capture card to a video file."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    src = VideoSource(SourceType.CAPTURE_CARD, device_index=device_index)
    if not src.open():
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not writer.isOpened():
        print(f"Error: Cannot create output file '{output_path}'")
        src.release()
        return

    print(f"[RECORD] Recording to {output_path}... Press 'q' to stop.")
    frame_count = 0
    start = time.time()

    while True:
        frame, meta = src.get_frame()
        if frame is None:
            continue

        # Resize to target if needed
        if frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(frame, (width, height))

        writer.write(frame)
        frame_count += 1

        # Show preview
        elapsed = time.time() - start
        info = f"REC {elapsed:.0f}s | Frames: {frame_count}"
        preview = frame.copy()
        # Red recording dot
        cv2.circle(preview, (30, 30), 12, (0, 0, 255), -1)
        cv2.putText(preview, info, (50, 38), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2)
        cv2.imshow("Recording", cv2.resize(preview, (960, 540)))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    writer.release()
    src.release()
    cv2.destroyAllWindows()
    duration = time.time() - start
    print(f"[RECORD] Saved {frame_count} frames ({duration:.1f}s) to {output_path}")


def extract_frames(video_path: str, output_dir: str, target_fps: float = 2.0,
                   prefix: str = "frame"):
    """Extract frames from a video file at the target FPS for training data."""
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open '{video_path}'")
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, int(video_fps / target_fps))

    print(f"[EXTRACT] Video: {video_path}")
    print(f"  Video FPS: {video_fps:.1f} | Total frames: {total_frames}")
    print(f"  Extracting every {frame_interval} frames (~{target_fps:.1f} FPS)")
    print(f"  Output: {output_dir}/")

    frame_idx = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            filename = f"{prefix}_{saved:06d}.jpg"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, frame)
            saved += 1

            if saved % 50 == 0:
                print(f"  Saved {saved} frames...")

        frame_idx += 1

    cap.release()
    print(f"[EXTRACT] Done! Saved {saved} frames to {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay Recorder & Frame Extractor")

    parser.add_argument("--record", action="store_true",
                        help="Record capture card to video file")
    parser.add_argument("--extract", action="store_true",
                        help="Extract frames from video file")

    parser.add_argument("--device-index", type=int, default=0,
                        help="Capture card device index (for recording)")
    parser.add_argument("--file", type=str,
                        help="Input video file (for extraction)")
    parser.add_argument("--output", type=str, default="replays/gameplay.mp4",
                        help="Output video file path (for recording)")
    parser.add_argument("--output-dir", type=str, default="vision/dataset/images/train",
                        help="Output directory for extracted frames")
    parser.add_argument("--fps", type=float, default=2.0,
                        help="Target FPS for frame extraction (default: 2)")
    parser.add_argument("--width", type=int, default=1920,
                        help="Recording width")
    parser.add_argument("--height", type=int, default=1080,
                        help="Recording height")
    args = parser.parse_args()

    if args.record:
        record_gameplay(args.device_index, args.output, args.width, args.height)
    elif args.extract:
        if not args.file:
            print("Error: --file is required for extraction")
        else:
            extract_frames(args.file, args.output_dir, args.fps)
    else:
        parser.print_help()
