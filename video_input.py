"""
video_input.py – Unified Video Source for League Bot

Provides a single interface for reading frames from:
  - Replay video files (.mp4, .webm, .avi)
  - USB capture card (HDMI dongle)
  - DXGI shared memory bridge (same-PC mode)

Usage:
  python video_input.py --list-devices
  python video_input.py --test-capture --device-index 1
  python video_input.py --test-replay --file path/to/clip.mp4
"""

import cv2
import numpy as np
import mmap
import time
import argparse
from dataclasses import dataclass
from typing import Optional, Tuple
from enum import Enum


class SourceType(Enum):
    REPLAY = "replay"
    CAPTURE_CARD = "capture_card"
    SHARED_MEMORY = "shared_memory"


@dataclass
class FrameMetadata:
    """Metadata returned with each frame."""
    width: int
    height: int
    timestamp: float          # seconds since start
    frame_number: int
    source: SourceType
    fps: float = 0.0


class VideoSource:
    """
    Unified video source. Abstracts away whether we're reading from
    a replay file, a USB capture card, or DXGI shared memory.
    """

    def __init__(self, source_type: SourceType, **kwargs):
        self.source_type = source_type
        self._cap: Optional[cv2.VideoCapture] = None
        self._shm: Optional[mmap.mmap] = None
        self._frame_number = 0
        self._start_time = time.time()
        self._fps = 0.0
        self._width = 0
        self._height = 0

        # Source-specific config
        self._replay_file = kwargs.get("replay_file", None)
        self._device_index = kwargs.get("device_index", 0)
        self._shm_name = kwargs.get("shm_name", "Global\\LoLBotFrame")
        self._shm_size = kwargs.get("shm_size", 1024 * 1024 * 64)
        self._target_width = kwargs.get("target_width", 1920)
        self._target_height = kwargs.get("target_height", 1080)
        self._playback_speed = kwargs.get("playback_speed", 1.0)

    def open(self) -> bool:
        """Open the video source. Returns True on success."""
        if self.source_type == SourceType.REPLAY:
            return self._open_replay()
        elif self.source_type == SourceType.CAPTURE_CARD:
            return self._open_capture_card()
        elif self.source_type == SourceType.SHARED_MEMORY:
            return self._open_shared_memory()
        return False

    def get_frame(self) -> Tuple[Optional[np.ndarray], Optional[FrameMetadata]]:
        """
        Read the next frame from the source.
        Returns (frame_bgr, metadata) or (None, None) if no frame available.
        """
        if self.source_type == SourceType.REPLAY:
            return self._read_replay()
        elif self.source_type == SourceType.CAPTURE_CARD:
            return self._read_capture_card()
        elif self.source_type == SourceType.SHARED_MEMORY:
            return self._read_shared_memory()
        return None, None

    def release(self):
        """Release all resources."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        if self._shm is not None:
            self._shm.close()
            self._shm = None

    @property
    def is_open(self) -> bool:
        if self.source_type in (SourceType.REPLAY, SourceType.CAPTURE_CARD):
            return self._cap is not None and self._cap.isOpened()
        elif self.source_type == SourceType.SHARED_MEMORY:
            return self._shm is not None
        return False

    # ─── Replay File ──────────────────────────────────────────────

    def _open_replay(self) -> bool:
        if not self._replay_file:
            print("Error: No replay file specified.")
            return False
        self._cap = cv2.VideoCapture(self._replay_file)
        if not self._cap.isOpened():
            print(f"Error: Cannot open replay file '{self._replay_file}'")
            return False
        self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total / self._fps if self._fps > 0 else 0
        print(f"[REPLAY] Opened: {self._replay_file}")
        print(f"  Resolution: {self._width}x{self._height} | FPS: {self._fps:.1f} | "
              f"Duration: {duration:.1f}s | Frames: {total}")
        self._start_time = time.time()
        return True

    def _read_replay(self) -> Tuple[Optional[np.ndarray], Optional[FrameMetadata]]:
        if self._cap is None:
            return None, None

        # Control playback speed
        if self._playback_speed > 0 and self._fps > 0:
            expected_time = self._frame_number / (self._fps * self._playback_speed)
            elapsed = time.time() - self._start_time
            if elapsed < expected_time:
                time.sleep(expected_time - elapsed)

        ret, frame = self._cap.read()
        if not ret:
            print("[REPLAY] End of replay file.")
            return None, None

        self._frame_number += 1
        meta = FrameMetadata(
            width=frame.shape[1],
            height=frame.shape[0],
            timestamp=self._frame_number / self._fps if self._fps > 0 else 0,
            frame_number=self._frame_number,
            source=SourceType.REPLAY,
            fps=self._fps,
        )
        return frame, meta

    # ─── Capture Card ─────────────────────────────────────────────

    def _open_capture_card(self) -> bool:
        print(f"[CAPTURE CARD] Opening device index {self._device_index}...")
        # Use DirectShow backend on Windows for better capture card support
        self._cap = cv2.VideoCapture(self._device_index, cv2.CAP_DSHOW)
        if not self._cap.isOpened():
            # Fallback to default backend
            self._cap = cv2.VideoCapture(self._device_index)

        if not self._cap.isOpened():
            print(f"Error: Cannot open capture card at device index {self._device_index}")
            return False

        # Try to set resolution to 1080p
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._target_width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._target_height)

        self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0

        print(f"[CAPTURE CARD] Opened: device {self._device_index}")
        print(f"  Resolution: {self._width}x{self._height} | FPS: {self._fps:.1f}")
        self._start_time = time.time()
        return True

    def _read_capture_card(self) -> Tuple[Optional[np.ndarray], Optional[FrameMetadata]]:
        if self._cap is None:
            return None, None
        ret, frame = self._cap.read()
        if not ret:
            return None, None
        self._frame_number += 1
        meta = FrameMetadata(
            width=frame.shape[1],
            height=frame.shape[0],
            timestamp=time.time() - self._start_time,
            frame_number=self._frame_number,
            source=SourceType.CAPTURE_CARD,
            fps=self._fps,
        )
        return frame, meta

    # ─── DXGI Shared Memory ───────────────────────────────────────

    def _open_shared_memory(self) -> bool:
        try:
            self._shm = mmap.mmap(
                -1, self._shm_size,
                tagname=self._shm_name,
                access=mmap.ACCESS_READ
            )
            self._width = self._target_width
            self._height = self._target_height
            print(f"[SHM] Connected to shared memory: {self._shm_name}")
            print(f"  Expected resolution: {self._width}x{self._height}")
            self._start_time = time.time()
            return True
        except FileNotFoundError:
            print(f"Error: Shared memory '{self._shm_name}' not found. "
                  "Run the C++ capturer first!")
            return False

    def _read_shared_memory(self) -> Tuple[Optional[np.ndarray], Optional[FrameMetadata]]:
        if self._shm is None:
            return None, None
        try:
            w, h = self._width, self._height
            self._shm.seek(0)
            frame_bytes = self._shm.read(w * h * 4)
            frame = np.frombuffer(frame_bytes, dtype=np.uint8).reshape((h, w, 4))
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            self._frame_number += 1
            meta = FrameMetadata(
                width=w, height=h,
                timestamp=time.time() - self._start_time,
                frame_number=self._frame_number,
                source=SourceType.SHARED_MEMORY,
                fps=0,
            )
            return frame_bgr, meta
        except Exception as e:
            print(f"[SHM] Read error: {e}")
            return None, None


# ─── Device Discovery ─────────────────────────────────────────────

def list_video_devices(max_check: int = 10):
    """Probe for available video capture devices."""
    print("Scanning for video devices...")
    found = []
    for i in range(max_check):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            backend = cap.getBackendName()
            print(f"  [Device {i}] {w}x{h} @ {fps:.0f} FPS ({backend})")
            found.append(i)
            cap.release()
    if not found:
        print("  No video devices found.")
    return found


# ─── CLI Test Modes ───────────────────────────────────────────────

def _test_source(src: VideoSource):
    """Interactive test: show frames in a window until 'q' is pressed."""
    if not src.open():
        return
    print("Showing video feed. Press 'q' to quit.")
    while src.is_open:
        frame, meta = src.get_frame()
        if frame is None:
            break
        # Draw info overlay
        info = f"Frame {meta.frame_number} | {meta.width}x{meta.height} | {meta.timestamp:.1f}s"
        cv2.putText(frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)
        preview = cv2.resize(frame, (960, 540)) if frame.shape[1] > 960 else frame
        cv2.imshow("League Bot – Video Source Test", preview)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    src.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="League Bot Video Input")
    parser.add_argument("--list-devices", action="store_true",
                        help="List available video capture devices")
    parser.add_argument("--test-capture", action="store_true",
                        help="Test capture card input")
    parser.add_argument("--test-replay", action="store_true",
                        help="Test replay file input")
    parser.add_argument("--test-shm", action="store_true",
                        help="Test shared memory input")
    parser.add_argument("--device-index", type=int, default=0,
                        help="Capture card device index (default: 0)")
    parser.add_argument("--file", type=str, default=None,
                        help="Path to replay video file")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Replay playback speed multiplier (default: 1.0)")
    args = parser.parse_args()

    if args.list_devices:
        list_video_devices()
    elif args.test_capture:
        src = VideoSource(SourceType.CAPTURE_CARD, device_index=args.device_index)
        _test_source(src)
    elif args.test_replay:
        if not args.file:
            print("Error: --file is required for --test-replay")
        else:
            src = VideoSource(SourceType.REPLAY, replay_file=args.file,
                              playback_speed=args.speed)
            _test_source(src)
    elif args.test_shm:
        src = VideoSource(SourceType.SHARED_MEMORY)
        _test_source(src)
    else:
        parser.print_help()
