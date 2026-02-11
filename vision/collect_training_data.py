"""
collect_training_data.py – Screenshot Collector & Labeling Tool for YOLO

Captures frames from a video source and saves them for YOLO training.
Includes an interactive labeling tool (draw bounding boxes with mouse).

Usage:
  # Extract frames from a replay for manual labeling
  python vision/collect_training_data.py --source path/to/gameplay.mp4 --output-dir vision/dataset --fps 1

  # Label images interactively
  python vision/collect_training_data.py --label --image-dir vision/dataset/images/train
"""

import cv2
import os
import sys
import argparse
import json
from typing import List, Tuple

# Add parent dir so we can import video_input
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

LEAGUE_CLASSES = [
    "champion_ally",      # 0
    "champion_enemy",     # 1
    "minion_ally",        # 2
    "minion_enemy",       # 3
    "tower",              # 4
    "inhibitor",          # 5
    "dragon",             # 6
    "baron",              # 7
    "skillshot",          # 8
    "health_bar",         # 9
]


def extract_frames(video_path: str, output_dir: str, target_fps: float = 1.0):
    """Extract frames from video at target FPS."""
    img_dir = os.path.join(output_dir, "images", "train")
    os.makedirs(img_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open '{video_path}'")
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    interval = max(1, int(video_fps / target_fps))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Extracting frames from: {video_path}")
    print(f"  Video FPS: {video_fps:.1f} | Taking every {interval} frames")
    print(f"  Output: {img_dir}/")

    idx = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % interval == 0:
            filename = f"frame_{saved:06d}.jpg"
            cv2.imwrite(os.path.join(img_dir, filename), frame)
            saved += 1
            if saved % 20 == 0:
                progress = (idx / total * 100) if total > 0 else 0
                print(f"  {saved} frames saved ({progress:.0f}%)")
        idx += 1

    cap.release()
    print(f"\nDone! {saved} frames saved to {img_dir}/")
    print(f"\nNext: Label them with:")
    print(f"  python vision/collect_training_data.py --label --image-dir {img_dir}")


class BBoxLabeler:
    """Interactive bounding box labeling tool using OpenCV."""

    def __init__(self, image_dir: str, label_dir: str = None):
        self.image_dir = image_dir
        self.label_dir = label_dir or image_dir.replace("images", "labels")
        os.makedirs(self.label_dir, exist_ok=True)

        self.images = sorted([
            f for f in os.listdir(image_dir)
            if f.lower().endswith(('.jpg', '.png', '.jpeg'))
        ])
        self.current_idx = 0
        self.current_boxes: List[Tuple[int, int, int, int, int]] = []  # (x1,y1,x2,y2,class)
        self.drawing = False
        self.start_x = 0
        self.start_y = 0
        self.current_class = 0
        self.frame = None
        self.display = None

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_x = x
            self.start_y = y
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.display = self.frame.copy()
            self._draw_existing_boxes(self.display)
            cv2.rectangle(self.display, (self.start_x, self.start_y), (x, y),
                          (0, 255, 0), 2)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            x1 = min(self.start_x, x)
            y1 = min(self.start_y, y)
            x2 = max(self.start_x, x)
            y2 = max(self.start_y, y)
            if (x2 - x1) > 5 and (y2 - y1) > 5:
                self.current_boxes.append((x1, y1, x2, y2, self.current_class))

    def _draw_existing_boxes(self, img):
        for (x1, y1, x2, y2, cls) in self.current_boxes:
            color = (0, 255, 0) if cls % 2 == 0 else (0, 0, 255)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, LEAGUE_CLASSES[cls], (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    def _draw_hud(self, img):
        h, w = img.shape[:2]
        # Class selector
        cv2.rectangle(img, (0, h - 40), (w, h), (30, 30, 30), -1)
        cls_text = f"Class [{self.current_class}]: {LEAGUE_CLASSES[self.current_class]}"
        cv2.putText(img, cls_text, (10, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        # Controls
        controls = "0-9:class | SPACE:save+next | Z:undo | S:skip | Q:quit"
        cv2.putText(img, controls, (w - 550, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
        # Image counter
        counter = f"[{self.current_idx + 1}/{len(self.images)}]"
        cv2.putText(img, counter, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        # Box count
        box_text = f"Boxes: {len(self.current_boxes)}"
        cv2.putText(img, box_text, (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    def _save_labels(self):
        """Save current boxes in YOLO format."""
        if not self.current_boxes:
            return
        img_name = self.images[self.current_idx]
        label_name = os.path.splitext(img_name)[0] + ".txt"
        label_path = os.path.join(self.label_dir, label_name)

        h, w = self.frame.shape[:2]
        with open(label_path, 'w') as f:
            for (x1, y1, x2, y2, cls) in self.current_boxes:
                cx = ((x1 + x2) / 2) / w
                cy = ((y1 + y2) / 2) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                f.write(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
        print(f"  Saved {len(self.current_boxes)} labels → {label_path}")

    def _load_existing_labels(self):
        """Load existing labels for current image."""
        self.current_boxes = []
        img_name = self.images[self.current_idx]
        label_name = os.path.splitext(img_name)[0] + ".txt"
        label_path = os.path.join(self.label_dir, label_name)

        if not os.path.exists(label_path):
            return

        h, w = self.frame.shape[:2]
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls = int(parts[0])
                    cx, cy, bw, bh = map(float, parts[1:])
                    x1 = int((cx - bw / 2) * w)
                    y1 = int((cy - bh / 2) * h)
                    x2 = int((cx + bw / 2) * w)
                    y2 = int((cy + bh / 2) * h)
                    self.current_boxes.append((x1, y1, x2, y2, cls))

    def run(self):
        """Run the interactive labeling loop."""
        if not self.images:
            print("No images found.")
            return

        # Skip already-labeled images
        first_unlabeled = 0
        for i, img_name in enumerate(self.images):
            label_name = os.path.splitext(img_name)[0] + ".txt"
            if not os.path.exists(os.path.join(self.label_dir, label_name)):
                first_unlabeled = i
                break
        self.current_idx = first_unlabeled

        cv2.namedWindow("Labeler", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Labeler", self._mouse_callback)

        print(f"\n{'='*50}")
        print(f"  LABELING TOOL – {len(self.images)} images")
        print(f"  Starting from image {self.current_idx + 1}")
        print(f"{'='*50}")
        print(f"\nControls:")
        print(f"  0-9     → Select class")
        print(f"  SPACE   → Save labels & next image")
        print(f"  S       → Skip (no labels)")
        print(f"  Z       → Undo last box")
        print(f"  Q       → Quit")
        print()

        while self.current_idx < len(self.images):
            img_path = os.path.join(self.image_dir, self.images[self.current_idx])
            self.frame = cv2.imread(img_path)
            if self.frame is None:
                self.current_idx += 1
                continue

            self._load_existing_labels()

            while True:
                self.display = self.frame.copy()
                self._draw_existing_boxes(self.display)
                self._draw_hud(self.display)
                cv2.imshow("Labeler", self.display)

                key = cv2.waitKey(30) & 0xFF

                if key == ord('q'):
                    cv2.destroyAllWindows()
                    return
                elif key == ord(' '):  # Save and next
                    self._save_labels()
                    self.current_idx += 1
                    break
                elif key == ord('s'):  # Skip
                    self.current_idx += 1
                    break
                elif key == ord('z'):  # Undo
                    if self.current_boxes:
                        self.current_boxes.pop()
                elif ord('0') <= key <= ord('9'):
                    self.current_class = key - ord('0')
                    if self.current_class >= len(LEAGUE_CLASSES):
                        self.current_class = len(LEAGUE_CLASSES) - 1

        cv2.destroyAllWindows()
        print("\nLabeling complete!")
        # Count labeled images
        labeled = sum(1 for f in os.listdir(self.label_dir) if f.endswith('.txt'))
        print(f"Total labeled images: {labeled}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO Training Data Collection")
    parser.add_argument("--source", type=str,
                        help="Video file to extract frames from")
    parser.add_argument("--output-dir", type=str, default="vision/dataset",
                        help="Dataset output directory")
    parser.add_argument("--fps", type=float, default=1.0,
                        help="Frames per second to extract (default: 1)")
    parser.add_argument("--label", action="store_true",
                        help="Launch interactive labeling tool")
    parser.add_argument("--image-dir", type=str,
                        default="vision/dataset/images/train",
                        help="Directory of images to label")
    args = parser.parse_args()

    if args.label:
        labeler = BBoxLabeler(args.image_dir)
        labeler.run()
    elif args.source:
        extract_frames(args.source, args.output_dir, args.fps)
    else:
        parser.print_help()
