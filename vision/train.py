"""
train.py – Train a Custom YOLO Model for League of Legends

Fine-tunes YOLOv8n on your labeled League dataset.
Outputs the best weights to vision/runs/detect/train/weights/best.pt

Prerequisites:
  1. Collect frames: python vision/collect_training_data.py --source gameplay.mp4 --fps 1
  2. Label them:     python vision/collect_training_data.py --label
  3. Train:          python vision/train.py --epochs 100

Usage:
  python vision/train.py --epochs 100 --data vision/data.yaml --batch 16
  python vision/train.py --resume  # resume interrupted training
"""

import os
import sys
import argparse
from pathlib import Path


def check_dataset(data_yaml: str) -> bool:
    """Verify the dataset directory structure exists and has images."""
    import yaml

    if not os.path.exists(data_yaml):
        print(f"Error: data.yaml not found at '{data_yaml}'")
        return False

    with open(data_yaml, 'r') as f:
        config = yaml.safe_load(f)

    base_path = config.get("path", ".")
    train_path = os.path.join(base_path, config.get("train", "images/train"))
    val_path = os.path.join(base_path, config.get("val", "images/val"))

    # Check train images
    if not os.path.exists(train_path):
        print(f"Error: Training images directory not found: {train_path}")
        print("  Run: python vision/collect_training_data.py --source <video> --fps 1")
        return False

    train_images = [f for f in os.listdir(train_path)
                    if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    if len(train_images) < 10:
        print(f"Warning: Only {len(train_images)} training images found.")
        print("  Recommended: 200+ images for decent results.")

    # Check labels
    train_labels_dir = train_path.replace("images", "labels")
    if os.path.exists(train_labels_dir):
        labels = [f for f in os.listdir(train_labels_dir) if f.endswith('.txt')]
        print(f"Dataset: {len(train_images)} images, {len(labels)} labeled")
    else:
        print(f"Warning: No labels directory found at {train_labels_dir}")
        print("  Run: python vision/collect_training_data.py --label")
        return False

    # Create val directory if needed (split some training data)
    if not os.path.exists(val_path):
        print(f"Creating validation set from training data...")
        os.makedirs(val_path, exist_ok=True)
        val_labels = val_path.replace("images", "labels")
        os.makedirs(val_labels, exist_ok=True)

        # Move ~20% of images to validation
        import shutil
        import random
        random.seed(42)
        val_count = max(1, len(train_images) // 5)
        val_images = random.sample(train_images, val_count)

        for img_name in val_images:
            shutil.move(
                os.path.join(train_path, img_name),
                os.path.join(val_path, img_name)
            )
            label_name = os.path.splitext(img_name)[0] + ".txt"
            label_src = os.path.join(train_labels_dir, label_name)
            if os.path.exists(label_src):
                shutil.move(label_src, os.path.join(val_labels, label_name))

        print(f"  Moved {val_count} images to validation set.")

    return True


def train(data_yaml: str, epochs: int = 100, batch: int = 16,
          img_size: int = 640, resume: bool = False,
          base_model: str = "yolov8n.pt", project: str = "vision/runs"):
    """Train YOLO on the League dataset."""
    from ultralytics import YOLO

    if resume:
        # Find the last training run
        last_weights = os.path.join(project, "detect", "train", "weights", "last.pt")
        if os.path.exists(last_weights):
            print(f"Resuming training from {last_weights}")
            model = YOLO(last_weights)
            model.train(resume=True)
            return
        else:
            print("No previous training found to resume. Starting fresh.")

    print(f"\n{'='*60}")
    print(f"  TRAINING LEAGUE YOLO MODEL")
    print(f"  Base Model: {base_model}")
    print(f"  Epochs: {epochs} | Batch: {batch} | Image Size: {img_size}")
    print(f"  Data: {data_yaml}")
    print(f"{'='*60}\n")

    model = YOLO(base_model)

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch,
        imgsz=img_size,
        project=project,
        name="detect/train",
        exist_ok=True,
        device=0,              # Use GPU
        patience=20,           # Early stopping patience
        save=True,
        save_period=10,        # Save checkpoint every 10 epochs
        plots=True,
        verbose=True,
    )

    # Copy best weights to vision/ for easy access
    best_src = os.path.join(project, "detect", "train", "weights", "best.pt")
    best_dst = os.path.join(os.path.dirname(data_yaml), "league_yolo.pt")
    if os.path.exists(best_src):
        import shutil
        shutil.copy2(best_src, best_dst)
        print(f"\n✓ Best weights copied to: {best_dst}")
        print(f"  The LeagueDetector will automatically use this model!")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train League YOLO Model")
    parser.add_argument("--data", type=str, default="vision/data.yaml",
                        help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs (default: 100)")
    parser.add_argument("--batch", type=int, default=16,
                        help="Batch size (default: 16)")
    parser.add_argument("--img-size", type=int, default=640,
                        help="Training image size (default: 640)")
    parser.add_argument("--base-model", type=str, default="yolov8n.pt",
                        help="Base YOLO model to fine-tune (default: yolov8n.pt)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume interrupted training")
    parser.add_argument("--check-only", action="store_true",
                        help="Only check dataset, don't train")
    args = parser.parse_args()

    if args.check_only:
        check_dataset(args.data)
    else:
        if check_dataset(args.data):
            train(
                data_yaml=args.data,
                epochs=args.epochs,
                batch=args.batch,
                img_size=args.img_size,
                resume=args.resume,
                base_model=args.base_model,
            )
