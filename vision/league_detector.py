"""
league_detector.py – League of Legends YOLO Object Detection

Wraps YOLO with League-specific entity classes and structured output.
Falls back to generic YOLO (COCO classes) if no custom model is found.

Entity Classes (when custom-trained):
  champion_ally, champion_enemy, minion_ally, minion_enemy,
  tower, inhibitor, dragon, baron, skillshot, health_bar

Usage:
  from vision.league_detector import LeagueDetector
  detector = LeagueDetector()
  entities = detector.detect(frame_bgr)
"""

import os
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from ultralytics import YOLO


# ─── Entity Classes ───────────────────────────────────────────────

LEAGUE_CLASSES = [
    "champion_ally",
    "champion_enemy",
    "minion_ally",
    "minion_enemy",
    "tower",
    "inhibitor",
    "dragon",
    "baron",
    "skillshot",
    "health_bar",
]

# Colors for drawing (BGR format)
CLASS_COLORS = {
    "champion_ally":  (255, 200, 0),    # Cyan-ish
    "champion_enemy": (0, 0, 255),       # Red
    "minion_ally":    (200, 200, 0),      # Light cyan
    "minion_enemy":   (0, 100, 255),      # Orange-red
    "tower":          (0, 255, 255),      # Yellow
    "inhibitor":      (255, 0, 255),      # Magenta
    "dragon":         (0, 255, 0),        # Green
    "baron":          (128, 0, 128),      # Purple
    "skillshot":      (0, 128, 255),      # Orange
    "health_bar":     (255, 255, 255),    # White
    # Generic COCO fallback
    "person":         (255, 100, 0),
    "default":        (128, 128, 128),
}


@dataclass
class Entity:
    """A detected game entity."""
    label: str
    confidence: float
    bbox: List[float]           # [x1, y1, x2, y2] in pixels
    center: Tuple[int, int] = (0, 0)
    area: float = 0.0

    def __post_init__(self):
        if self.bbox:
            x1, y1, x2, y2 = self.bbox
            self.center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            self.area = (x2 - x1) * (y2 - y1)

    @property
    def is_enemy(self) -> bool:
        return "enemy" in self.label

    @property
    def is_ally(self) -> bool:
        return "ally" in self.label

    @property
    def is_champion(self) -> bool:
        return "champion" in self.label

    @property
    def is_minion(self) -> bool:
        return "minion" in self.label

    @property
    def is_objective(self) -> bool:
        return self.label in ("dragon", "baron")

    @property
    def is_structure(self) -> bool:
        return self.label in ("tower", "inhibitor")


@dataclass
class GameScene:
    """Structured snapshot of all detected entities in a frame."""
    entities: List[Entity] = field(default_factory=list)
    frame_number: int = 0
    timestamp: float = 0.0

    @property
    def enemy_champions(self) -> List[Entity]:
        return [e for e in self.entities if e.is_champion and e.is_enemy]

    @property
    def ally_champions(self) -> List[Entity]:
        return [e for e in self.entities if e.is_champion and e.is_ally]

    @property
    def enemy_minions(self) -> List[Entity]:
        return [e for e in self.entities if e.is_minion and e.is_enemy]

    @property
    def ally_minions(self) -> List[Entity]:
        return [e for e in self.entities if e.is_minion and e.is_ally]

    @property
    def structures(self) -> List[Entity]:
        return [e for e in self.entities if e.is_structure]

    @property
    def objectives(self) -> List[Entity]:
        return [e for e in self.entities if e.is_objective]

    def summary(self) -> str:
        """Generate a text summary of the scene for LLM consumption."""
        parts = []
        if self.enemy_champions:
            locs = [f"({e.center[0]},{e.center[1]})" for e in self.enemy_champions]
            parts.append(f"Enemy champions at: {', '.join(locs)}")
        if self.ally_champions:
            locs = [f"({e.center[0]},{e.center[1]})" for e in self.ally_champions]
            parts.append(f"Ally champions at: {', '.join(locs)}")
        if self.enemy_minions:
            parts.append(f"{len(self.enemy_minions)} enemy minions nearby")
        if self.ally_minions:
            parts.append(f"{len(self.ally_minions)} ally minions nearby")
        if self.structures:
            for s in self.structures:
                parts.append(f"{s.label} at ({s.center[0]},{s.center[1]})")
        if self.objectives:
            for o in self.objectives:
                parts.append(f"{o.label} at ({o.center[0]},{o.center[1]})")
        if not parts:
            parts.append("No significant entities detected")
        return " | ".join(parts)


# ─── Detector ─────────────────────────────────────────────────────

class LeagueDetector:
    """
    YOLO-based League of Legends entity detector.

    Tries to load a custom-trained League model first.
    Falls back to generic YOLOv8n (COCO classes) for testing.
    """

    def __init__(self,
                 custom_model_path: Optional[str] = None,
                 confidence_threshold: float = 0.4,
                 device: str = "0"):
        self.confidence_threshold = confidence_threshold
        self.device = device
        self._model: Optional[YOLO] = None
        self._is_custom = False
        self._class_names = {}

        # Try to find custom model
        if custom_model_path is None:
            # Default search paths
            search_paths = [
                os.path.join(os.path.dirname(__file__), "best.pt"),
                os.path.join(os.path.dirname(__file__), "league_yolo.pt"),
                os.path.join(os.path.dirname(__file__), "runs", "detect", "train", "weights", "best.pt"),
            ]
            for p in search_paths:
                if os.path.exists(p):
                    custom_model_path = p
                    break

        self._load_model(custom_model_path)

    def _load_model(self, model_path: Optional[str]):
        """Load YOLO model – custom if available, generic otherwise."""
        if model_path and os.path.exists(model_path):
            print(f"[VISION] Loading custom League model: {model_path}")
            self._model = YOLO(model_path)
            self._is_custom = True
        else:
            print("[VISION] No custom model found. Using generic YOLOv8n (COCO).")
            print("         Train a custom model with: python vision/train.py")
            self._model = YOLO("yolov8n.pt")
            self._is_custom = False

        self._class_names = self._model.names
        print(f"[VISION] Model loaded. Classes: {list(self._class_names.values())[:10]}...")

    def detect(self, frame_bgr: np.ndarray,
               frame_number: int = 0,
               timestamp: float = 0.0) -> GameScene:
        """
        Run detection on a BGR frame. Returns a GameScene.
        """
        results = self._model(
            frame_bgr,
            verbose=False,
            device=self.device,
            conf=self.confidence_threshold
        )

        entities = []
        for result in results:
            for box in result.boxes:
                label = self._class_names[int(box.cls)]
                conf = float(box.conf)
                bbox = box.xyxy[0].tolist()

                entities.append(Entity(
                    label=label,
                    confidence=conf,
                    bbox=bbox,
                ))

        return GameScene(
            entities=entities,
            frame_number=frame_number,
            timestamp=timestamp,
        )

    def draw_detections(self, frame: np.ndarray, scene: GameScene) -> np.ndarray:
        """Draw bounding boxes and labels on a frame for visualization."""
        annotated = frame.copy()

        for entity in scene.entities:
            x1, y1, x2, y2 = [int(v) for v in entity.bbox]
            color = CLASS_COLORS.get(entity.label, CLASS_COLORS["default"])

            # Draw box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Draw label
            label_text = f"{entity.label} {entity.confidence:.0%}"
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(annotated, label_text, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Draw summary
        summary = scene.summary()
        cv2.putText(annotated, summary[:100], (10, annotated.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        return annotated

    @property
    def is_custom_model(self) -> bool:
        return self._is_custom
