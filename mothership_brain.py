"""
mothership_brain.py – League Bot AI Brain (Mothership PC)

The central orchestrator: reads video → detects entities → reasons tactically → sends input.

Modes:
  replay       – Process a video file (for offline testing)
  capture_card – Live feed from USB capture card
  shm          – DXGI shared memory (same-PC mode)

Usage:
  # Offline replay testing (start here!)
  python mothership_brain.py --mode replay --replay-file gameplay.mp4 --dry-run --show-preview

  # Live with capture card + Pico
  python mothership_brain.py --mode capture_card --device-index 1 --pico-port COM3

  # Full debug mode
  python mothership_brain.py --mode capture_card --device-index 1 --pico-port COM3 --show-preview --log-commands
"""

import cv2
import numpy as np
import time
import json
import argparse
import os
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field

# Local imports
from video_input import VideoSource, SourceType, FrameMetadata
from input_controller import InputController, DryRunController
from vision.league_detector import LeagueDetector, GameScene, Entity

# Try to import Ollama (optional for first-time setup)
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("[WARNING] ollama package not installed. Tactical reasoning disabled.")


# ─── Game State Tracker ──────────────────────────────────────────

@dataclass
class GameState:
    """Accumulated game state across frames."""
    frame_count: int = 0
    last_scene: Optional[GameScene] = None
    last_decision: str = ""
    last_decision_time: float = 0.0

    # Tracking
    enemy_champions_seen: int = 0
    ally_minions_nearby: int = 0
    enemy_minions_nearby: int = 0
    near_tower: bool = False
    objective_visible: bool = False

    # Health/danger estimation from scene
    danger_level: str = "safe"  # safe, caution, danger

    def update(self, scene: GameScene):
        """Update state from a new scene detection."""
        self.frame_count += 1
        self.last_scene = scene

        self.enemy_champions_seen = len(scene.enemy_champions)
        self.ally_minions_nearby = len(scene.ally_minions)
        self.enemy_minions_nearby = len(scene.enemy_minions)
        self.near_tower = len(scene.structures) > 0
        self.objective_visible = len(scene.objectives) > 0

        # Simple danger estimation
        if self.enemy_champions_seen >= 2:
            self.danger_level = "danger"
        elif self.enemy_champions_seen == 1:
            self.danger_level = "caution"
        else:
            self.danger_level = "safe"

    def summary_for_llm(self) -> str:
        """Generate a compact summary for the LLM prompt."""
        parts = [
            f"Danger: {self.danger_level}",
            f"Enemy champs visible: {self.enemy_champions_seen}",
            f"Ally minions: {self.ally_minions_nearby}",
            f"Enemy minions: {self.enemy_minions_nearby}",
            f"Near tower: {self.near_tower}",
            f"Objective visible: {self.objective_visible}",
        ]
        if self.last_scene:
            parts.append(f"Scene: {self.last_scene.summary()}")
        return " | ".join(parts)


# ─── Mothership Brain ────────────────────────────────────────────

class MothershipBrain:
    """
    Main AI brain. Orchestrates:
      1. Video input (replay / capture card / shared memory)
      2. YOLO object detection
      3. Ollama LLM tactical reasoning
      4. Input commands (Pico HID or dry-run)
    """

    def __init__(self, args):
        self.args = args
        self.video_source: Optional[VideoSource] = None
        self.detector: Optional[LeagueDetector] = None
        self.controller: Optional[InputController] = None
        self.game_state = GameState()

        # Tactical LLM config
        self.tactical_model = args.tactical_model
        self.tactical_interval = args.tactical_interval  # seconds between LLM calls

        # Performance
        self._frame_times: List[float] = []

    def initialize(self) -> bool:
        """Initialize all subsystems."""
        print("\n" + "=" * 60)
        print("  MOTHERSHIP BRAIN – League Bot AI")
        print("=" * 60)

        # 1. Video Source
        print(f"\n[1/3] Video Source: {self.args.mode}")
        if self.args.mode == "replay":
            self.video_source = VideoSource(
                SourceType.REPLAY,
                replay_file=self.args.replay_file,
                playback_speed=self.args.speed,
            )
        elif self.args.mode == "capture_card":
            self.video_source = VideoSource(
                SourceType.CAPTURE_CARD,
                device_index=self.args.device_index,
            )
        elif self.args.mode == "shm":
            self.video_source = VideoSource(SourceType.SHARED_MEMORY)
        else:
            print(f"Error: Unknown mode '{self.args.mode}'")
            return False

        if not self.video_source.open():
            return False

        # 2. Vision Detector
        print(f"\n[2/3] Vision Detector")
        self.detector = LeagueDetector(
            custom_model_path=self.args.model_path,
            confidence_threshold=self.args.confidence,
            device=self.args.device,
        )

        # 3. Input Controller
        print(f"\n[3/3] Input Controller")
        if self.args.dry_run:
            self.controller = DryRunController()
        else:
            self.controller = InputController(
                port=self.args.pico_port,
                baudrate=self.args.baudrate,
            )
        if not self.controller.connect():
            print("Warning: Controller connection failed. Continuing in observation mode.")

        print("\n" + "=" * 60)
        print("  All systems initialized. Starting main loop...")
        print("  Press 'q' to quit" + (" (in preview window)" if self.args.show_preview else ""))
        print("=" * 60 + "\n")
        return True

    def run(self):
        """Main processing loop."""
        if not self.initialize():
            print("Initialization failed. Exiting.")
            return

        last_tactical_time = 0
        fps_update_time = time.time()
        fps_count = 0

        try:
            while True:
                loop_start = time.time()

                # ── 1. READ FRAME ──
                frame, meta = self.video_source.get_frame()
                if frame is None:
                    if self.args.mode == "replay":
                        print("\n[END] Replay finished.")
                        break
                    continue

                # ── 2. DETECT ENTITIES (YOLO) ──
                scene = self.detector.detect(
                    frame,
                    frame_number=meta.frame_number,
                    timestamp=meta.timestamp,
                )
                self.game_state.update(scene)

                # ── 3. TACTICAL REASONING (Ollama LLM) ──
                current_time = time.time()
                if (current_time - last_tactical_time > self.tactical_interval
                        and OLLAMA_AVAILABLE
                        and not self.args.no_llm):
                    decision = self._ask_tactical_expert(frame, scene)
                    self.game_state.last_decision = decision
                    self.game_state.last_decision_time = current_time
                    last_tactical_time = current_time

                    if decision:
                        print(f"\n[TACTICAL] {decision}")

                # ── 4. EXECUTE ACTIONS ──
                self._execute_decision(scene)

                # ── 5. DISPLAY / LOGGING ──
                fps_count += 1
                if current_time - fps_update_time >= 1.0:
                    fps = fps_count / (current_time - fps_update_time)
                    self._print_status(meta, scene, fps)
                    fps_count = 0
                    fps_update_time = current_time

                if self.args.show_preview:
                    annotated = self.detector.draw_detections(frame, scene)
                    # Add decision overlay
                    if self.game_state.last_decision:
                        cv2.putText(annotated, f"AI: {self.game_state.last_decision[:80]}",
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    # Add danger level
                    danger_colors = {"safe": (0, 255, 0), "caution": (0, 255, 255), "danger": (0, 0, 255)}
                    cv2.putText(annotated, f"Danger: {self.game_state.danger_level.upper()}",
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                danger_colors.get(self.game_state.danger_level, (255, 255, 255)), 2)

                    preview = cv2.resize(annotated, (960, 540)) if annotated.shape[1] > 960 else annotated
                    cv2.imshow("Mothership Brain", preview)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                # Frame timing
                loop_time = time.time() - loop_start
                self._frame_times.append(loop_time)

        except KeyboardInterrupt:
            print("\n[INTERRUPT] Shutting down...")
        finally:
            self._shutdown()

    def _ask_tactical_expert(self, frame: np.ndarray, scene: GameScene) -> str:
        """Send game state to local Ollama for strategic analysis."""
        if not OLLAMA_AVAILABLE:
            return ""

        state_summary = self.game_state.summary_for_llm()

        prompt = f"""You are an AI playing League of Legends. Analyze the current game state and provide a tactical command.

GAME STATE:
{state_summary}

RULES:
- Give exactly ONE tactical command as a short sentence
- Be specific: mention targets, positions (left/right/center of screen), or actions
- Examples: "Farm the 3 enemy minions on the right", "Retreat to tower - 2 enemies visible",
  "Engage the enemy champion in center", "Move to dragon pit", "Back to base, low resources"
- If no enemies or objectives visible, default to farming or positioning

YOUR COMMAND:"""

        try:
            # Resize frame for LLM vision
            frame_small = cv2.resize(frame, (640, 360))
            _, buffer = cv2.imencode('.jpg', frame_small)

            response = ollama.chat(
                model=self.tactical_model,
                messages=[{
                    'role': 'user',
                    'content': prompt,
                    'images': [buffer.tobytes()]
                }]
            )
            return response['message']['content'].strip()
        except Exception as e:
            return f"(LLM error: {e})"

    def _execute_decision(self, scene: GameScene):
        """
        Convert game state + tactical decision into concrete input actions.
        This is the reactive execution layer.
        """
        if not self.controller:
            return

        decision = self.game_state.last_decision.lower()

        # ── DANGER: Retreat logic ──
        if self.game_state.danger_level == "danger" or "retreat" in decision or "back" in decision:
            # Move toward bottom-right (usually toward base/tower)
            self.controller.move_to(1600, 900)
            return

        # ── FARMING: Attack nearby minions ──
        if ("farm" in decision or "minion" in decision) and scene.enemy_minions:
            # Attack the closest enemy minion
            target = min(scene.enemy_minions,
                         key=lambda e: abs(e.center[0] - 960) + abs(e.center[1] - 540))
            self.controller.attack_move(target.center[0], target.center[1])
            return

        # ── ENGAGE: Attack enemy champion ──
        if ("engage" in decision or "attack" in decision) and scene.enemy_champions:
            target = scene.enemy_champions[0]
            self.controller.cast_ability("Q", target.center[0], target.center[1])
            return

        # ── OBJECTIVE: Move to dragon/baron ──
        if ("dragon" in decision or "baron" in decision) and scene.objectives:
            target = scene.objectives[0]
            self.controller.move_to(target.center[0], target.center[1])
            return

        # ── DEFAULT: Farm if minions present, otherwise hold position ──
        if scene.enemy_minions:
            target = scene.enemy_minions[0]
            self.controller.attack_move(target.center[0], target.center[1])

    def _print_status(self, meta: FrameMetadata, scene: GameScene, fps: float):
        """Print a compact status line."""
        entities = len(scene.entities)
        danger = self.game_state.danger_level
        source = meta.source.value
        line = (f"[{source}] Frame {meta.frame_number} | "
                f"FPS: {fps:.1f} | Entities: {entities} | "
                f"Danger: {danger}")
        print(line, end='\r')

    def _shutdown(self):
        """Clean shutdown of all subsystems."""
        print("\n\nShutting down...")
        if self.controller:
            self.controller.reset()
            self.controller.disconnect()
        if self.video_source:
            self.video_source.release()
        cv2.destroyAllWindows()

        # Log stats
        if self.args.log_commands and self.controller:
            log = self.controller.get_command_log()
            log_path = "command_log.json"
            with open(log_path, 'w') as f:
                json.dump([{
                    "time": l.timestamp,
                    "cmd": l.command,
                    "resp": l.response
                } for l in log], f, indent=2)
            print(f"Command log saved: {log_path} ({len(log)} commands)")

        if self._frame_times:
            avg_ms = (sum(self._frame_times) / len(self._frame_times)) * 1000
            print(f"Average frame time: {avg_ms:.1f}ms ({1000 / avg_ms:.0f} FPS)")

        print("Goodbye!")


# ─── CLI ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Mothership Brain – League Bot AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Offline replay test (no hardware needed):
  python mothership_brain.py --mode replay --replay-file gameplay.mp4 --dry-run --show-preview

  # Live with capture card:
  python mothership_brain.py --mode capture_card --device-index 1 --pico-port COM3

  # Observation mode (no input sent):
  python mothership_brain.py --mode capture_card --device-index 1 --dry-run --show-preview
        """
    )

    # Mode
    parser.add_argument("--mode", type=str, required=True,
                        choices=["replay", "capture_card", "shm"],
                        help="Video source mode")

    # Video source options
    parser.add_argument("--replay-file", type=str,
                        help="Path to replay video file (for replay mode)")
    parser.add_argument("--device-index", type=int, default=0,
                        help="Capture card device index (for capture_card mode)")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Replay playback speed (default: 1.0)")

    # Input controller
    parser.add_argument("--dry-run", action="store_true",
                        help="Log commands instead of sending to Pico")
    parser.add_argument("--pico-port", type=str, default="COM3",
                        help="Pico serial port (default: COM3)")
    parser.add_argument("--baudrate", type=int, default=115200,
                        help="Serial baudrate")

    # Vision
    parser.add_argument("--model-path", type=str, default=None,
                        help="Custom YOLO model path (auto-detects if not set)")
    parser.add_argument("--confidence", type=float, default=0.4,
                        help="YOLO confidence threshold (default: 0.4)")
    parser.add_argument("--device", type=str, default="0",
                        help="YOLO device: '0' for GPU, 'cpu' for CPU")

    # Tactical LLM
    parser.add_argument("--tactical-model", type=str, default="llama3.2-vision",
                        help="Ollama model for tactical reasoning")
    parser.add_argument("--tactical-interval", type=float, default=2.0,
                        help="Seconds between LLM tactical calls (default: 2.0)")
    parser.add_argument("--no-llm", action="store_true",
                        help="Disable Ollama LLM reasoning (YOLO only)")

    # Display / Debug
    parser.add_argument("--show-preview", action="store_true",
                        help="Show annotated preview window")
    parser.add_argument("--log-commands", action="store_true",
                        help="Save command log to JSON on exit")

    args = parser.parse_args()

    # Validation
    if args.mode == "replay" and not args.replay_file:
        parser.error("--replay-file is required when --mode is 'replay'")

    brain = MothershipBrain(args)
    brain.run()


if __name__ == "__main__":
    main()
