"""
input_controller.py – Python → Raspberry Pi Pico Serial Bridge

Sends game commands to the Pico HID controller over USB serial.
Includes DryRunController for offline testing without hardware.

Usage:
  # Test mode (sends ping/key/mouse commands)
  python input_controller.py --test --port COM3

  # Dry run (no hardware, logs to console)
  python input_controller.py --test --dry-run
"""

import json
import time
import argparse
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum


class MouseButton(Enum):
    LEFT = "left"
    RIGHT = "right"


@dataclass
class CommandLog:
    """Record of a command sent (for replay analysis)."""
    timestamp: float
    command: Dict[str, Any]
    response: Optional[Dict[str, Any]] = None


class InputController:
    """
    Sends HID commands to the Raspberry Pi Pico over USB serial.

    The Pico acts as a USB keyboard/mouse connected to the gaming laptop.
    This controller runs on the mothership PC and sends JSON commands.
    """

    def __init__(self, port: str = "COM3", baudrate: int = 115200, timeout: float = 0.1):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self._serial = None
        self._command_log: list = []

    def connect(self) -> bool:
        """Connect to the Pico over USB serial."""
        try:
            import serial
            self._serial = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout
            )
            time.sleep(1)  # Wait for Pico to initialize
            print(f"[INPUT] Connected to Pico on {self.port}")

            # Verify connection with ping
            resp = self._send_raw({"a": "ping"})
            if resp and resp.get("s") == "pong":
                print("[INPUT] Pico responded to ping [OK]")
                return True
            else:
                print("[INPUT] Warning: Pico did not respond to ping")
                return True  # Still connected, just no response yet
        except Exception as e:
            print(f"[INPUT] Error connecting to {self.port}: {e}")
            return False

    def disconnect(self):
        """Disconnect from the Pico."""
        if self._serial and self._serial.is_open:
            self.reset()
            self._serial.close()
            print("[INPUT] Disconnected from Pico")

    def _send_raw(self, cmd: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Send a raw JSON command and optionally read response."""
        if not self._serial or not self._serial.is_open:
            return None
        try:
            msg = json.dumps(cmd) + "\n"
            self._serial.write(msg.encode("utf-8"))
            self._serial.flush()

            # Try to read response
            time.sleep(0.02)
            if self._serial.in_waiting > 0:
                resp_line = self._serial.readline().decode("utf-8", errors="ignore").strip()
                if resp_line:
                    return json.loads(resp_line)
        except Exception as e:
            print(f"[INPUT] Serial error: {e}")
        return None

    def _send(self, cmd: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Send command, log it, and return response."""
        log = CommandLog(timestamp=time.time(), command=cmd)
        resp = self._send_raw(cmd)
        log.response = resp
        self._command_log.append(log)
        return resp

    # --- High-Level Game Actions ---------------------------------

    def cast_ability(self, ability: str, x: Optional[int] = None, y: Optional[int] = None):
        """Cast an ability (Q/W/E/R). Optionally aim at screen position."""
        ability = ability.lower()
        if ability not in ("q", "w", "e", "r", "d", "f"):
            print(f"[INPUT] Invalid ability: {ability}")
            return
        if x is not None and y is not None:
            self.move_mouse_to(x, y)
            time.sleep(0.01)
        self._send({"a": "key", "k": ability})

    def move_to(self, x: int, y: int):
        """Right-click at position to move champion."""
        self._send({"a": "click", "b": "right", "x": x, "y": y})

    def attack_move(self, x: int, y: int):
        """Attack-move to position (A-click)."""
        self._send({"a": "keydown", "k": "a"})
        time.sleep(0.02)
        self._send({"a": "click", "b": "left", "x": x, "y": y})
        time.sleep(0.02)
        self._send({"a": "keyup", "k": "a"})

    def right_click(self, x: int, y: int):
        """Right-click at screen position."""
        self._send({"a": "click", "b": "right", "x": x, "y": y})

    def left_click(self, x: int, y: int):
        """Left-click at screen position."""
        self._send({"a": "click", "b": "left", "x": x, "y": y})

    def move_mouse_to(self, x: int, y: int):
        """Move mouse to absolute screen position."""
        self._send({"a": "moveto", "x": x, "y": y})

    def move_mouse_relative(self, dx: int, dy: int):
        """Move mouse relative to current position."""
        self._send({"a": "move", "x": dx, "y": dy})

    def press_key(self, key: str):
        """Press and release a key."""
        self._send({"a": "key", "k": key.lower()})

    def hold_key(self, key: str):
        """Hold a key down."""
        self._send({"a": "keydown", "k": key.lower()})

    def release_key(self, key: str):
        """Release a held key."""
        self._send({"a": "keyup", "k": key.lower()})

    def key_combo(self, *keys: str):
        """Press a key combination (e.g., key_combo('ctrl', 'q'))."""
        self._send({"a": "combo", "keys": [k.lower() for k in keys]})

    def open_shop(self):
        """Open/close the shop (P key in League)."""
        self._send({"a": "key", "k": "p"})

    def center_camera(self):
        """Center camera on champion (Space)."""
        self._send({"a": "key", "k": "space"})

    def level_up_ability(self, ability: str):
        """Level up an ability (Ctrl+Q/W/E/R)."""
        self._send({"a": "combo", "keys": ["ctrl", ability.lower()]})

    def use_item(self, slot: int):
        """Use an active item (slots 1-7)."""
        if 1 <= slot <= 7:
            self._send({"a": "key", "k": str(slot)})

    def recall(self):
        """Start recall (B key)."""
        self._send({"a": "key", "k": "b"})

    def reset(self):
        """Release all keys and buttons."""
        self._send({"a": "reset"})

    def ping(self) -> bool:
        """Ping the Pico to check connectivity."""
        resp = self._send({"a": "ping"})
        return resp is not None and resp.get("s") == "pong"

    def get_command_log(self) -> list:
        """Get the full command log for replay analysis."""
        return self._command_log


class DryRunController(InputController):
    """
    Mock controller that logs commands instead of sending them.
    Use for offline replay testing without a Pico connected.
    """

    def __init__(self):
        super().__init__()
        self._connected = True

    def connect(self) -> bool:
        print("[DRY RUN] Controller initialized (no hardware)")
        return True

    def disconnect(self):
        print("[DRY RUN] Controller stopped")

    def _send_raw(self, cmd: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        return {"s": "ok", "dry_run": True}

    def _send(self, cmd: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        log = CommandLog(timestamp=time.time(), command=cmd)
        resp = {"s": "ok", "dry_run": True}
        log.response = resp
        self._command_log.append(log)

        # Pretty-print the command
        action = cmd.get("a", "?")
        details = {k: v for k, v in cmd.items() if k != "a"}
        detail_str = ", ".join(f"{k}={v}" for k, v in details.items())
        print(f"[DRY RUN] {action}: {detail_str}")
        return resp

    def ping(self) -> bool:
        return True


# --- CLI Test Mode -----------------------------------------------

def run_test(controller: InputController):
    """Run through a test sequence of commands."""
    print("\n=== Input Controller Test Sequence ===\n")

    print("1. Ping...")
    if controller.ping():
        print("   Pong received [OK]")
    else:
        print("   No response [FAIL]")

    print("\n2. Pressing Q (ability cast)...")
    controller.cast_ability("Q")
    time.sleep(0.3)

    print("\n3. Moving mouse to center (960, 540)...")
    controller.move_mouse_to(960, 540)
    time.sleep(0.3)

    print("\n4. Right-clicking at (800, 400) (move command)...")
    controller.move_to(800, 400)
    time.sleep(0.3)

    print("\n5. Attack-move to (1000, 500)...")
    controller.attack_move(1000, 500)
    time.sleep(0.3)

    print("\n6. Level up Q (Ctrl+Q)...")
    controller.level_up_ability("Q")
    time.sleep(0.3)

    print("\n7. Open shop...")
    controller.open_shop()
    time.sleep(0.3)

    print("\n8. Reset (release all)...")
    controller.reset()

    print(f"\n=== Test complete. {len(controller.get_command_log())} commands logged. ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="League Bot Input Controller")
    parser.add_argument("--test", action="store_true", help="Run test sequence")
    parser.add_argument("--dry-run", action="store_true",
                        help="Use dry-run controller (no hardware)")
    parser.add_argument("--port", type=str, default="COM3",
                        help="Serial port for Pico (default: COM3)")
    args = parser.parse_args()

    if args.test:
        if args.dry_run:
            ctrl = DryRunController()
        else:
            ctrl = InputController(port=args.port)
        ctrl.connect()
        run_test(ctrl)
        ctrl.disconnect()
    else:
        parser.print_help()
