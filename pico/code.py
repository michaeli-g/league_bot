"""
code.py – Raspberry Pi Pico League Bot HID Controller

Receives JSON commands over USB serial from the mothership PC and
translates them into keyboard/mouse HID actions on the gaming laptop.

Command Protocol (JSON, newline-terminated):
  {"a":"key",    "k":"q"}                    → Press and release Q
  {"a":"key",    "k":"q", "mod":"shift"}     → Shift+Q
  {"a":"keydown","k":"q"}                    → Hold Q
  {"a":"keyup",  "k":"q"}                    → Release Q
  {"a":"move",   "x":100, "y":-50}           → Move mouse relative
  {"a":"moveto", "x":960, "y":540}           → Move mouse absolute (via incremental steps)
  {"a":"click",  "b":"left"}                 → Left click
  {"a":"click",  "b":"right"}                → Right click
  {"a":"click",  "b":"left", "x":500,"y":300}→ Move then click
  {"a":"combo",  "keys":["ctrl","q"]}        → Press key combo
  {"a":"ping"}                               → Responds {"s":"pong"}
  {"a":"reset"}                              → Release all keys

Copy this to the root of your CIRCUITPY drive.
Requires: adafruit_hid library in /lib/
"""

import json
import time
import supervisor
import usb_cdc

# HID imports
from adafruit_hid.keyboard import Keyboard
from adafruit_hid.keycode import Keycode
from adafruit_hid.mouse import Mouse
import usb_hid

# ─── Initialize HID Devices ──────────────────────────────────────

kbd = Keyboard(usb_hid.devices)
mouse = Mouse(usb_hid.devices)

# ─── Keycode Mapping ─────────────────────────────────────────────
# Maps string names to CircuitPython Keycodes

KEYMAP = {
    # Letters
    "a": Keycode.A, "b": Keycode.B, "c": Keycode.C, "d": Keycode.D,
    "e": Keycode.E, "f": Keycode.F, "g": Keycode.G, "h": Keycode.H,
    "i": Keycode.I, "j": Keycode.J, "k": Keycode.K, "l": Keycode.L,
    "m": Keycode.M, "n": Keycode.N, "o": Keycode.O, "p": Keycode.P,
    "q": Keycode.Q, "r": Keycode.R, "s": Keycode.S, "t": Keycode.T,
    "u": Keycode.U, "v": Keycode.V, "w": Keycode.W, "x": Keycode.X,
    "y": Keycode.Y, "z": Keycode.Z,
    # Numbers
    "1": Keycode.ONE, "2": Keycode.TWO, "3": Keycode.THREE,
    "4": Keycode.FOUR, "5": Keycode.FIVE, "6": Keycode.SIX,
    "7": Keycode.SEVEN,
    # Function keys
    "f1": Keycode.F1, "f2": Keycode.F2, "f3": Keycode.F3, "f4": Keycode.F4,
    "f5": Keycode.F5, "f6": Keycode.F6, "f7": Keycode.F7,
    # Modifiers
    "shift": Keycode.SHIFT, "ctrl": Keycode.CONTROL,
    "alt": Keycode.ALT, "tab": Keycode.TAB,
    # Game-relevant
    "space": Keycode.SPACE, "enter": Keycode.ENTER,
    "esc": Keycode.ESCAPE, "backspace": Keycode.BACKSPACE,
    # League shop / scoreboard
    "p": Keycode.P, "tab": Keycode.TAB,
}

# ─── Serial Setup ────────────────────────────────────────────────
# Use the data serial port (not the REPL console)
serial = usb_cdc.data
if serial is None:
    # Fallback to console if data port not available
    serial = usb_cdc.console

# ─── Command Handlers ────────────────────────────────────────────

def handle_key(cmd):
    """Press and release a single key (with optional modifier)."""
    key_name = cmd.get("k", "").lower()
    mod_name = cmd.get("mod", "").lower()

    kc = KEYMAP.get(key_name)
    if kc is None:
        send_response({"s": "error", "msg": f"unknown key: {key_name}"})
        return

    if mod_name and mod_name in KEYMAP:
        kbd.press(KEYMAP[mod_name], kc)
        time.sleep(0.05)
        kbd.release_all()
    else:
        kbd.press(kc)
        time.sleep(0.05)
        kbd.release(kc)

    send_response({"s": "ok", "a": "key", "k": key_name})


def handle_keydown(cmd):
    """Press and hold a key."""
    key_name = cmd.get("k", "").lower()
    kc = KEYMAP.get(key_name)
    if kc:
        kbd.press(kc)
        send_response({"s": "ok", "a": "keydown", "k": key_name})


def handle_keyup(cmd):
    """Release a key."""
    key_name = cmd.get("k", "").lower()
    kc = KEYMAP.get(key_name)
    if kc:
        kbd.release(kc)
        send_response({"s": "ok", "a": "keyup", "k": key_name})


def handle_move(cmd):
    """Move mouse relative to current position."""
    dx = cmd.get("x", 0)
    dy = cmd.get("y", 0)
    # Mouse.move() takes int8 values (-127 to 127), so we chunk large moves
    while abs(dx) > 0 or abs(dy) > 0:
        step_x = max(-127, min(127, dx))
        step_y = max(-127, min(127, dy))
        mouse.move(x=step_x, y=step_y)
        dx -= step_x
        dy -= step_y
        time.sleep(0.001)
    send_response({"s": "ok", "a": "move"})


def handle_moveto(cmd):
    """
    Move mouse to absolute screen position.
    This is approximate: we move to 0,0 first (via large negative move),
    then move to the target position.
    """
    target_x = cmd.get("x", 960)
    target_y = cmd.get("y", 540)

    # First, slam cursor to top-left corner
    for _ in range(20):
        mouse.move(x=-127, y=-127)
        time.sleep(0.001)

    # Now move to target
    remaining_x = target_x
    remaining_y = target_y
    while abs(remaining_x) > 0 or abs(remaining_y) > 0:
        step_x = max(-127, min(127, remaining_x))
        step_y = max(-127, min(127, remaining_y))
        mouse.move(x=step_x, y=step_y)
        remaining_x -= step_x
        remaining_y -= step_y
        time.sleep(0.001)

    send_response({"s": "ok", "a": "moveto", "x": target_x, "y": target_y})


def handle_click(cmd):
    """Click a mouse button, optionally at a position."""
    button_name = cmd.get("b", "left").lower()
    button = Mouse.LEFT_BUTTON if button_name == "left" else Mouse.RIGHT_BUTTON

    # Optional position
    if "x" in cmd and "y" in cmd:
        handle_moveto(cmd)
        time.sleep(0.01)

    mouse.click(button)
    send_response({"s": "ok", "a": "click", "b": button_name})


def handle_combo(cmd):
    """Press a combination of keys simultaneously."""
    key_names = cmd.get("keys", [])
    keycodes = []
    for kn in key_names:
        kc = KEYMAP.get(kn.lower())
        if kc:
            keycodes.append(kc)
    if keycodes:
        kbd.press(*keycodes)
        time.sleep(0.05)
        kbd.release_all()
    send_response({"s": "ok", "a": "combo", "keys": key_names})


def handle_reset(cmd):
    """Release all keys and buttons."""
    kbd.release_all()
    mouse.release_all()
    send_response({"s": "ok", "a": "reset"})


# ─── Response Helper ─────────────────────────────────────────────

def send_response(data):
    """Send a JSON response back to the mothership."""
    try:
        msg = json.dumps(data) + "\n"
        serial.write(msg.encode("utf-8"))
    except Exception:
        pass  # Don't crash on serial errors


# ─── Action Dispatch ─────────────────────────────────────────────

ACTIONS = {
    "key": handle_key,
    "keydown": handle_keydown,
    "keyup": handle_keyup,
    "move": handle_move,
    "moveto": handle_moveto,
    "click": handle_click,
    "combo": handle_combo,
    "reset": handle_reset,
}

# ─── Main Loop ────────────────────────────────────────────────────

print("League Bot Pico HID Controller v1.0")
print("Waiting for commands on serial...")

buffer = ""

while True:
    if serial.in_waiting > 0:
        try:
            chunk = serial.read(serial.in_waiting)
            buffer += chunk.decode("utf-8", errors="ignore")

            # Process complete lines
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.strip()
                if not line:
                    continue

                if line == "ping":
                    send_response({"s": "pong"})
                    continue

                try:
                    cmd = json.loads(line)
                    action = cmd.get("a", "")

                    if action == "ping":
                        send_response({"s": "pong"})
                    elif action in ACTIONS:
                        ACTIONS[action](cmd)
                    else:
                        send_response({"s": "error", "msg": f"unknown action: {action}"})
                except (ValueError, KeyError) as e:
                    send_response({"s": "error", "msg": str(e)})

        except Exception as e:
            send_response({"s": "error", "msg": str(e)})

    time.sleep(0.001)  # Tiny sleep to prevent busy-wait
