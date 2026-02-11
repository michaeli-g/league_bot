"""
boot.py – Raspberry Pi Pico CircuitPython Boot Configuration

This runs ONCE at boot before code.py. It configures USB devices.
We enable both USB HID (keyboard + mouse) and USB CDC serial (for commands).

Copy this to the root of your CIRCUITPY drive.
"""

import usb_hid
import usb_cdc

# Enable full keyboard + mouse HID
# The default CircuitPython HID includes keyboard, mouse, and consumer control
usb_hid.enable(
    (
        usb_hid.Device.KEYBOARD,
        usb_hid.Device.MOUSE,
    )
)

# Enable USB CDC serial (data channel for receiving commands from mothership)
# This creates a second serial port alongside the REPL console
usb_cdc.enable(console=True, data=True)
