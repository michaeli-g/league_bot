# Raspberry Pi Pico – HID Controller Setup Guide

## What This Does
The Pico acts as a **USB keyboard and mouse** plugged into your gaming laptop.
The mothership PC sends commands to the Pico over serial, and the Pico
"types" keys and moves the mouse as if a human were playing.

## Hardware Needed
- **Raspberry Pi Pico** (RP2040, non-W version is fine)
- **USB Micro-B cable** (data cable, not charge-only)
- Optionally: a second USB cable or jumper wires for serial from mothership

## Connection Topology

### Option A: Single USB + Network Commands (Simplest)
```
Gaming Laptop ◄─── USB (HID) ──── Pi Pico
Mothership PC ──── WiFi/LAN ────► Gaming Laptop ──── USB Serial ──► Pi Pico
```
The mothership sends serial commands over the **same USB cable** that provides HID.
The laptop forwards serial data through a Python relay script, or you can share
the COM port over the network.

### Option B: Dual Connection (Recommended)
```
Gaming Laptop ◄─── USB (HID) ──── Pi Pico ──── UART Wires ──── Mothership GPIO/USB-Serial
```
- USB cable from Pico to laptop (HID)
- UART wires from Pico GPIO to a USB-to-UART adapter on the mothership
- Pico GPIO 0 (TX) → Adapter RX
- Pico GPIO 1 (RX) → Adapter TX
- GND → GND

> **Note:** The current firmware uses USB CDC serial. For Option B with UART,
> you'd modify `code.py` to read from `board.UART()` instead of `usb_cdc`.

## Flashing CircuitPython

### Step 1: Download CircuitPython
1. Go to https://circuitpython.org/board/raspberry_pi_pico/
2. Download the latest `.uf2` file (9.x recommended)

### Step 2: Flash the Firmware
1. **Unplug** the Pico from everything
2. **Hold the BOOTSEL button** on the Pico
3. While holding BOOTSEL, **plug the Pico into your PC via USB**
4. Release BOOTSEL – a drive called `RPI-RP2` appears
5. **Drag and drop** the `.uf2` file onto `RPI-RP2`
6. The Pico reboots and a new drive called `CIRCUITPY` appears

### Step 3: Install adafruit_hid Library
1. Go to https://circuitpython.org/libraries
2. Download the **Bundle for your CircuitPython version** (e.g., `adafruit-circuitpython-bundle-9.x-...zip`)
3. Unzip the bundle
4. From the `lib/` folder in the bundle, copy the **`adafruit_hid`** folder
5. Paste it into `CIRCUITPY/lib/` on your Pico

Your Pico `lib/` folder should look like:
```
CIRCUITPY/
├── lib/
│   └── adafruit_hid/
│       ├── __init__.py
│       ├── keyboard.py
│       ├── keycode.py
│       ├── mouse.py
│       └── ...
├── boot.py       ← copy from pico/boot.py
└── code.py       ← copy from pico/code.py
```

### Step 4: Copy Bot Firmware
1. Copy `pico/boot.py` → `CIRCUITPY/boot.py`
2. Copy `pico/code.py` → `CIRCUITPY/code.py`
3. The Pico will reboot automatically

### Step 5: Verify
1. Plug the Pico into your **gaming laptop**
2. The laptop should recognize it as a keyboard and mouse
3. On the **mothership PC**, run:
   ```bash
   python input_controller.py --test --port COM3
   ```
   (Replace COM3 with the actual port – check Device Manager)
4. You should see keystrokes appearing on the laptop!

## Testing Without Laptop
You can test the Pico by plugging it into **any PC** and running:
```bash
python input_controller.py --test --dry-run
```
This runs through all commands in dry-run mode (no serial needed).

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `CIRCUITPY` drive doesn't appear | Re-flash CircuitPython (BOOTSEL + plug in) |
| `adafruit_hid` import error | Make sure `adafruit_hid/` is in `CIRCUITPY/lib/` |
| No COM port appears | Try a different USB cable (must be data, not charge-only) |
| Keys not registering | Check that `boot.py` is on the Pico and it was rebooted |
| Serial timeout | Increase timeout in `InputController.__init__()` |
