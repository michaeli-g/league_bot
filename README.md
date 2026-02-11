# League Bot – AI Game Agent

An AI-powered League of Legends bot using distributed hardware:
- **Gaming Laptop**: Runs the game
- **Capture Card**: Sends video feed to mothership
- **Mothership PC (RTX 5080)**: YOLO vision + Ollama LLM tactical reasoning
- **Raspberry Pi Pico**: USB HID bridge – sends keyboard/mouse input to laptop

## Architecture

```
┌─────────────┐    HDMI     ┌──────────────┐    USB     ┌─────────────────┐
│   Gaming     │───────────►│ Capture Card  │──────────►│   Mothership PC  │
│   Laptop     │            └──────────────┘           │   (RTX 5080)     │
│   (League)   │◄──── USB HID ────┐                     │                  │
└─────────────┘               │                     │  YOLO + Ollama   │
                          ┌───┴───────┐              │                  │
                          │  Pi Pico   │◄── Serial ──│  Python Brain    │
                          │  (HID)     │              └─────────────────┘
                          └───────────┘
```

## Quick Start

### 1. Python Environment (Mothership PC)
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Install Ollama
Download from https://ollama.com, then:
```bash
ollama pull llama3.2-vision
```

### 3. Flash Raspberry Pi Pico
See [pico/SETUP.md](pico/SETUP.md) for detailed instructions.

### 4. Running

#### Replay Mode (Offline Testing – Start Here!)
```bash
# Dry run with a replay video (no Pico needed)
python mothership_brain.py --mode replay --replay-file path/to/gameplay.mp4 --dry-run

# With visual debug overlay
python mothership_brain.py --mode replay --replay-file path/to/gameplay.mp4 --dry-run --show-preview
```

#### Capture Card Mode (Live)
```bash
# Find your capture card device index
python video_input.py --list-devices

# Run live with Pico connected
python mothership_brain.py --mode capture_card --device-index 1 --pico-port COM3
```

#### Record Gameplay for Replay
```bash
python replay_extractor.py --record --device-index 1 --output replays/my_game.mp4
```

### 5. Training Custom YOLO Model
```bash
# Collect training screenshots from a replay
python vision/collect_training_data.py --source path/to/gameplay.mp4 --output-dir vision/dataset

# Train the model
python vision/train.py --epochs 100 --data vision/data.yaml
```

## Project Structure
```
league_bot/
├── mothership_brain.py        # Main brain – orchestrates everything
├── video_input.py             # Unified video source (replay/capture card/shm)
├── input_controller.py        # Serial bridge to Pico HID controller
├── replay_extractor.py        # Record & extract frames from gameplay
├── verify_vision.py           # Quick shared memory viewer
├── requirements.txt           # Python dependencies
├── capturer/                  # C++ DXGI screen capturer (same-PC mode)
│   ├── CMakeLists.txt
│   └── src/
├── pico/                      # Raspberry Pi Pico firmware
│   ├── boot.py                # HID + serial config
│   ├── code.py                # Main firmware (serial → HID)
│   └── SETUP.md               # Flash & wiring guide
└── vision/                    # YOLO League detection
    ├── league_detector.py     # League-specific YOLO wrapper
    ├── data.yaml              # Dataset config
    ├── collect_training_data.py
    └── train.py
```
