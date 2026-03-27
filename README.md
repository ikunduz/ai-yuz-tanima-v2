# AI Yuz Tanima Prototype

This folder contains the first local-only webcam prototype for the museum exhibit idea.

## Setup

```bash
cd ai-yuz-tanima
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/main.py
```

## Controls

- `q` or `ESC`: quit
- `f`: toggle fullscreen
- `l`: toggle landmark lines

## Notes

- Uses the built-in webcam by default on macOS.
- Runs fully local. No internet or remote API is required.
- The first version shows face landmarks and explainable facial signal proxies instead of age/gender predictions.
- The exhibit flow now has `standby`, `acquiring`, `tracking`, and short `hold` states for smoother face loss recovery.
- The most dominant face is selected based on size, position, and short-term tracking stability.
- The first run downloads the MediaPipe face landmarker model once into `models/`.
- On macOS, camera access must be granted to the terminal or Codex app under `System Settings > Privacy & Security > Camera`.
