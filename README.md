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
- Shows face landmarks, expression labels, and local age estimation.
- The exhibit flow now has `standby`, `acquiring`, `tracking`, and short `hold` states for smoother face loss recovery.
- The most dominant face is selected based on size, position, and short-term tracking stability.
- The first run downloads the MediaPipe face landmarker model and the MiVOLO v2 checkpoint cache into `models/`.
- Age estimation now uses MiVOLO v2 with face + upper-body crops. It is heavier than the old OpenVINO model but substantially better for age stability.
- Emotion estimation can now be switched away from hand-written blendshape rules to EmotiEffLib. The current default uses `enet_b2_7` over ONNX for a cleaner angry/sad separation test.
- The MiVOLO checkpoint is downloaded from Hugging Face on first run, so internet is required once for the initial cache fill.
- MiVOLO is pinned to `timm==0.8.13.dev0`, which matches the checkpoint's expected model code.
- On macOS, camera access must be granted to the terminal or Codex app under `System Settings > Privacy & Security > Camera`.
