from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class AppConfig:
    camera_index: int = 0
    frame_width: int = 1280
    frame_height: int = 720
    max_faces: int = 1
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    smoothing_alpha: float = 0.35
    mirror_preview: bool = True
    draw_landmarks: bool = True
    show_labels: bool = True
    show_header: bool = False
    window_name: str = "AI Yuz Tanima Prototype"
    model_path: str = str(PROJECT_ROOT / "models" / "face_landmarker.task")


DEFAULT_CONFIG = AppConfig()
