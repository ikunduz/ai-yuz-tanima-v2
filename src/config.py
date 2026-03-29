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
    emotion_backend: str = "emotiefflib"
    emotion_emotiefflib_engine: str = "onnx"
    emotion_emotiefflib_model_name: str = "enet_b2_7"
    emotion_inference_interval_ms: int = 140
    emotion_min_face_width: int = 88
    emotion_crop_padding_ratio: float = 0.16
    emotion_crop_top_bias_ratio: float = 0.10
    emotion_context_padding_ratio: float = 0.28
    emotion_context_top_bias_ratio: float = 0.14
    emotion_model_base_weight: float = 0.66
    emotion_model_min_weight: float = 0.42
    emotion_model_max_weight: float = 0.82
    emotion_stability_max_motion: float = 0.11
    emotion_stability_max_yaw_ratio: float = 0.20
    emotion_stability_max_roll_degrees: float = 14.0
    emotion_label_threshold: float = 0.30
    emotion_label_hold_threshold: float = 0.24
    emotion_label_switch_margin: float = 0.08
    emotion_label_switch_frames: int = 2
    mirror_preview: bool = True
    draw_landmarks: bool = True
    show_labels: bool = True
    show_header: bool = False
    window_name: str = "AI Yüz Tanıma Prototype"
    model_path: str = str(PROJECT_ROOT / "models" / "face_landmarker.task")
    age_enabled: bool = True
    age_backend: str = "mivolo"
    age_mivolo_model_id: str = "iitolstykh/mivolo_v2"
    age_mivolo_cache_dir: str = str(PROJECT_ROOT / "models" / "mivolo_v2")
    age_mivolo_device: str = "auto"
    age_model_precision: str = "FP16"
    age_model_path: str = str(
        PROJECT_ROOT
        / "models"
        / "age-gender-recognition-retail-0013"
        / "FP16"
        / "age-gender-recognition-retail-0013.xml"
    )
    age_inference_interval_ms: int = 320
    age_smoothing_alpha: float = 0.22
    age_min_face_width: int = 84
    age_crop_padding_ratio: float = 0.18
    age_crop_top_bias_ratio: float = 0.08
    age_context_padding_ratio: float = 0.32
    age_context_top_bias_ratio: float = 0.18
    age_aligned_crop_size: int = 224
    age_body_width_scale: float = 3.4
    age_body_above_face_scale: float = 1.15
    age_body_below_face_scale: float = 4.8
    age_bias_years: float = 0.0
    age_history_size: int = 10
    age_history_trim: int = 2
    gray_hair_bonus_max_years: float = 0.0
    age_stability_required_frames: int = 3
    age_stability_min_tracking_confidence: float = 0.82
    age_stability_min_presence: float = 0.55
    age_stability_max_expression: float = 0.22
    age_stability_max_mouth_open: float = 0.18
    age_stability_max_motion: float = 0.055
    age_stability_max_yaw_ratio: float = 0.14
    age_stability_max_roll_degrees: float = 10.0


DEFAULT_CONFIG = AppConfig()
