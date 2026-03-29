import time
from typing import List

import cv2
import numpy as np
from mediapipe.tasks.python import vision

try:
    from .analyzer import FaceAnalysis
    from .text_renderer import draw_text, measure_text
except ImportError:
    from analyzer import FaceAnalysis
    from text_renderer import draw_text, measure_text

LANDMARK_CONNECTIONS = (
    vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
    vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS,
)

LABEL_COLORS = {
    "Mutlu": (70, 220, 125),
    "Şaşkın": (75, 200, 255),
    "Kızgın": (70, 95, 255),
    "Üzgün": (245, 150, 80),
    "Sakin": (205, 205, 205),
}


def draw_overlay(
    frame: np.ndarray,
    analyses: List[FaceAnalysis],
    fps: float,
    draw_landmarks: bool,
) -> np.ndarray:
    canvas = frame.copy()

    for analysis in analyses:
        _draw_focus_aura(canvas, analysis)
        _draw_face_box(canvas, analysis)
        _draw_expression_label(canvas, analysis)
        if draw_landmarks:
            _draw_landmarks(canvas, analysis.points)

    return canvas


def _label_color(label: str) -> tuple[int, int, int]:
    return LABEL_COLORS.get(label, (205, 205, 205))


def _draw_focus_aura(frame: np.ndarray, analysis: FaceAnalysis) -> None:
    x_min, y_min, x_max, y_max = analysis.bbox
    width = x_max - x_min
    height = y_max - y_min
    color = _label_color(analysis.top_label)

    overlay = frame.copy()
    padding_x = int(width * 0.18)
    padding_y = int(height * 0.16)
    cv2.ellipse(
        overlay,
        (x_min + width // 2, y_min + height // 2),
        (width // 2 + padding_x, height // 2 + padding_y),
        0.0,
        0,
        360,
        color,
        -1,
        cv2.LINE_AA,
    )
    cv2.addWeighted(overlay, 0.08, frame, 0.92, 0.0, frame)


def _draw_expression_label(frame: np.ndarray, analysis: FaceAnalysis) -> None:
    x_min, y_min, x_max, _ = analysis.bbox
    color = _label_color(analysis.top_label)
    title = analysis.top_label
    subtitle = f"~{analysis.age_years:0.0f} YAŞ" if analysis.age_years is not None else (
        analysis.age_label or "Yaş tahmini"
    )

    title_font = cv2.FONT_HERSHEY_DUPLEX
    title_scale = 1.18
    title_thickness = 2
    sub_scale = 0.72
    sub_thickness = 2
    title_font_size = 34
    subtitle_font_size = 22

    title_w, title_h = measure_text(title, title_font_size)
    sub_w, sub_h = measure_text(subtitle, subtitle_font_size)

    card_w = max(title_w, sub_w) + 36
    card_h = title_h + sub_h + 44
    center_x = x_min + (x_max - x_min) // 2
    card_x1 = max(center_x - card_w // 2, 12)
    card_x2 = min(card_x1 + card_w, frame.shape[1] - 12)
    card_x1 = card_x2 - card_w
    card_y2 = max(y_min - 12, card_h + 12)
    card_y1 = card_y2 - card_h

    overlay = frame.copy()
    cv2.rectangle(overlay, (card_x1, card_y1), (card_x2, card_y2), (14, 16, 20), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0.0, frame)

    cv2.rectangle(frame, (card_x1, card_y1), (card_x2, card_y2), color, 2)
    cv2.rectangle(frame, (card_x1, card_y2 - 6), (card_x2, card_y2), color, -1)

    title_x = card_x1 + (card_w - title_w) // 2
    title_y = card_y1 + 12
    sub_x = card_x1 + (card_w - sub_w) // 2
    sub_y = title_y + title_h + 8

    draw_text(frame, title, (title_x, title_y), title_font_size, (250, 250, 250))
    draw_text(frame, subtitle, (sub_x, sub_y), subtitle_font_size, (242, 242, 242))


def _draw_face_box(frame: np.ndarray, analysis: FaceAnalysis) -> None:
    x_min, y_min, x_max, y_max = analysis.bbox
    width = max(x_max - x_min, 1)
    height = max(y_max - y_min, 1)
    corner = max(min(width, height) // 5, 18)
    color = _label_color(analysis.top_label)

    overlay = frame.copy()
    cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), color, -1)
    cv2.addWeighted(overlay, 0.05, frame, 0.95, 0.0, frame)

    for start, end in (
        ((x_min, y_min), (x_min + corner, y_min)),
        ((x_min, y_min), (x_min, y_min + corner)),
        ((x_max, y_min), (x_max - corner, y_min)),
        ((x_max, y_min), (x_max, y_min + corner)),
        ((x_min, y_max), (x_min + corner, y_max)),
        ((x_min, y_max), (x_min, y_max - corner)),
        ((x_max, y_max), (x_max - corner, y_max)),
        ((x_max, y_max), (x_max, y_max - corner)),
    ):
        cv2.line(frame, start, end, color, 3, cv2.LINE_AA)

    tracking = float(analysis.metrics.get("tracking_confidence", 0.0))
    scan_phase = (time.monotonic() * 1.7 + analysis.face_id * 0.27) % 1.0
    scan_y = y_min + int(height * scan_phase)
    scan_alpha = 0.18 + tracking * 0.16
    scan_overlay = frame.copy()
    cv2.line(scan_overlay, (x_min + 6, scan_y), (x_max - 6, scan_y), color, 2, cv2.LINE_AA)
    cv2.addWeighted(scan_overlay, scan_alpha, frame, 1.0 - scan_alpha, 0.0, frame)

    center = (x_min + width // 2, y_min + height // 2)
    radius = max(min(width, height) // 2 + 14, 28)
    cv2.circle(frame, center, radius, color, 1, cv2.LINE_AA)


def _draw_landmarks(frame: np.ndarray, points: np.ndarray) -> None:
    for connection_group in LANDMARK_CONNECTIONS:
        for connection in connection_group:
            start = tuple(points[connection.start].astype(int))
            end = tuple(points[connection.end].astype(int))
            cv2.line(frame, start, end, (90, 225, 130), 1, cv2.LINE_AA)
