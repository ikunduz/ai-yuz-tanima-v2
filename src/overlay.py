from typing import Iterable, List

import cv2
import numpy as np
from mediapipe.tasks.python import vision

try:
    from .analyzer import FaceAnalysis
except ImportError:
    from analyzer import FaceAnalysis

LANDMARK_CONNECTIONS = (
    vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
    vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS,
)


def draw_overlay(
    frame: np.ndarray,
    analyses: List[FaceAnalysis],
    fps: float,
    draw_landmarks: bool,
) -> np.ndarray:
    canvas = frame.copy()

    for analysis in analyses:
        _draw_face_box(canvas, analysis)
        _draw_expression_label(canvas, analysis)
        # _draw_metrics_panel(canvas, analysis)  # User said only bottom values
        if draw_landmarks:
            _draw_landmarks(canvas, analysis.points)

    return canvas


def _draw_expression_label(frame: np.ndarray, analysis: FaceAnalysis) -> None:
    x_min, y_min, x_max, _ = analysis.bbox
    label = analysis.top_label
    
    # Text appearance
    font = cv2.FONT_HERSHEY_DUPLEX
    scale = 0.95
    thickness = 2
    
    # Get text size for background box
    (w, h), baseline = cv2.getTextSize(label, font, scale, thickness)
    
    # Position: Center above the face box
    label_x = x_min + (x_max - x_min) // 2 - w // 2
    label_y = y_min - 15
    
    # Draw background box (Neon glow style)
    padding = 8
    box_x1 = label_x - padding
    box_y1 = label_y - h - padding
    box_x2 = label_x + w + padding
    box_y2 = label_y + padding
    
    # Color mapping
    colors = {
        "Mutlu": (60, 220, 120),    # Emerald
        "Saskin": (80, 200, 255),   # Cloud Blue
        "Kizgin": (60, 60, 255),    # Vibrant Red
        "Uzgun": (255, 120, 60),    # Deep Orange/Blue
        "Notr": (180, 180, 180)     # Muted Grey
    }
    color = colors.get(label, (180, 180, 180))
    
    # Semi-transparent dark background
    overlay = frame.copy()
    cv2.rectangle(overlay, (box_x1, box_y1), (box_x2, box_y2), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0.0, frame)
    
    # Accent line
    cv2.line(frame, (box_x1, box_y2), (box_x2, box_y2), color, 2)
    
    # Text
    cv2.putText(
        frame,
        label,
        (label_x, label_y),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def _draw_face_box(frame: np.ndarray, analysis: FaceAnalysis) -> None:
    x_min, y_min, x_max, y_max = analysis.bbox
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (60, 220, 120), 2)


def _draw_metrics_panel(frame: np.ndarray, analysis: FaceAnalysis) -> None:
    x_min, y_min, _, _ = analysis.bbox
    metrics = analysis.metrics
    rows: Iterable[tuple[str, float]] = (
        ("Takip", metrics["tracking_confidence"]),
        ("Goz Acikligi", metrics["eye_open"]),
        ("Mutlu", metrics["happy"]),
        ("Saskin", metrics["surprised"]),
        ("Kizgin", metrics["angry"]),
        ("Uzgun", metrics["sad"]),
    )

    panel_x = max(x_min - 10, 10)
    panel_y = max(y_min - 170, 10)
    panel_w = 280
    panel_h = 150

    cv2.rectangle(
        frame,
        (panel_x, panel_y),
        (panel_x + panel_w, panel_y + panel_h),
        (18, 18, 18),
        -1,
    )
    cv2.rectangle(
        frame,
        (panel_x, panel_y),
        (panel_x + panel_w, panel_y + panel_h),
        (80, 200, 255),
        1,
    )

    cv2.putText(
        frame,
        "YUZ ANALIZI",
        (panel_x + 12, panel_y + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    y = panel_y + 48
    for label, value in rows:
        _draw_metric_bar(frame, label, value, panel_x + 12, y, panel_w - 24)
        y += 22


def _draw_landmarks(frame: np.ndarray, points: np.ndarray) -> None:
    for connection_group in LANDMARK_CONNECTIONS:
        for connection in connection_group:
            start = tuple(points[connection.start].astype(int))
            end = tuple(points[connection.end].astype(int))
            cv2.line(frame, start, end, (80, 220, 120), 1, cv2.LINE_AA)


def _draw_metric_bar(
    frame: np.ndarray,
    label: str,
    value: float,
    x: int,
    y: int,
    width: int,
) -> None:
    bar_x = x + 110
    bar_y = y - 10
    bar_w = width - 120
    bar_h = 10
    filled_w = int(bar_w * max(0.0, min(1.0, value)))

    cv2.putText(
        frame,
        label,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled_w, bar_y + bar_h), (80, 200, 255), -1)


def _draw_header(frame: np.ndarray, fps: float, face_count: int) -> None:
    label = f"Yuz: {face_count} | FPS: {fps:.1f} | q/ESC cikis | f tam ekran"
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 36), (12, 12, 12), -1)
    cv2.putText(
        frame,
        label,
        (16, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
