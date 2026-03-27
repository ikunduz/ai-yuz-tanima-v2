import platform
import time
from typing import Optional

import cv2

try:
    from .analyzer import FaceAnalysis, FaceAnalyzer
    from .camera import CameraSource
    from .config import DEFAULT_CONFIG
    from .overlay import draw_overlay
except ImportError:
    from analyzer import FaceAnalysis, FaceAnalyzer
    from camera import CameraSource
    from config import DEFAULT_CONFIG
    from overlay import draw_overlay


ENGAGE_AFTER_SECONDS = 0.45
HOLD_AFTER_LOSS_SECONDS = 1.1
FPS_SMOOTHING_ALPHA = 0.2


def main() -> None:
    config = DEFAULT_CONFIG
    camera = CameraSource(config)
    analyzer: Optional[FaceAnalyzer] = None

    last_frame_ts = time.monotonic()
    display_fps = 0.0
    fullscreen = False
    draw_landmarks = config.draw_landmarks

    live_face_started_at: Optional[float] = None
    last_face_seen_at = 0.0
    last_face_snapshot: Optional[FaceAnalysis] = None
    active_face_id: Optional[int] = None

    cv2.namedWindow(config.window_name, cv2.WINDOW_NORMAL)

    try:
        _open_camera_or_raise(camera)
        analyzer = FaceAnalyzer(config)

        while True:
            frame = camera.read()
            if frame is None:
                raise RuntimeError("Camera returned an empty frame.")

            if config.mirror_preview:
                frame = cv2.flip(frame, 1)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            analyses = analyzer.analyze(rgb_frame)
            primary_face = analyzer.select_primary_face(analyses)

            now = time.monotonic()
            raw_fps = 1.0 / max(now - last_frame_ts, 1e-6)
            last_frame_ts = now
            if display_fps == 0.0:
                display_fps = raw_fps
            else:
                display_fps = (
                    display_fps * (1.0 - FPS_SMOOTHING_ALPHA)
                    + raw_fps * FPS_SMOOTHING_ALPHA
                )

            mode = "standby"
            face_for_overlay: Optional[FaceAnalysis] = None
            status_face: Optional[FaceAnalysis] = None

            if primary_face is not None:
                if primary_face.face_id != active_face_id:
                    active_face_id = primary_face.face_id
                    live_face_started_at = now
                elif live_face_started_at is None:
                    live_face_started_at = now

                last_face_seen_at = now
                last_face_snapshot = primary_face
                face_for_overlay = primary_face
                status_face = primary_face

                dwell_time = now - live_face_started_at
                tracking_ready = primary_face.metrics["tracking_confidence"] >= 0.6
                mode = "tracking" if dwell_time >= ENGAGE_AFTER_SECONDS and tracking_ready else "acquiring"
            else:
                if (
                    last_face_snapshot is not None
                    and now - last_face_seen_at <= HOLD_AFTER_LOSS_SECONDS
                ):
                    mode = "hold"
                    status_face = last_face_snapshot
                else:
                    active_face_id = None
                    live_face_started_at = None
                    last_face_snapshot = None

            output = draw_overlay(
                frame=frame,
                analyses=[face_for_overlay] if face_for_overlay is not None else [],
                fps=display_fps,
                draw_landmarks=draw_landmarks and face_for_overlay is not None,
            )
            _draw_runtime_chrome(
                frame=output,
                mode=mode,
                fps=display_fps,
                face=status_face,
                draw_landmarks=draw_landmarks,
            )

            cv2.imshow(config.window_name, output)
            key = cv2.waitKey(1) & 0xFF

            if key in (27, ord("q")):
                break

            if key == ord("f"):
                fullscreen = not fullscreen
                mode_value = cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL
                cv2.setWindowProperty(
                    config.window_name,
                    cv2.WND_PROP_FULLSCREEN,
                    mode_value,
                )

            if key == ord("l"):
                draw_landmarks = not draw_landmarks
    finally:
        if analyzer is not None:
            analyzer.close()
        camera.release()
        cv2.destroyAllWindows()


def _open_camera_or_raise(camera: CameraSource) -> None:
    try:
        camera.open()
    except RuntimeError as exc:
        message = "Camera could not be opened."
        if platform.system() == "Darwin":
            message += (
                " macOS camera izni gerekli. System Settings > Privacy & Security > "
                "Camera altindan bu scripti calistiran uygulamaya izin verip tekrar deneyin."
            )
        else:
            message += " Webcam baska bir uygulama tarafindan kullaniliyor olabilir."
        raise RuntimeError(message) from exc


def _draw_runtime_chrome(
    frame,
    mode: str,
    fps: float,
    face: Optional[FaceAnalysis],
    draw_landmarks: bool,
) -> None:
    # Top header removed per request. Only standby card and bottom signals.
    if mode == "standby":
        _draw_standby_card(frame)
        return

    # Draw bottom signal strip for calibration
    if face is not None:
        # Determine accent color from mode
        accent = (60, 220, 120) if mode == "tracking" else (80, 200, 255)
        _draw_signal_strip(frame, face, accent)


def _mode_copy(mode: str) -> tuple[tuple[int, int, int], str, str]:
    if mode == "tracking":
        return (60, 220, 120), "TAKIPTE", "Ana yuz sabitlendi"
    if mode == "acquiring":
        return (80, 200, 255), "ALGILANIYOR", "Yuz noktalarinin takibi oturuyor"
    if mode == "hold":
        return (255, 205, 80), "KISA BEKLEME", "Yuz kisa sureligine kayboldu"
    return (180, 180, 180), "BEKLEME", "Kameranin onunde bir yuz bekleniyor"


def _draw_signal_strip(
    frame,
    face: FaceAnalysis,
    accent: tuple[int, int, int],
) -> None:
    metrics = face.metrics
    labels = (
        f"Yuz {face.face_id}",
        f"Takip {metrics['tracking_confidence'] * 100:0.0f}%",
        f"Goz {metrics['eye_open'] * 100:0.0f}%",
        f"Mutlu {metrics['happy'] * 100:0.0f}%",
        f"Saskin {metrics['surprised'] * 100:0.0f}%",
        f"Kizgin {metrics['angry'] * 100:0.0f}%",
        f"Uzgun {metrics['sad'] * 100:0.0f}%",
    )

    top = frame.shape[0] - 42
    cv2.rectangle(frame, (0, top - 22), (frame.shape[1], frame.shape[0]), (12, 12, 12), -1)
    cv2.line(frame, (0, top - 22), (frame.shape[1], top - 22), accent, 2)

    x = 16
    for label in labels:
        cv2.putText(
            frame,
            label,
            (x, top),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.54,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )
        x += 138


def _draw_standby_card(frame) -> None:
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (6, 12, 18), -1)
    cv2.addWeighted(overlay, 0.36, frame, 0.64, 0.0, frame)

    center_x = frame.shape[1] // 2
    center_y = frame.shape[0] // 2

    cv2.circle(frame, (center_x, center_y - 76), 74, (80, 200, 255), 2, cv2.LINE_AA)
    cv2.circle(frame, (center_x, center_y - 76), 6, (80, 200, 255), -1, cv2.LINE_AA)

    _draw_centered_text(frame, "YAPAY ZEKA AYNASI", center_y + 10, 1.05, (255, 255, 255), 2)
    _draw_centered_text(
        frame,
        "Kameranin onune gecin",
        center_y + 48,
        0.78,
        (220, 220, 220),
        2,
    )
    _draw_centered_text(
        frame,
        "Tek baskin yuz izlenir | Tamamen lokal calisir",
        center_y + 84,
        0.58,
        (170, 205, 230),
        1,
    )


def _draw_centered_text(
    frame,
    text: str,
    y: int,
    scale: float,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    (width, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x = max((frame.shape[1] - width) // 2, 12)
    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


if __name__ == "__main__":
    main()
