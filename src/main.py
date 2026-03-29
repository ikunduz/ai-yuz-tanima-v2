import math
import platform
import threading
import time
from typing import List, Optional

import cv2

try:
    from .analyzer import FaceAnalysis, FaceAnalyzer
    from .camera import CameraSource
    from .config import DEFAULT_CONFIG
    from .overlay import draw_overlay
    from .text_renderer import draw_text, measure_text
except ImportError:
    from analyzer import FaceAnalysis, FaceAnalyzer
    from camera import CameraSource
    from config import DEFAULT_CONFIG
    from overlay import draw_overlay
    from text_renderer import draw_text, measure_text


ENGAGE_AFTER_SECONDS = 0.45
HOLD_AFTER_LOSS_SECONDS = 1.1
FPS_SMOOTHING_ALPHA = 0.2


def _turkish_upper(text: str) -> str:
    translation = str.maketrans({
        "i": "İ",
        "ı": "I",
        "ş": "Ş",
        "ğ": "Ğ",
        "ü": "Ü",
        "ö": "Ö",
        "ç": "Ç",
    })
    return text.translate(translation).upper()


class AsyncAnalysisRunner:
    def __init__(self, analyzer: FaceAnalyzer) -> None:
        self._analyzer = analyzer
        self._lock = threading.Lock()
        self._frame_ready = threading.Event()
        self._stop_requested = threading.Event()
        self._worker_error: Optional[BaseException] = None
        self._pending_frame: Optional[cv2.typing.MatLike] = None
        self._latest_analyses: List[FaceAnalysis] = []
        self._thread = threading.Thread(
            target=self._run,
            name="async-face-analyzer",
            daemon=True,
        )
        self._thread.start()

    def submit(self, frame_rgb: cv2.typing.MatLike) -> None:
        if self._stop_requested.is_set():
            return
        with self._lock:
            self._pending_frame = frame_rgb
        self._frame_ready.set()

    def latest_analyses(self) -> List[FaceAnalysis]:
        self._raise_if_failed()
        with self._lock:
            return list(self._latest_analyses)

    def close(self) -> None:
        self._stop_requested.set()
        self._frame_ready.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._raise_if_failed()

    def _run(self) -> None:
        try:
            while not self._stop_requested.is_set():
                self._frame_ready.wait(timeout=0.1)
                if self._stop_requested.is_set():
                    break

                while not self._stop_requested.is_set():
                    with self._lock:
                        frame_rgb = self._pending_frame
                        self._pending_frame = None
                        if frame_rgb is None:
                            self._frame_ready.clear()
                            break

                    analyses = self._analyzer.analyze(frame_rgb)

                    with self._lock:
                        self._latest_analyses = analyses
                        has_pending_frame = self._pending_frame is not None
                        if not has_pending_frame:
                            self._frame_ready.clear()
                    if not has_pending_frame:
                        break
        except BaseException as exc:
            self._worker_error = exc
            self._stop_requested.set()
            self._frame_ready.set()

    def _raise_if_failed(self) -> None:
        if self._worker_error is not None:
            raise RuntimeError("Background analysis worker failed.") from self._worker_error


def main() -> None:
    config = DEFAULT_CONFIG
    camera = CameraSource(config)
    analyzer: Optional[FaceAnalyzer] = None
    analysis_runner: Optional[AsyncAnalysisRunner] = None

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
        analysis_runner = AsyncAnalysisRunner(analyzer)

        while True:
            frame = camera.read()
            if frame is None:
                raise RuntimeError("Camera returned an empty frame.")

            if config.mirror_preview:
                frame = cv2.flip(frame, 1)

            now = time.monotonic()
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            analysis_runner.submit(rgb_frame)
            analyses = analysis_runner.latest_analyses()
            primary_face = FaceAnalyzer.select_primary_face(analyses)

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

                live_start = live_face_started_at
                if live_start is not None:
                    assert live_start is not None
                    dwell_time = now - live_start
                else:
                    dwell_time = 0.0
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
        if analysis_runner is not None:
            analysis_runner.close()
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
    if mode == "standby":
        _draw_standby_card(frame)
        return

    _draw_top_badges(frame, mode, fps)
    if face is not None:
        accent = (60, 220, 120) if mode == "tracking" else (80, 200, 255)
        _draw_signal_strip(frame, face, accent)


def _mode_copy(mode: str) -> tuple[tuple[int, int, int], str, str]:
    if mode == "tracking":
        return (60, 220, 120), "TAKİPTE", "Ana yüz sabitlendi"
    if mode == "acquiring":
        return (80, 200, 255), "ALGILANIYOR", "Yüz takibi oturuyor"
    if mode == "hold":
        return (255, 205, 80), "KISA BEKLEME", "Yüz kısa süreliğine kayboldu"
    return (180, 180, 180), "BEKLEME", "Kameranın önünde bir yüz bekleniyor"


def _draw_signal_strip(
    frame,
    face: FaceAnalysis,
    accent: tuple[int, int, int],
) -> None:
    metrics = face.metrics
    age_value = f"~{face.age_years:0.0f}" if face.age_years is not None else "--"
    age_label = face.age_label or "--"
    panel_h = 110
    panel_y1 = frame.shape[0] - panel_h
    panel_y2 = frame.shape[0]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, panel_y1), (frame.shape[1], panel_y2), (10, 12, 16), -1)
    cv2.addWeighted(overlay, 0.84, frame, 0.16, 0.0, frame)
    cv2.line(frame, (0, panel_y1), (frame.shape[1], panel_y1), accent, 3)

    summary_w = min(360, max(frame.shape[1] // 3, 330))
    cv2.rectangle(frame, (18, panel_y1 + 16), (18 + summary_w, panel_y2 - 16), (18, 22, 28), -1)
    cv2.rectangle(frame, (18, panel_y1 + 16), (18 + summary_w, panel_y2 - 16), accent, 1)
    draw_text(frame, f"YÜZ {face.face_id}", (36, panel_y1 + 20), 24, (245, 245, 245))
    draw_text(frame, age_value, (34, panel_y1 + 50), 46, (255, 255, 255))
    info_x = 160
    draw_text(
        frame,
        f"Takip {metrics['tracking_confidence'] * 100:0.0f}%",
        (info_x, panel_y1 + 26),
        20,
        (220, 226, 232),
    )
    draw_text(
        frame,
        f"Göz {metrics['eye_open'] * 100:0.0f}%",
        (info_x, panel_y1 + 52),
        20,
        (220, 226, 232),
    )
    draw_text(
        frame,
        f"Aralık {age_label}",
        (info_x, panel_y1 + 78),
        20,
        (220, 226, 232),
    )

    meter_x = summary_w + 42
    meter_w = frame.shape[1] - meter_x - 22
    meter_gap = 14
    meter_count = 4
    meter_item_w = max((meter_w - meter_gap * (meter_count - 1)) // meter_count, 80)
    emotions = (
        ("Mutlu", metrics["happy"], (70, 220, 125)),
        ("Şaşkın", metrics["surprised"], (75, 200, 255)),
        ("Kızgın", metrics["angry"], (70, 95, 255)),
        ("Üzgün", metrics["sad"], (245, 150, 80)),
    )
    for index, (label, value, color) in enumerate(emotions):
        card_x = meter_x + index * (meter_item_w + meter_gap)
        _draw_emotion_meter(
            frame=frame,
            x=card_x,
            y=panel_y1 + 16,
            w=meter_item_w,
            h=panel_h - 32,
            label=label,
            value=float(value),
            color=color,
        )


def _draw_emotion_meter(
    frame,
    x: int,
    y: int,
    w: int,
    h: int,
    label: str,
    value: float,
    color: tuple[int, int, int],
) -> None:
    cv2.rectangle(frame, (x, y), (x + w, y + h), (18, 22, 28), -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1)

    draw_text(frame, _turkish_upper(label), (x + 12, y + 8), 19, (245, 245, 245))
    cv2.putText(
        frame,
        f"{value * 100:0.0f}%",
        (x + 12, y + 48),
        cv2.FONT_HERSHEY_DUPLEX,
        0.9,
        color,
        2,
        cv2.LINE_AA,
    )

    bar_x1 = x + 12
    bar_y1 = y + h - 24
    bar_x2 = x + w - 12
    bar_h = 12
    cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x2, bar_y1 + bar_h), (50, 56, 64), -1)
    fill_w = int((bar_x2 - bar_x1) * max(0.0, min(1.0, value)))
    cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x1 + fill_w, bar_y1 + bar_h), color, -1)


def _draw_top_badges(frame, mode: str, fps: float) -> None:
    accent, title, subtitle = _mode_copy(mode)
    labels = (
        title,
        "LOKAL",
        "TEK YÜZ",
        f"{fps:0.0f} FPS",
    )
    font_scale = 0.48
    gap = 8
    padding_x = 10
    padding_y = 7
    x = 18
    y = 16
    start_x = x
    row_bottom = y
    chip_layout = []

    for index, label in enumerate(labels):
        text_w, text_h = measure_text(label, 18)
        baseline = 3
        box_w = text_w + padding_x * 2
        box_h = text_h + baseline + padding_y * 2
        if x + box_w > frame.shape[1] - 18:
            x = start_x
            y = row_bottom + gap
        chip_layout.append((index, label, x, y, box_w, box_h, text_h, baseline))
        x += box_w + gap
        row_bottom = max(row_bottom, y + box_h)

    panel_x1 = 12
    panel_y1 = 10
    panel_x2 = min(max(item[2] + item[4] for item in chip_layout) + 10, frame.shape[1] - 12)
    panel_y2 = min(row_bottom + 34, frame.shape[0] - 12)
    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x1, panel_y1), (panel_x2, panel_y2), (10, 12, 16), -1)
    cv2.addWeighted(overlay, 0.62, frame, 0.38, 0.0, frame)

    for index, label, chip_x, chip_y, box_w, box_h, _text_h, _baseline in chip_layout:
        color = accent if index == 0 else (70, 78, 88)
        overlay = frame.copy()
        cv2.rectangle(overlay, (chip_x, chip_y), (chip_x + box_w, chip_y + box_h), (14, 16, 20), -1)
        cv2.addWeighted(overlay, 0.82, frame, 0.18, 0.0, frame)
        cv2.rectangle(frame, (chip_x, chip_y), (chip_x + box_w, chip_y + box_h), color, 1)
        draw_text(frame, label, (chip_x + padding_x, chip_y + padding_y - 1), 18, (245, 245, 245))

    subtitle_y = min(row_bottom + 22, frame.shape[0] - 18)
    draw_text(frame, subtitle, (20, subtitle_y - 15), 18, (238, 238, 238))


def _draw_standby_card(frame) -> None:
    phase = time.monotonic()
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (6, 10, 16), -1)
    cv2.addWeighted(overlay, 0.48, frame, 0.52, 0.0, frame)

    center_x = frame.shape[1] // 2
    center_y = frame.shape[0] // 2
    pulse = 0.5 + 0.5 * (1.0 + math.sin(phase * 2.2)) / 2.0

    for radius, color, thickness in (
        (86 + int(pulse * 8), (80, 200, 255), 2),
        (116 + int(pulse * 12), (60, 220, 120), 1),
        (148 + int(pulse * 16), (245, 150, 80), 1),
    ):
        cv2.circle(frame, (center_x, center_y - 92), radius, color, thickness, cv2.LINE_AA)
    cv2.circle(frame, (center_x, center_y - 92), 8, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(frame, (center_x, center_y - 92), 22, (80, 200, 255), 2, cv2.LINE_AA)

    _draw_centered_text(frame, "YAPAY ZEKA AYNASI", center_y + 4, 32, (255, 255, 255))
    _draw_centered_text(
        frame,
        "Yüzünü tarat, duygunu gör",
        center_y + 44,
        24,
        (220, 220, 220),
    )
    _draw_centered_text(
        frame,
        "Tek baskın yüz izlenir  |  Tamamen lokal çalışır",
        center_y + 82,
        17,
        (170, 205, 230),
    )
    _draw_standby_chips(frame, center_y + 130)


def _draw_standby_chips(frame, y: int) -> None:
    chips = (
        ("DUYGU", (70, 220, 125)),
        ("YAŞ", (80, 200, 255)),
        ("CANLI", (245, 150, 80)),
    )
    total_w = 0
    chip_sizes = []
    for label, _ in chips:
        text_w, text_h = measure_text(label, 20)
        baseline = 3
        width = text_w + 28
        chip_sizes.append((label, width, text_h + baseline + 14))
        total_w += width
    total_w += (len(chips) - 1) * 14

    x = max((frame.shape[1] - total_w) // 2, 18)
    for (label, color), (_, width, height) in zip(chips, chip_sizes):
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + width, y + height), (12, 16, 22), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0.0, frame)
        cv2.rectangle(frame, (x, y), (x + width, y + height), color, 1)
        draw_text(frame, label, (x + 14, y + 8), 20, (245, 245, 245))
        x += width + 14


def _draw_centered_text(
    frame,
    text: str,
    y: int,
    font_size: int,
    color: tuple[int, int, int],
) -> None:
    width, height = measure_text(text, font_size)
    x = max((frame.shape[1] - width) // 2, 12)
    draw_text(frame, text, (x, y - height), font_size, color)


if __name__ == "__main__":
    main()
