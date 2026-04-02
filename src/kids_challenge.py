"""Sevimli Hayvanlar Yarışması – Single-player animal imitation for kids."""

import math
import os
import sys
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

try:
    from .analyzer import FaceAnalysis
    from .config import AppConfig
    from .text_renderer import draw_text, measure_text
except ImportError:
    from analyzer import FaceAnalysis
    from config import AppConfig
    from text_renderer import draw_text, measure_text

# ── Colors (BGR) ─────────────────────────────────────────
ACCENT_GREEN = (70, 220, 125)
ACCENT_GOLD = (55, 215, 255)
ACCENT_PINK = (180, 130, 255)
PANEL_BG = (14, 18, 24)
TEXT_WHITE = (255, 255, 255)
TEXT_LIGHT = (230, 236, 242)
TEXT_DIM = (170, 180, 195)

KIDS_TASKS = (
    ("MUTLU KÖPEK", "happy", (70, 220, 125), "Köpek"),
    ("ŞAŞKIN BAYKUŞ", "surprised", (75, 200, 255), "Baykuş"),
    ("KIZGIN ASLAN", "angry", (70, 95, 255), "Aslan"),
    ("ÜZGÜN KEDİCİK", "sad", (245, 150, 80), "Kedicik"),
)


def _turkish_upper(text: str) -> str:
    table = str.maketrans(
        {"i": "İ", "ı": "I", "ş": "Ş", "ğ": "Ğ", "ü": "Ü", "ö": "Ö", "ç": "Ç"}
    )
    return text.translate(table).upper()


class KidsChallengeManager:
    """Manages the Sevimli Hayvanlar challenge lifecycle."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.state = "idle"
        self._detected_at: Optional[float] = None
        self._smile_at: Optional[float] = None
        self._countdown_at: Optional[float] = None
        self._started_at: Optional[float] = None
        self._result_until: Optional[float] = None
        self._cooldown_until: Optional[float] = None
        self._task_idx = 0
        self._scores: List[float] = [0.0] * len(KIDS_TASKS)
        self._player: Optional[FaceAnalysis] = None
        self._player_face_id: Optional[int] = None
        self._load_images()

    def _load_images(self) -> None:
        self._images = {}
        module_dir = Path(__file__).resolve().parent
        assets_candidates = [
            module_dir.parent / "assets" / "kids",
            module_dir.parents[1] / "Resources" / "assets" / "kids",
            Path(getattr(sys, "_MEIPASS", module_dir.parent)) / "assets" / "kids",
        ]

        assets_dir = None
        for candidate in assets_candidates:
            if candidate.exists():
                assets_dir = candidate
                break
        if assets_dir is None:
            return

        for animal, filename in [
            ("Köpek", "happy_dog.png"),
            ("Baykuş", "surprised_owl.png"),
            ("Aslan", "angry_lion.png"),
            ("Kedicik", "sad_cat.png"),
        ]:
            path = str(assets_dir / filename)
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is not None:
                self._images[animal] = img

    # ── public API ────────────────────────────────────────

    @property
    def is_active(self) -> bool:
        return self.state != "idle"

    @property
    def blocks_solo(self) -> bool:
        return self.state != "idle"

    @property
    def overlay_analyses(self) -> List[FaceAnalysis]:
        return [self._player] if self._player is not None else []

    def update(self, analyses: List[FaceAnalysis], now: float) -> None:
        handler = {
            "idle": self._tick_idle,
            "invite": self._tick_invite,
            "countdown": self._tick_countdown,
            "active": self._tick_active,
            "result": self._tick_result,
        }.get(self.state)
        if handler:
            handler(analyses, now)

    def draw(self, frame: np.ndarray, now: float) -> None:
        renderer = {
            "invite": self._render_invite,
            "countdown": self._render_countdown,
            "active": self._render_active,
            "result": self._render_result,
        }.get(self.state)
        if renderer:
            renderer(frame, now)

    # ── player assignment ─────────────────────────────────

    def _is_kid(self, face: FaceAnalysis) -> bool:
        return face.age_years is not None and face.age_years <= self.config.kids_max_age

    @staticmethod
    def _face_area(face: FaceAnalysis) -> int:
        return max(face.bbox[2] - face.bbox[0], 1) * max(face.bbox[3] - face.bbox[1], 1)

    def _select_candidate(self, analyses: List[FaceAnalysis]) -> bool:
        candidates = [face for face in analyses if self._is_kid(face)]
        if not candidates:
            self._player = None
            self._player_face_id = None
            return False

        self._player = max(candidates, key=self._face_area)
        self._player_face_id = self._player.face_id
        return True

    def _refresh_player(self, analyses: List[FaceAnalysis], allow_new: bool = False) -> bool:
        if self._player_face_id is not None:
            for analysis in analyses:
                if analysis.face_id == self._player_face_id:
                    self._player = analysis
                    return True
            self._player = None
            if not allow_new:
                return False
            self._player_face_id = None

        if allow_new:
            return self._select_candidate(analyses)

        self._player = None
        return False

    # ── state transitions ─────────────────────────────────

    def _tick_idle(self, analyses: List[FaceAnalysis], now: float) -> None:
        if self._cooldown_until and now < self._cooldown_until:
            return

        previous_player_id = self._player_face_id
        if not self._refresh_player(analyses, allow_new=True):
            self._detected_at = None
            self._smile_at = None
            return

        if previous_player_id != self._player_face_id:
            self._detected_at = now
            self._smile_at = None
            return

        if self._detected_at is None:
            self._detected_at = now
        elif now - self._detected_at >= self.config.kids_invite_delay_seconds:
            self.state = "invite"
            self._detected_at = None

    def _tick_invite(self, analyses: List[FaceAnalysis], now: float) -> None:
        if not self._refresh_player(analyses):
            self._reset()
            return

        h = float(self._player.metrics.get("happy", 0.0))
        if h >= self.config.kids_smile_threshold:
            if self._smile_at is None:
                self._smile_at = now
            elif now - self._smile_at >= self.config.kids_smile_hold_seconds:
                self.state = "countdown"
                self._countdown_at = now
                self._smile_at = None
        else:
            self._smile_at = None

    def _tick_countdown(self, analyses: List[FaceAnalysis], now: float) -> None:
        self._refresh_player(analyses)
        if self._countdown_at and now - self._countdown_at >= self.config.kids_countdown_seconds:
            self.state = "active"
            self._started_at = now
            self._task_idx = 0
            self._scores = [0.0] * len(KIDS_TASKS)

    def _tick_active(self, analyses: List[FaceAnalysis], now: float) -> None:
        self._refresh_player(analyses)
        if not self._started_at:
            return
        elapsed = now - self._started_at
        self._task_idx = min(int(elapsed / self.config.kids_task_seconds), len(KIDS_TASKS) - 1)
        key = KIDS_TASKS[self._task_idx][1]
        
        if self._player:
            self._scores[self._task_idx] = max(self._scores[self._task_idx], float(self._player.metrics.get(key, 0.0)))
            
        if elapsed >= len(KIDS_TASKS) * self.config.kids_task_seconds:
            self.state = "result"
            self._result_until = now + self.config.kids_result_seconds
            self._started_at = None

    def _tick_result(self, analyses: List[FaceAnalysis], now: float) -> None:
        self._refresh_player(analyses)
        if self._result_until and now >= self._result_until:
            self._cooldown_until = now + self.config.kids_cooldown_seconds
            self._reset()

    def _reset(self) -> None:
        self.state = "idle"
        self._detected_at = None
        self._smile_at = None
        self._countdown_at = None
        self._started_at = None
        self._result_until = None
        self._task_idx = 0
        self._player = None
        self._player_face_id = None

    # ═══════════════════════════════════════════════════════
    #  Drawing helpers
    # ═══════════════════════════════════════════════════════

    @staticmethod
    def _darken(frame: np.ndarray, alpha: float = 0.82) -> None:
        ov = frame.copy()
        cv2.rectangle(ov, (0, 0), (frame.shape[1], frame.shape[0]), (6, 8, 12), -1)
        cv2.addWeighted(ov, alpha, frame, 1.0 - alpha, 0.0, frame)

    @staticmethod
    def _panel(frame, x1, y1, x2, y2, accent, alpha=0.88):
        ov = frame.copy()
        cv2.rectangle(ov, (x1, y1), (x2, y2), PANEL_BG, -1)
        cv2.addWeighted(ov, alpha, frame, 1.0 - alpha, 0.0, frame)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (44, 50, 58), 1)
        cv2.rectangle(frame, (x1, y1), (x2, y1 + 4), accent, -1)

    @staticmethod
    def _bar(frame, x1, y, x2, h, value, color, bg=(50, 56, 64)):
        cv2.rectangle(frame, (x1, y), (x2, y + h), bg, -1)
        fw = int((x2 - x1) * max(0.0, min(1.0, value)))
        if fw > 0:
            cv2.rectangle(frame, (x1, y), (x1 + fw, y + h), color, -1)

    @staticmethod
    def _glow(frame, center, radius, color, alpha=0.12):
        g = frame.copy()
        cv2.circle(g, center, radius, color, -1, cv2.LINE_AA)
        cv2.addWeighted(g, alpha, frame, 1.0 - alpha, 0.0, frame)

    @staticmethod
    def _center_text(frame, text, y, size, color):
        w, h = measure_text(text, size)
        x = max((frame.shape[1] - w) // 2, 12)
        draw_text(frame, text, (x, y), size, color)

    @staticmethod
    def _draw_icon(frame, img, x, y, size):
        if img is None:
            return
        
        # Ensure sizes are integers
        size = int(size)
        x, y = int(x), int(y)
        
        resized = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        
        # Create a clean circular blending mask
        mask = np.zeros((size, size), dtype=np.float32)
        radius = size // 2 - 2
        cv2.circle(mask, (size // 2, size // 2), radius, 1.0, -1, cv2.LINE_AA)
        
        y1, y2 = max(0, y), min(frame.shape[0], y + size)
        x1, x2 = max(0, x), min(frame.shape[1], x + size)
        
        draw_h, draw_w = y2 - y1, x2 - x1
        if draw_h <= 0 or draw_w <= 0:
            return
            
        ry1, ry2 = 0, draw_h
        rx1, rx2 = 0, draw_w
        if y < 0:
            ry1 = -y
            ry2 = ry1 + draw_h
        if x < 0:
            rx1 = -x
            rx2 = rx1 + draw_w
            
        roi = frame[y1:y2, x1:x2]
        img_crop = resized[ry1:ry2, rx1:rx2]
        alpha = mask[ry1:ry2, rx1:rx2]
        
        # Blend smoothly
        for c in range(3):
            roi[:, :, c] = roi[:, :, c] * (1.0 - alpha) + img_crop[:, :, c] * alpha

    # ═══════════════════════════════════════════════════════
    #  Invite
    # ═══════════════════════════════════════════════════════

    def _render_invite(self, frame: np.ndarray, now: float) -> None:
        h = float(self._player.metrics.get("happy", 0.0)) if self._player else 0.0

        prog = 0.0
        if self._smile_at:
            prog = min((now - self._smile_at) / self.config.kids_smile_hold_seconds, 1.0)

        cw = min(500, frame.shape[1] - 48)
        ch = 228
        cx = frame.shape[1] // 2
        x1, y1 = cx - cw // 2, 22
        x2, y2 = x1 + cw, y1 + ch

        self._panel(frame, x1, y1, x2, y2, accent=ACCENT_PINK)

        draw_text(frame, "SEVİMLİ HAYVANLAR YARIŞMASI", (x1 + 22, y1 + 16), 18, ACCENT_PINK)
        title = "Hayvan Yüzleri Yap!"
        draw_text(frame, title, (x1 + 22, y1 + 48), 30, TEXT_WHITE)
        _, title_h = measure_text(title, 30)

        icon_size = 52
        icon_gap = 16
        total_icon_w = (icon_size * len(KIDS_TASKS)) + (icon_gap * (len(KIDS_TASKS) - 1))
        x_offset = x1 + max((cw - total_icon_w) // 2, 22)
        icon_y = y1 + 48 + title_h + 20
        for _, _, _, animal_name in KIDS_TASKS:
            img = self._images.get(animal_name)
            if img is not None:
                self._draw_icon(frame, img, x_offset, icon_y, icon_size)
            x_offset += icon_size + icon_gap

        draw_text(
            frame,
            "Gülümse ve 2 saniye tut, yarışma başlasın!",
            (x1 + 22, icon_y + icon_size + 18),
            18,
            TEXT_LIGHT,
        )

        by = y2 - 24
        self._bar(frame, x1 + 22, by, x2 - 22, 12, h, ACCENT_PINK)

        if prog > 0:
            self._bar(frame, x1 + 22, y2 - 12, x2 - 22, 6, prog, ACCENT_GOLD)

    # ═══════════════════════════════════════════════════════
    #  Countdown: 4-3-2-1
    # ═══════════════════════════════════════════════════════

    def _render_countdown(self, frame: np.ndarray, now: float) -> None:
        self._darken(frame, 0.86)
        elapsed = (now - self._countdown_at) if self._countdown_at else 0.0
        remaining = max(self.config.kids_countdown_seconds - elapsed, 0.0)
        number = max(1, int(math.ceil(remaining)))

        fw, fh = frame.shape[1], frame.shape[0]
        cx, cy = fw // 2, fh // 2

        phase = elapsed - math.floor(elapsed)
        flash = 0.70 + 0.30 * abs(math.sin(phase * math.pi))
        progress = 1.0 - remaining / max(self.config.kids_countdown_seconds, 0.01)
        ring_r = 120 + int(progress * 40)

        # Glow ring
        self._glow(frame, (cx, cy + 8), ring_r + 20, ACCENT_PINK, alpha=0.10 * flash)
        cv2.circle(frame, (cx, cy + 8), ring_r, ACCENT_PINK, 5, cv2.LINE_AA)

        self._center_text(frame, "SEVİMLİ HAYVANLAR YARIŞMASI", cy - 150, 26, ACCENT_PINK)
        self._center_text(frame, "Hazır mısın?", cy - 100, 32, TEXT_LIGHT)
        self._center_text(frame, str(number), cy + 44, 180, TEXT_WHITE)

    # ═══════════════════════════════════════════════════════
    #  Active
    # ═══════════════════════════════════════════════════════

    def _render_active(self, frame: np.ndarray, now: float) -> None:
        if not self._started_at:
            return
        elapsed = now - self._started_at
        label, key, color, animal_name = KIDS_TASKS[self._task_idx]
        task_elapsed = elapsed - self._task_idx * self.config.kids_task_seconds
        remaining = max(self.config.kids_task_seconds - task_elapsed, 0.0)

        live = float(self._player.metrics.get(key, 0.0)) if self._player else 0.0
        best = self._scores[self._task_idx]

        fw, fh = frame.shape[1], frame.shape[0]
        cx = fw // 2

        # ── Top panel ──
        ph = 128
        cw = min(620, fw - 48)
        x1 = cx - cw // 2
        x2 = x1 + cw
        self._panel(frame, x1, 20, x2, 20 + ph, accent=color, alpha=0.92)

        meta_x = x2 - 176
        meta_y = 28
        draw_text(frame, "SEVİMLİ HAYVANLAR YARIŞMASI", (x1 + 22, 36), 16, ACCENT_PINK)
        draw_text(frame, label, (x1 + 22, 72), 34, TEXT_WHITE)
        draw_text(frame, f"Görev {self._task_idx + 1}/{len(KIDS_TASKS)}", (meta_x, meta_y), 18, TEXT_DIM)
        draw_text(frame, f"{remaining:.1f} sn", (meta_x, meta_y + 32), 26, color)
        draw_text(frame, f"Skor %{live * 100:.0f}", (meta_x, meta_y + 64), 18, TEXT_WHITE)
        draw_text(frame, f"En iyi %{best * 100:.0f}", (meta_x, meta_y + 88), 16, TEXT_DIM)

        timer_prog = 1.0 - remaining / max(self.config.kids_task_seconds, 0.01)
        self._bar(frame, x1 + 22, 20 + ph - 16, x2 - 22, 8, timer_prog, color)

        # ── Cartoon Animal Next to Face ──
        img = self._images.get(animal_name)
        if img is not None and self._player:
            fx1, fy1, fx2, fy2 = self._player.bbox
            face_w = fx2 - fx1
            face_h = fy2 - fy1
            face_cy = (fy1 + fy2) / 2

            safe_top = 20 + ph + 26
            safe_bottom = fh - 28
            max_h = max(safe_bottom - safe_top, 96)
            right_space = max(fw - fx2 - 48, 96)
            left_space = max(fx1 - 48, 96)
            max_w = max(right_space, left_space)
            icon_size = int(max(face_w, face_h) * 1.02)
            icon_size = max(96, min(icon_size, 220, max_h, max_w))

            icon_x = fx2 + 28
            if right_space < icon_size and left_space >= right_space:
                icon_x = fx1 - icon_size - 28
            icon_x = max(20, min(icon_x, fw - icon_size - 20))

            icon_y = int(face_cy - icon_size / 2)
            icon_y = max(safe_top, min(icon_y, safe_bottom - icon_size))

            if icon_y < fh and icon_y + icon_size > 0:
                self._glow(frame, (int(icon_x + icon_size / 2), int(icon_y + icon_size / 2)), int(icon_size / 1.7), color, alpha=0.34)
                self._draw_icon(frame, img, icon_x, icon_y, icon_size)

    # ═══════════════════════════════════════════════════════
    #  Result
    # ═══════════════════════════════════════════════════════

    def _render_result(self, frame: np.ndarray, now: float) -> None:
        self._darken(frame, 0.88)
        fw, fh = frame.shape[1], frame.shape[0]
        cx = fw // 2

        # Title
        self._center_text(frame, "SEVİMLİ HAYVANLAR YARIŞMASI", 30, 22, ACCENT_PINK)

        # Find best animal
        best_idx = max(range(len(self._scores)), key=lambda i: self._scores[i])
        best_animal = KIDS_TASKS[best_idx][3]
        best_color = KIDS_TASKS[best_idx][2]
        best_score = self._scores[best_idx]
        
        phase = (now * 2.5) % (2.0 * math.pi)
        pulse = 0.08 + 0.06 * abs(math.sin(phase))

        icon_size = min(max(int(min(fw, fh) * 0.24), 120), 220)
        icon_y = 72
        icon_center_y = icon_y + icon_size // 2
        title_y = icon_y + icon_size + 22
        animal_y = title_y + 56
        score_y = animal_y + 54
        note_y = score_y + 92

        # Giant Icon
        best_img = self._images.get(best_animal)
        if best_img is not None:
            self._glow(frame, (cx, icon_center_y), int(icon_size * 0.92), best_color, alpha=pulse * 1.5)
            self._draw_icon(frame, best_img, cx - icon_size // 2, icon_y, icon_size)
        else:
            self._glow(frame, (cx, icon_center_y), int(icon_size * 0.88), best_color, alpha=pulse)

        self._center_text(frame, "HARİKASIN!", title_y, 52, ACCENT_GOLD)
        self._center_text(frame, _turkish_upper(best_animal), animal_y, 30, best_color)
        self._center_text(frame, f"%{best_score * 100:.0f}", score_y, 88, TEXT_WHITE)
        self._center_text(frame, "En iyi taklidin bu oldu", note_y, 24, TEXT_LIGHT)
