"""İki Kişilik Yarış – Competitive two-player emotion challenge."""

import math
import time
from typing import List, Optional, Tuple

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
P1_ACCENT = (255, 185, 60)
P1_DIM = (180, 130, 40)
P2_ACCENT = (55, 160, 255)
P2_DIM = (40, 110, 180)
GOLD = (55, 215, 255)
PANEL_BG = (14, 18, 24)
TEXT_WHITE = (255, 255, 255)
TEXT_LIGHT = (230, 236, 242)
TEXT_DIM = (170, 180, 195)
DIVIDER = (50, 56, 64)

DUO_TASKS = (
    ("Mutlu ol!", "happy", (70, 220, 125)),
    ("Şaşır!", "surprised", (75, 200, 255)),
    ("Kızgın görün!", "angry", (70, 95, 255)),
    ("Üzgün görün!", "sad", (245, 150, 80)),
)

RESULT_LABELS = ("Mutlu", "Şaşkın", "Kızgın", "Üzgün")


def _turkish_upper(text: str) -> str:
    table = str.maketrans(
        {"i": "İ", "ı": "I", "ş": "Ş", "ğ": "Ğ", "ü": "Ü", "ö": "Ö", "ç": "Ç"}
    )
    return text.translate(table).upper()


# ═══════════════════════════════════════════════════════════
#  State Machine
# ═══════════════════════════════════════════════════════════


class DuoChallengeManager:
    """Manages the two-player emotion challenge lifecycle."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.state = "idle"
        self._detected_at: Optional[float] = None
        self._smile_at: Optional[float] = None
        self._intro_at: Optional[float] = None
        self._countdown_at: Optional[float] = None
        self._started_at: Optional[float] = None
        self._result_until: Optional[float] = None
        self._cooldown_until: Optional[float] = None
        self._task_idx = 0
        self._s1: List[float] = [0.0] * len(DUO_TASKS)
        self._s2: List[float] = [0.0] * len(DUO_TASKS)
        self._p1: Optional[FaceAnalysis] = None
        self._p2: Optional[FaceAnalysis] = None

    # ── public API ────────────────────────────────────────

    @property
    def is_active(self) -> bool:
        return self.state != "idle"

    @property
    def blocks_solo(self) -> bool:
        return self.state != "idle"

    def update(self, analyses: List[FaceAnalysis], now: float) -> None:
        handler = {
            "idle": self._tick_idle,
            "invite": self._tick_invite,
            "intro": self._tick_intro,
            "countdown": self._tick_countdown,
            "active": self._tick_active,
            "result": self._tick_result,
        }.get(self.state)
        if handler:
            handler(analyses, now)

    def draw(self, frame: np.ndarray, now: float) -> None:
        renderer = {
            "invite": self._render_invite,
            "intro": self._render_intro,
            "countdown": self._render_countdown,
            "active": self._render_active,
            "result": self._render_result,
        }.get(self.state)
        if renderer:
            renderer(frame, now)

    # ── player assignment ─────────────────────────────────

    def _assign(self, analyses: List[FaceAnalysis]) -> None:
        if len(analyses) < 2:
            return
        s = sorted(analyses[:2], key=lambda f: (f.bbox[0] + f.bbox[2]) / 2.0)
        self._p1, self._p2 = s[0], s[1]

    # ── state transitions ─────────────────────────────────

    def _tick_idle(self, analyses: List[FaceAnalysis], now: float) -> None:
        if self._cooldown_until and now < self._cooldown_until:
            return
        if len(analyses) >= 2:
            if self._detected_at is None:
                self._detected_at = now
            elif now - self._detected_at >= self.config.duo_detection_seconds:
                self._assign(analyses)
                self.state = "invite"
                self._detected_at = None
        else:
            self._detected_at = None
            self._smile_at = None

    def _tick_invite(self, analyses: List[FaceAnalysis], now: float) -> None:
        if len(analyses) < 2:
            self._reset()
            return
        self._assign(analyses)
        h1 = float(self._p1.metrics.get("happy", 0.0)) if self._p1 else 0.0
        h2 = float(self._p2.metrics.get("happy", 0.0)) if self._p2 else 0.0
        th = self.config.duo_smile_threshold
        if h1 >= th and h2 >= th:
            if self._smile_at is None:
                self._smile_at = now
            elif now - self._smile_at >= self.config.duo_smile_hold_seconds:
                self.state = "intro"
                self._intro_at = now
                self._smile_at = None
        else:
            self._smile_at = None

    def _tick_intro(self, analyses: List[FaceAnalysis], now: float) -> None:
        if len(analyses) >= 2:
            self._assign(analyses)
        if self._intro_at and now - self._intro_at >= self.config.duo_intro_seconds:
            self.state = "countdown"
            self._countdown_at = now

    def _tick_countdown(self, analyses: List[FaceAnalysis], now: float) -> None:
        if len(analyses) >= 2:
            self._assign(analyses)
        if self._countdown_at and now - self._countdown_at >= self.config.duo_countdown_seconds:
            self.state = "active"
            self._started_at = now
            self._task_idx = 0
            self._s1 = [0.0] * len(DUO_TASKS)
            self._s2 = [0.0] * len(DUO_TASKS)

    def _tick_active(self, analyses: List[FaceAnalysis], now: float) -> None:
        if len(analyses) >= 2:
            self._assign(analyses)
        if not self._started_at:
            return
        elapsed = now - self._started_at
        self._task_idx = min(int(elapsed / self.config.duo_task_seconds), len(DUO_TASKS) - 1)
        key = DUO_TASKS[self._task_idx][1]
        if self._p1:
            self._s1[self._task_idx] = max(self._s1[self._task_idx], float(self._p1.metrics.get(key, 0.0)))
        if self._p2:
            self._s2[self._task_idx] = max(self._s2[self._task_idx], float(self._p2.metrics.get(key, 0.0)))
        if elapsed >= len(DUO_TASKS) * self.config.duo_task_seconds:
            self.state = "result"
            self._result_until = now + self.config.duo_result_seconds
            self._started_at = None

    def _tick_result(self, analyses: List[FaceAnalysis], now: float) -> None:
        if len(analyses) >= 2:
            self._assign(analyses)
        if self._result_until and now >= self._result_until:
            self._cooldown_until = now + self.config.duo_cooldown_seconds
            self._reset()

    def _reset(self) -> None:
        self.state = "idle"
        self._detected_at = None
        self._smile_at = None
        self._intro_at = None
        self._countdown_at = None
        self._started_at = None
        self._result_until = None
        self._task_idx = 0
        self._p1 = None
        self._p2 = None

    def _winner(self) -> str:
        w1 = sum(1 for a, b in zip(self._s1, self._s2) if a > b)
        w2 = sum(1 for a, b in zip(self._s1, self._s2) if b > a)
        if w1 > w2:
            return "p1"
        if w2 > w1:
            return "p2"
        return "draw"

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

    # ═══════════════════════════════════════════════════════
    #  Invite: "Birlikte gülümseyin!"
    # ═══════════════════════════════════════════════════════

    def _render_invite(self, frame: np.ndarray, now: float) -> None:
        h1 = float(self._p1.metrics.get("happy", 0.0)) if self._p1 else 0.0
        h2 = float(self._p2.metrics.get("happy", 0.0)) if self._p2 else 0.0

        prog = 0.0
        if self._smile_at:
            prog = min((now - self._smile_at) / self.config.duo_smile_hold_seconds, 1.0)

        cw = min(500, frame.shape[1] - 48)
        ch = 200
        cx = frame.shape[1] // 2
        x1, y1 = cx - cw // 2, 22
        x2, y2 = x1 + cw, y1 + ch

        self._panel(frame, x1, y1, x2, y2, accent=GOLD)

        draw_text(frame, "İKİ KİŞİLİK YARIŞ", (x1 + 22, y1 + 14), 16, GOLD)
        draw_text(frame, "Hazır mısınız?", (x1 + 22, y1 + 40), 36, TEXT_WHITE)
        draw_text(
            frame,
            "Birlikte gülümseyin, yarışma başlasın!",
            (x1 + 22, y1 + 88),
            20,
            TEXT_LIGHT,
        )

        # Player smile bars
        bw = (cw - 70) // 2
        by = y2 - 56

        draw_text(frame, "Oyuncu 1", (x1 + 22, by - 20), 15, P1_ACCENT)
        self._bar(frame, x1 + 22, by, x1 + 22 + bw, 14, h1, P1_ACCENT)

        p2x = x2 - 22 - bw
        draw_text(frame, "Oyuncu 2", (p2x, by - 20), 15, P2_ACCENT)
        self._bar(frame, p2x, by, x2 - 22, 14, h2, P2_ACCENT)

        # Overall progress
        if prog > 0:
            self._bar(frame, x1 + 22, y2 - 22, x2 - 22, 10, prog, GOLD)

    # ═══════════════════════════════════════════════════════
    #  Intro: Yaş tahminleri + "Hazırlanın!"
    # ═══════════════════════════════════════════════════════

    def _render_intro(self, frame: np.ndarray, now: float) -> None:
        self._darken(frame, 0.84)
        fw, fh = frame.shape[1], frame.shape[0]
        cx, cy = fw // 2, fh // 2

        # Title
        self._center_text(frame, "İKİ KİŞİLİK YARIŞ", cy - 170, 44, GOLD)

        # VS glow + badge
        phase = (now * 2.0) % (2.0 * math.pi)
        pulse = 0.07 + 0.04 * abs(math.sin(phase))
        self._glow(frame, (cx, cy - 20), 90, GOLD, alpha=pulse)
        cv2.circle(frame, (cx, cy - 20), 48, GOLD, 2, cv2.LINE_AA)
        self._center_text(frame, "VS", cy - 48, 52, TEXT_WHITE)

        # Player cards
        cw = min(240, (fw - 100) // 2)
        card_h = 150

        # P1 card (left)
        p1x = cx - cw - 60
        p1y = cy - card_h // 2 - 10
        self._draw_intro_card(frame, p1x, p1y, cw, card_h, "OYUNCU 1", self._p1, P1_ACCENT)

        # P2 card (right)
        p2x = cx + 60
        p2y = cy - card_h // 2 - 10
        self._draw_intro_card(frame, p2x, p2y, cw, card_h, "OYUNCU 2", self._p2, P2_ACCENT)

        # Instruction
        elapsed = (now - self._intro_at) if self._intro_at else 0.0
        remaining = max(self.config.duo_intro_seconds - elapsed, 0.0)
        self._center_text(frame, "Yarışmaya hazırlanın!", cy + card_h // 2 + 40, 30, TEXT_LIGHT)
        self._center_text(frame, f"{remaining:.0f} saniye", cy + card_h // 2 + 78, 20, TEXT_DIM)

    def _draw_intro_card(self, frame, x, y, w, h, title, face, accent):
        self._panel(frame, x, y, x + w, y + h, accent=accent)
        draw_text(frame, title, (x + 18, y + 16), 17, accent)

        if face and face.age_years is not None:
            draw_text(frame, f"~{face.age_years:.0f}", (x + 18, y + 44), 54, TEXT_WHITE)
            label = face.age_label or ""
            draw_text(frame, f"Yaş aralığı {label}", (x + 18, y + 108), 18, TEXT_LIGHT)
        else:
            draw_text(frame, "Yaş tahmini", (x + 18, y + 56), 24, TEXT_DIM)
            draw_text(frame, "hesaplanıyor…", (x + 18, y + 88), 20, TEXT_DIM)

    # ═══════════════════════════════════════════════════════
    #  Countdown: 5-4-3-2-1
    # ═══════════════════════════════════════════════════════

    def _render_countdown(self, frame: np.ndarray, now: float) -> None:
        self._darken(frame, 0.86)
        elapsed = (now - self._countdown_at) if self._countdown_at else 0.0
        remaining = max(self.config.duo_countdown_seconds - elapsed, 0.0)
        number = max(1, int(math.ceil(remaining)))

        fw, fh = frame.shape[1], frame.shape[0]
        cx, cy = fw // 2, fh // 2

        phase = elapsed - math.floor(elapsed)
        flash = 0.70 + 0.30 * abs(math.sin(phase * math.pi))
        progress = 1.0 - remaining / max(self.config.duo_countdown_seconds, 0.01)
        ring_r = 140 + int(progress * 34)

        # Glow ring
        self._glow(frame, (cx, cy + 8), ring_r + 20, GOLD, alpha=0.10 * flash)
        cv2.circle(frame, (cx, cy + 8), ring_r, GOLD, 5, cv2.LINE_AA)

        # P1/P2 arcs
        angle = int(progress * 180)
        cv2.ellipse(frame, (cx, cy + 8), (ring_r - 14, ring_r - 14),
                    -90, 0, angle, P1_ACCENT, 4, cv2.LINE_AA)
        cv2.ellipse(frame, (cx, cy + 8), (ring_r - 14, ring_r - 14),
                    -90, 0, -angle, P2_ACCENT, 4, cv2.LINE_AA)

        self._center_text(frame, "İKİ KİŞİLİK YARIŞ", cy - 160, 32, GOLD)
        self._center_text(frame, "Hazır olun!", cy - 110, 26, TEXT_LIGHT)
        self._center_text(frame, str(number), cy + 44, 198, TEXT_WHITE)
        self._center_text(frame, "Başlıyor", cy + 140, 24, TEXT_DIM)

    # ═══════════════════════════════════════════════════════
    #  Active: 4 duygu turu + canlı skor
    # ═══════════════════════════════════════════════════════

    def _render_active(self, frame: np.ndarray, now: float) -> None:
        if not self._started_at:
            return
        elapsed = now - self._started_at
        label, key, color = DUO_TASKS[self._task_idx]
        task_elapsed = elapsed - self._task_idx * self.config.duo_task_seconds
        remaining = max(self.config.duo_task_seconds - task_elapsed, 0.0)

        p1_live = float(self._p1.metrics.get(key, 0.0)) if self._p1 else 0.0
        p2_live = float(self._p2.metrics.get(key, 0.0)) if self._p2 else 0.0
        p1_best = self._s1[self._task_idx]
        p2_best = self._s2[self._task_idx]

        fw, fh = frame.shape[1], frame.shape[0]

        # ── Top panel ──
        ph = 110
        self._panel(frame, 0, 0, fw, ph, accent=color, alpha=0.82)
        draw_text(frame, "İKİ KİŞİLİK YARIŞ", (22, 12), 14, color)
        draw_text(frame, _turkish_upper(label), (22, 36), 40, TEXT_WHITE)

        rt = f"Tur {self._task_idx + 1}/{len(DUO_TASKS)}"
        tt = f"{remaining:.1f} sn"
        rw, _ = measure_text(rt, 18)
        tw, _ = measure_text(tt, 26)
        draw_text(frame, rt, (fw - rw - 22, 16), 18, TEXT_DIM)
        draw_text(frame, tt, (fw - tw - 22, 42), 26, color)

        timer_prog = 1.0 - remaining / max(self.config.duo_task_seconds, 0.01)
        self._bar(frame, 22, ph - 16, fw - 22, 10, timer_prog, color)

        # ── Bottom score panel ──
        sph = 140
        sy = fh - sph
        ov = frame.copy()
        cv2.rectangle(ov, (0, sy), (fw, fh), (10, 12, 16), -1)
        cv2.addWeighted(ov, 0.84, frame, 0.16, 0.0, frame)
        cv2.line(frame, (0, sy), (fw, sy), color, 3)

        mid = fw // 2
        cv2.line(frame, (mid, sy + 10), (mid, fh - 10), DIVIDER, 2)

        # VS badge
        vs_y = sy + (sph // 2) - 12
        self._glow(frame, (mid, vs_y), 24, color, 0.15)
        cv2.circle(frame, (mid, vs_y), 20, color, 2, cv2.LINE_AA)
        vw, vh = measure_text("VS", 16)
        draw_text(frame, "VS", (mid - vw // 2, vs_y - vh // 2), 16, TEXT_WHITE)

        # P1 score (left)
        self._draw_live_score(frame, 28, sy + 14, mid - 40, "OYUNCU 1",
                              p1_live, p1_best, P1_ACCENT, color)
        # P2 score (right)
        self._draw_live_score(frame, mid + 16, sy + 14, mid - 40, "OYUNCU 2",
                              p2_live, p2_best, P2_ACCENT, color)

    def _draw_live_score(self, frame, x, y, w, title, live, best, accent, task_color):
        draw_text(frame, title, (x, y), 15, accent)
        draw_text(frame, f"%{live * 100:.0f}", (x, y + 24), 48, TEXT_WHITE)
        draw_text(frame, f"En iyi: %{best * 100:.0f}", (x, y + 78), 17, TEXT_DIM)
        self._bar(frame, x, y + 102, x + w, 16, live, accent)

    # ═══════════════════════════════════════════════════════
    #  Result: Kazanan / Berabere
    # ═══════════════════════════════════════════════════════

    def _render_result(self, frame: np.ndarray, now: float) -> None:
        self._darken(frame, 0.88)
        winner = self._winner()
        fw, fh = frame.shape[1], frame.shape[0]
        cx = fw // 2

        # Title
        self._center_text(frame, "İKİ KİŞİLİK YARIŞ", 28, 20, GOLD)
        self._center_text(frame, "─── SONUÇ ───", 56, 18, TEXT_DIM)

        # Winner announcement
        if winner == "p1":
            wtxt, wcol = "OYUNCU 1 KAZANDI!", P1_ACCENT
        elif winner == "p2":
            wtxt, wcol = "OYUNCU 2 KAZANDI!", P2_ACCENT
        else:
            wtxt, wcol = "BERABERE!", GOLD

        phase = (now * 2.5) % (2.0 * math.pi)
        pulse = 0.08 + 0.06 * abs(math.sin(phase))
        self._glow(frame, (cx, 118), 110, wcol, alpha=pulse)
        self._center_text(frame, wtxt, 96, 42, wcol)

        # Score cards
        cw = min(280, (fw - 60) // 2 - 10)
        card_y = 170
        card_h = fh - card_y - 30

        p1w = sum(1 for a, b in zip(self._s1, self._s2) if a > b)
        p2w = sum(1 for a, b in zip(self._s1, self._s2) if b > a)

        # P1 card
        self._draw_result_card(
            frame, cx - cw - 16, card_y, cw, card_h,
            "OYUNCU 1", self._s1, p1w, P1_ACCENT,
            self._p1, winner == "p1",
        )
        # P2 card
        self._draw_result_card(
            frame, cx + 16, card_y, cw, card_h,
            "OYUNCU 2", self._s2, p2w, P2_ACCENT,
            self._p2, winner == "p2",
        )

        # VS between cards
        vs_y = card_y + card_h // 2
        self._glow(frame, (cx, vs_y), 30, GOLD, 0.10)
        cv2.circle(frame, (cx, vs_y), 24, GOLD, 2, cv2.LINE_AA)
        vw, vh = measure_text("VS", 18)
        draw_text(frame, "VS", (cx - vw // 2, vs_y - vh // 2), 18, TEXT_WHITE)

    def _draw_result_card(self, frame, x, y, w, h, title, scores, wins, accent, face, is_winner):
        border_color = accent if is_winner else (44, 50, 58)
        ov = frame.copy()
        cv2.rectangle(ov, (x, y), (x + w, y + h), PANEL_BG, -1)
        cv2.addWeighted(ov, 0.90, frame, 0.10, 0.0, frame)
        cv2.rectangle(frame, (x, y), (x + w, y + h), border_color, 2 if is_winner else 1)

        # Accent strip
        cv2.rectangle(frame, (x, y), (x + w, y + 5), accent, -1)

        # Winner star
        if is_winner:
            draw_text(frame, "★", (x + w - 36, y + 12), 26, GOLD)

        # Title
        draw_text(frame, title, (x + 16, y + 16), 17, accent)

        # Age
        if face and face.age_years is not None:
            draw_text(frame, f"~{face.age_years:.0f} yaş", (x + 16, y + 40), 20, TEXT_LIGHT)
        row_y = y + 72

        # Per-round scores
        for i, (label, s1, s2) in enumerate(
            zip(RESULT_LABELS, scores, self._s2 if title == "OYUNCU 1" else self._s1)
        ):
            _, task_key, task_color = DUO_TASKS[i]
            score_val = scores[i]
            won_round = scores[i] > (self._s2[i] if "1" in title else self._s1[i])

            # Round label
            draw_text(frame, label, (x + 16, row_y), 17, task_color)

            # Score
            score_text = f"%{score_val * 100:.0f}"
            sw, _ = measure_text(score_text, 20)
            draw_text(frame, score_text, (x + w - sw - 42, row_y), 20, TEXT_WHITE)

            # Win indicator
            if won_round:
                draw_text(frame, "✓", (x + w - 28, row_y), 18, (80, 255, 120))

            # Score bar
            self._bar(frame, x + 16, row_y + 24, x + w - 16, 8, score_val, task_color)
            row_y += 52

        # Wins summary
        wins_y = row_y + 8
        draw_text(frame, f"Kazanılan tur: {wins}", (x + 16, wins_y), 20, accent)
