import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import face_landmarker as face_landmarker_lib
from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
    VisionTaskRunningMode,
)

try:
    from .config import AppConfig
    from .model_manager import ensure_face_landmarker_model
except ImportError:
    from config import AppConfig
    from model_manager import ensure_face_landmarker_model


@dataclass
class FaceAnalysis:
    bbox: tuple[int, int, int, int]
    metrics: Dict[str, float]
    points: np.ndarray
    face_id: int
    selection_score: float
    top_label: str = "Notr"


@dataclass
class TrackedFace:
    track_id: int
    anchor: np.ndarray
    bbox: tuple[int, int, int, int]
    smoother: "MetricSmoother"
    hits: int = 0
    misses: int = 0
    last_seen_ms: int = 0


class MetricSmoother:
    def __init__(self, alpha: float) -> None:
        self.alpha = alpha
        self.state: Dict[str, float] = {}

    def update(self, metrics: Dict[str, float]) -> Dict[str, float]:
        smoothed: Dict[str, float] = {}
        for key, value in metrics.items():
            previous = self.state.get(key, value)
            alpha = self._alpha_for_key(key)
            current = previous * (1.0 - alpha) + value * alpha
            self.state[key] = current
            smoothed[key] = current
        return smoothed

    def _alpha_for_key(self, key: str) -> float:
        if key == "angry":
            return max(0.18, self.alpha * 0.55)
        if key == "sad":
            return max(0.16, self.alpha * 0.50)
        return self.alpha


class FaceAnalyzer:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        model_path = ensure_face_landmarker_model(self.config.model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionTaskRunningMode.VIDEO,
            num_faces=self.config.max_faces,
            min_face_detection_confidence=self.config.min_detection_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence,
            output_face_blendshapes=True,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)
        self.last_timestamp_ms = 0
        self.tracks: Dict[int, TrackedFace] = {}
        self.next_track_id = 1
        self.match_distance_threshold = 0.55
        self.max_track_misses = 12

    def analyze(self, frame_rgb: np.ndarray) -> List[FaceAnalysis]:
        timestamp_ms = max(int(time.monotonic() * 1000), self.last_timestamp_ms + 1)
        self.last_timestamp_ms = timestamp_ms

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        if not result.face_landmarks:
            self._age_unmatched_tracks(set())
            return []

        detections: List[Dict[str, object]] = []
        frame_height, frame_width = frame_rgb.shape[:2]

        for index, landmarks in enumerate(result.face_landmarks):
            points = np.array(
                [(lm.x * frame_width, lm.y * frame_height) for lm in landmarks],
                dtype=np.float32,
            )
            bbox = self._bbox_from_points(points, frame_width, frame_height)
            blendshape_scores = self._blendshape_scores(
                result.face_blendshapes[index] if index < len(result.face_blendshapes) else []
            )
            detections.append(
                {
                    "bbox": bbox,
                    "points": points,
                    "center": self._bbox_center(bbox),
                    "blendshape_scores": blendshape_scores,
                    "face_width": max(float(bbox[2] - bbox[0]), 1.0),
                }
            )

        analyses: List[FaceAnalysis] = []
        matched_track_ids = set()

        for detection, track in self._match_tracks(detections, timestamp_ms):
            bbox = detection["bbox"]
            points = detection["points"]
            center = detection["center"]
            blendshape_scores = detection["blendshape_scores"]
            metrics = self._calculate_metrics(points, bbox, blendshape_scores)
            metrics["tracking_confidence"] = self._tracking_confidence(track, center, bbox)
            metrics["presence"] = self._clamp((track.hits + 1) / 8.0)

            smoothed_metrics = track.smoother.update(metrics)
            track.anchor = center
            track.bbox = bbox
            track.hits += 1
            track.misses = 0
            track.last_seen_ms = timestamp_ms
            matched_track_ids.add(track.track_id)

            # top_label calculation for UI (User requested 30% threshold)
            valid_expressions = {
                "Mutlu": smoothed_metrics.get("happy", 0.0),
                "Saskin": smoothed_metrics.get("surprised", 0.0),
                "Kizgin": smoothed_metrics.get("angry", 0.0),
                "Uzgun": smoothed_metrics.get("sad", 0.0)
            }
            top_name = "Notr"
            best_score = 0.30
            for name, score in valid_expressions.items():
                if score > best_score:
                    best_score = score
                    top_label = name
            
            # Use top_label effectively
            top_label = max(valid_expressions.items(), key=lambda x: x[1])
            final_label = top_label[0] if top_label[1] >= 0.30 else "Notr"

            analyses.append(
                FaceAnalysis(
                    bbox=bbox,
                    metrics=smoothed_metrics,
                    points=points,
                    face_id=track.track_id,
                    selection_score=self._selection_score(
                        bbox=bbox,
                        center=center,
                        frame_width=frame_width,
                        frame_height=frame_height,
                        metrics=smoothed_metrics,
                    ),
                    top_label=final_label
                )
            )

        self._age_unmatched_tracks(matched_track_ids)
        analyses.sort(key=lambda analysis: analysis.selection_score, reverse=True)
        return analyses

    @staticmethod
    def select_primary_face(analyses: List[FaceAnalysis]) -> Optional[FaceAnalysis]:
        if not analyses:
            return None
        return analyses[0]

    def _match_tracks(
        self,
        detections: List[Dict[str, object]],
        timestamp_ms: int,
    ) -> List[tuple[Dict[str, object], TrackedFace]]:
        assignments: List[tuple[Dict[str, object], TrackedFace]] = []
        available_track_ids = set(self.tracks.keys())

        for detection in sorted(
            detections,
            key=lambda item: self._bbox_area(item["bbox"]),
            reverse=True,
        ):
            center = detection["center"]
            face_width = detection["face_width"]
            matched_track_id = None
            matched_distance = float("inf")

            for track_id in available_track_ids:
                track = self.tracks[track_id]
                track_width = max(float(track.bbox[2] - track.bbox[0]), 1.0)
                normalized_distance = float(
                    np.linalg.norm(center - track.anchor) / max(face_width, track_width)
                )
                if normalized_distance < matched_distance:
                    matched_distance = normalized_distance
                    matched_track_id = track_id

            if (
                matched_track_id is not None
                and matched_distance <= self.match_distance_threshold
            ):
                track = self.tracks[matched_track_id]
                available_track_ids.remove(matched_track_id)
            else:
                track = self._create_track(
                    anchor=center,
                    bbox=detection["bbox"],
                    timestamp_ms=timestamp_ms,
                )

            assignments.append((detection, track))

        return assignments

    def _create_track(
        self,
        anchor: np.ndarray,
        bbox: tuple[int, int, int, int],
        timestamp_ms: int,
    ) -> TrackedFace:
        track = TrackedFace(
            track_id=self.next_track_id,
            anchor=np.array(anchor, dtype=np.float32),
            bbox=bbox,
            smoother=MetricSmoother(self.config.smoothing_alpha),
            last_seen_ms=timestamp_ms,
        )
        self.tracks[track.track_id] = track
        self.next_track_id += 1
        return track

    def _age_unmatched_tracks(self, matched_track_ids: set[int]) -> None:
        stale_track_ids = []
        for track_id, track in self.tracks.items():
            if track_id not in matched_track_ids:
                track.misses += 1
            if track.misses > self.max_track_misses:
                stale_track_ids.append(track_id)

        for track_id in stale_track_ids:
            self.tracks.pop(track_id, None)

    @staticmethod
    def _bbox_from_points(
        points: np.ndarray, frame_width: int, frame_height: int
    ) -> tuple[int, int, int, int]:
        x_min = int(np.clip(points[:, 0].min(), 0, frame_width - 1))
        y_min = int(np.clip(points[:, 1].min(), 0, frame_height - 1))
        x_max = int(np.clip(points[:, 0].max(), 0, frame_width - 1))
        y_max = int(np.clip(points[:, 1].max(), 0, frame_height - 1))
        return x_min, y_min, x_max, y_max

    @staticmethod
    def _bbox_center(bbox: tuple[int, int, int, int]) -> np.ndarray:
        x_min, y_min, x_max, y_max = bbox
        return np.array(
            [(x_min + x_max) / 2.0, (y_min + y_max) / 2.0],
            dtype=np.float32,
        )

    @staticmethod
    def _bbox_area(bbox: tuple[int, int, int, int]) -> int:
        x_min, y_min, x_max, y_max = bbox
        return max(x_max - x_min, 1) * max(y_max - y_min, 1)

    def _tracking_confidence(
        self,
        track: TrackedFace,
        center: np.ndarray,
        bbox: tuple[int, int, int, int],
    ) -> float:
        if track.hits == 0:
            return 0.42

        current_area = float(self._bbox_area(bbox))
        previous_area = float(self._bbox_area(track.bbox))
        scale = max(float(bbox[2] - bbox[0]), float(track.bbox[2] - track.bbox[0]), 1.0)

        motion_delta = float(np.linalg.norm(center - track.anchor) / scale)
        motion_score = 1.0 - self._clamp(motion_delta / 0.65)

        area_delta = abs(current_area - previous_area) / max(current_area, previous_area, 1.0)
        size_score = 1.0 - self._clamp(area_delta)

        continuity_score = self._clamp(track.hits / 6.0)
        return self._clamp(
            0.2 + (continuity_score * 0.45) + (motion_score * 0.25) + (size_score * 0.1)
        )

    def _selection_score(
        self,
        bbox: tuple[int, int, int, int],
        center: np.ndarray,
        frame_width: int,
        frame_height: int,
        metrics: Dict[str, float],
    ) -> float:
        frame_center = np.array([frame_width / 2.0, frame_height / 2.0], dtype=np.float32)
        frame_diagonal = max(float(np.linalg.norm(frame_center)), 1.0)

        area_ratio = self._bbox_area(bbox) / float(frame_width * frame_height)
        area_score = self._clamp(area_ratio * 5.0)
        center_distance = float(np.linalg.norm(center - frame_center) / frame_diagonal)
        center_score = 1.0 - self._clamp(center_distance * 1.25)

        return self._clamp(
            (area_score * 0.5)
            + (center_score * 0.25)
            + (metrics.get("tracking_confidence", 0.0) * 0.2)
            + (metrics.get("presence", 0.0) * 0.05)
        )

    def _calculate_metrics(
        self,
        points: np.ndarray,
        bbox: tuple[int, int, int, int],
        blendshape_scores: Dict[str, float],
    ) -> Dict[str, float]:
        face_width = max(float(bbox[2] - bbox[0]), 1.0)

        def point(index: int) -> np.ndarray:
            return points[index]

        def point_distance(a: int, b: int) -> float:
            return float(np.linalg.norm(point(a) - point(b)))

        def average_point(*indices: int) -> np.ndarray:
            return np.mean([point(index) for index in indices], axis=0)

        def normalized_distance(a: int, b: int) -> float:
            return float(np.linalg.norm(point(a) - point(b)) / face_width)

        mouth_width = normalized_distance(61, 291)
        mouth_open_geo = normalized_distance(13, 14)
        mouth_center_y = float((point(13)[1] + point(14)[1]) / 2.0)
        mouth_corner_y = float((point(61)[1] + point(291)[1]) / 2.0)

        mouth_smile = self._average_score(
            blendshape_scores,
            "mouth_smile_left",
            "mouth_smile_right",
        )
        mouth_frown = self._average_score(
            blendshape_scores,
            "mouth_frown_left",
            "mouth_frown_right",
        )
        mouth_dimple = self._average_score(
            blendshape_scores,
            "mouth_dimple_left",
            "mouth_dimple_right",
        )
        mouth_stretch = self._average_score(
            blendshape_scores,
            "mouth_stretch_left",
            "mouth_stretch_right",
        )
        mouth_press = self._average_score(
            blendshape_scores,
            "mouth_press_left",
            "mouth_press_right",
        )
        mouth_lower_down = self._average_score(
            blendshape_scores,
            "mouth_lower_down_left",
            "mouth_lower_down_right",
        )
        eye_blink = self._average_score(
            blendshape_scores,
            "eye_blink_left",
            "eye_blink_right",
        )
        eye_wide = self._average_score(
            blendshape_scores,
            "eye_wide_left",
            "eye_wide_right",
        )
        eye_squint = self._average_score(
            blendshape_scores,
            "eye_squint_left",
            "eye_squint_right",
        )
        cheek_squint = self._average_score(
            blendshape_scores,
            "cheek_squint_left",
            "cheek_squint_right",
        )
        brow_down = self._average_score(
            blendshape_scores,
            "brow_down_left",
            "brow_down_right",
        )
        brow_outer_up = self._average_score(
            blendshape_scores,
            "brow_outer_up_left",
            "brow_outer_up_right",
        )
        brow_inner_up = self._score(blendshape_scores, "brow_inner_up")
        mouth_funnel = self._score(blendshape_scores, "mouth_funnel")
        mouth_pucker = self._score(blendshape_scores, "mouth_pucker")
        mouth_roll_lower = self._score(blendshape_scores, "mouth_roll_lower")
        mouth_roll_upper = self._score(blendshape_scores, "mouth_roll_upper")
        mouth_shrug_lower = self._score(blendshape_scores, "mouth_shrug_lower")
        mouth_shrug_upper = self._score(blendshape_scores, "mouth_shrug_upper")
        jaw_forward = self._score(blendshape_scores, "jaw_forward")
        nose_sneer = self._average_score(
            blendshape_scores,
            "nose_sneer_left",
            "nose_sneer_right",
        )

        mouth_open = self._clamp(
            self._score(blendshape_scores, "jaw_open") * 1.25 + (mouth_open_geo * 4.0)
        )
        mouth_closed = self._clamp(1.0 - mouth_open * 1.15)
        left_eye_ratio = (
            point_distance(159, 145)
            + point_distance(160, 144)
            + point_distance(158, 153)
        ) / (3.0 * max(point_distance(33, 133), 1e-6))
        right_eye_ratio = (
            point_distance(386, 374)
            + point_distance(385, 380)
            + point_distance(387, 373)
        ) / (3.0 * max(point_distance(362, 263), 1e-6))
        eye_open_geo = self._clamp((((left_eye_ratio + right_eye_ratio) / 2.0) - 0.11) / 0.12)
        eye_open_geo = float(eye_open_geo**1.35)
        eye_open = self._clamp(
            eye_open_geo * 0.82
            + eye_wide * 0.22
            - eye_blink * 0.26
            - eye_squint * 0.12
        )
        surprise_eye = self._clamp(
            eye_open_geo * 0.54 + eye_wide * 0.34 + eye_open * 0.18 - eye_squint * 0.14
        )
        left_brow_center = average_point(70, 63, 105, 66, 107)
        right_brow_center = average_point(336, 296, 334, 293, 300)
        left_eye_center = average_point(33, 133)
        right_eye_center = average_point(362, 263)
        brow_eye_gap = (
            max(float(left_eye_center[1] - left_brow_center[1]), 0.0)
            + max(float(right_eye_center[1] - right_brow_center[1]), 0.0)
        ) / (2.0 * face_width)
        brow_lowered_geo = self._clamp((0.114 - brow_eye_gap) / 0.042)
        inner_brow_gap = min(
            normalized_distance(107, 336),
            normalized_distance(66, 296),
            normalized_distance(105, 334),
        )
        brow_furrow = self._clamp((0.255 - inner_brow_gap) / 0.042)

        smile_curve = self._clamp((mouth_center_y - mouth_corner_y) / face_width * 14.0)
        # Moderated negative offset to balance neutral vs sad
        frown_curve = self._clamp(
            ((mouth_corner_y - mouth_center_y) / face_width * 18.0) - 0.07
        )
        smile_width = self._clamp((mouth_width - 0.26) * 4.0)
        mouth_narrow = self._clamp((0.34 - mouth_width) / 0.1)
        upper_lip_depth = normalized_distance(0, 13)
        lower_lip_depth = normalized_distance(14, 17)
        lower_lip_bias = self._clamp(
            (
                (
                    (lower_lip_depth - upper_lip_depth)
                    / max(lower_lip_depth + upper_lip_depth, 1e-6)
                )
                - 0.03
            )
            / 0.22
        )
        mouth_roundness = self._clamp(
            ((mouth_open_geo / max(mouth_width, 1e-6)) - 0.24) * 4.6
            + mouth_funnel * 0.32
            - smile_width * 0.08
        )
        mouth_ellipse = self._clamp(
            smile_width * 0.46
            + smile_curve * 0.3
            + mouth_stretch * 0.16
            - mouth_roundness * 0.3
            - mouth_funnel * 0.08
        )
        smile_proxy = self._clamp(
            mouth_smile * 0.74
            + mouth_dimple * 0.16
            + cheek_squint * 0.22
            + mouth_stretch * 0.1
            + smile_curve * 0.28
            + smile_width * 0.12
            - mouth_frown * 0.24
            - mouth_press * 0.05
            - mouth_open * 0.04
        )

        happy_cheek = self._clamp(max(cheek_squint, mouth_dimple) * 0.8 + mouth_dimple * 0.2)
        friendly_signal = self._clamp(
            smile_proxy * 0.84
            + happy_cheek * 0.58
            + mouth_ellipse * 0.38
            + mouth_smile * 0.28
        )
        happy_eye = self._clamp(
            eye_squint * 0.34
            + cheek_squint * 0.18
            - eye_wide * 0.12
            - eye_blink * 0.06
        )
        angry_brow_penalty = self._clamp(
            brow_furrow * 1.06
            + brow_lowered_geo * 0.52
            + brow_down * 0.28
        )
        happy_combo = float(
            np.sqrt(max(smile_proxy, 0.0) * max(happy_cheek + mouth_ellipse * 0.6, 0.0))
        )
        happy_base = (
            smile_proxy * 1.0
            + mouth_smile * 0.32
            + happy_cheek * 0.3
            + mouth_ellipse * 0.34
            + smile_curve * 0.16
            + happy_eye * 0.06
            - mouth_frown * 0.1
            - eye_wide * 0.08
            - mouth_roundness * 0.16
            - angry_brow_penalty * 0.24
        )
        happy_boost = (
            happy_combo * 0.4
            + max(mouth_smile, smile_curve) * 0.26
            + mouth_ellipse * 0.18
        )
        happy_score = self._clamp(happy_base * 1.28 + happy_boost - angry_brow_penalty * 0.12)
        happy_floor = self._clamp(
            friendly_signal * 0.94
            + mouth_smile * 0.24
            + happy_cheek * 0.18
            - brow_furrow * 0.06
        )
        happy_score = max(happy_score, happy_floor)

        surprise_mouth = self._clamp(
            mouth_open * 0.42
            + mouth_roundness * 0.34
            + mouth_funnel * 0.16
            - smile_curve * 0.16
            - mouth_ellipse * 0.1
        )
        surprise_combo = float(np.sqrt(max(surprise_mouth, 0.0) * max(surprise_eye, 0.0)))
        surprise_boost = max(surprise_combo - 0.4, 0.0) * 0.5
        surprised_score = self._clamp(
            surprise_combo * 0.66
            + surprise_mouth * 0.12
            + surprise_eye * 0.12
            + brow_inner_up * 0.14
            + brow_outer_up * 0.08
            + surprise_boost
            - smile_proxy * 0.2
            - cheek_squint * 0.12
            - eye_squint * 0.12
        )
        surprise_guard = self._clamp(
            surprise_combo * 0.74
            + surprise_mouth * 0.24
            + surprise_eye * 0.18
            + mouth_roundness * 0.12
            - mouth_press * 0.1
            - friendly_signal * 0.08
        )
        angry_brow_core = float(np.sqrt(max(brow_furrow, 0.0) * max(brow_lowered_geo, 0.0)))
        # Offset 0.72 is the absolute hard floor for user's unique neutral eyebrows
        angry_brow_gate = self._clamp((angry_brow_core - 0.72) / 0.10)
        angry_brow_gate = float(angry_brow_gate**0.50)  # Maximum aggressive ramp once triggered

        # ── sad expression signals ──────────────────────────────
        sad_brow = self._clamp(
            brow_inner_up * 0.85
            + brow_outer_up * 0.12
            - brow_down * 0.42
            - brow_furrow * 0.55
            - angry_brow_gate * 0.35
        )
        upper_lip_tuck = self._clamp(
            mouth_roll_upper * 0.72
            + mouth_shrug_upper * 0.12
            + mouth_press * 0.08
            + mouth_pucker * 0.18
            - mouth_open * 0.14
            - mouth_smile * 0.08
        )
        lower_lip_curl = self._clamp(
            lower_lip_bias * 0.62
            + mouth_shrug_lower * 0.26
            + mouth_lower_down * 0.18
            + mouth_frown * 0.22
            + mouth_pucker * 0.3
            - mouth_roll_lower * 0.08
            - mouth_open * 0.12
            - mouth_smile * 0.06
        )
        sad_pout_shape = self._clamp(
            lower_lip_bias * 0.72
            + mouth_pucker * 0.62
            + upper_lip_tuck * 0.24
            + mouth_press * 0.12
            + mouth_closed * 0.18
            + mouth_narrow * 0.32
            - mouth_open * 0.18
            - mouth_smile * 0.34
            - smile_proxy * 0.32
        )
        sad_pout_combo = self._clamp(
            float(
                np.sqrt(
                    max(sad_pout_shape, 0.0)
                    * max(lower_lip_curl + upper_lip_tuck * 0.6, 0.0)
                )
            )
            * 1.0
            + sad_pout_shape * 0.26
        )
        sad_frown_signal = self._clamp(
            frown_curve * 0.52
            + mouth_frown * 0.38
            + lower_lip_curl * 0.14
            + mouth_lower_down * 0.08
            - smile_curve * 0.28
            - mouth_smile * 0.22
            - smile_proxy * 0.14
            - 0.15
        )
        sad_frown_brow_combo = float(
            np.sqrt(max(sad_frown_signal, 0.0) * max(sad_brow, 0.0))
        )
        sad_lip_signal = self._clamp(
            float(
                np.sqrt(
                    max(lower_lip_curl, 0.0)
                    * max(upper_lip_tuck + frown_curve * 0.72, 0.0)
                )
            )
            * 0.52
            + lower_lip_curl * 0.14
            + upper_lip_tuck * 0.08
            + frown_curve * 0.34
            + mouth_frown * 0.26
            + mouth_pucker * 0.14
            + sad_pout_combo * 0.3
            + sad_frown_signal * 0.18
            - mouth_smile * 0.22
            - smile_curve * 0.2
            - smile_proxy * 0.14
        )
        sad_pout_gate = self._clamp(
            mouth_pucker * 0.24
            + sad_pout_combo * 0.72
            + lower_lip_curl * 0.22
            + upper_lip_tuck * 0.12
            + mouth_press * 0.06
            - mouth_smile * 0.24
        )
        # ── angry expression signals ───────────────────────────
        angry_brow = self._clamp(
            brow_down * 0.30
            + brow_lowered_geo * 0.65
            + brow_furrow * 1.55
            - brow_inner_up * 0.12
        )
        angry_eye = self._clamp(
            eye_squint * 0.1
            + (1.0 - eye_open) * 0.03
            - eye_wide * 0.1
            - eye_blink * 0.03
            - friendly_signal * 0.12
        )
        angry_mouth = self._clamp(
            mouth_press * 0.42
            + mouth_closed * 0.18
            + jaw_forward * 0.1
            - mouth_roundness * 0.12
            - mouth_smile * 0.28
            - happy_cheek * 0.12
            - mouth_ellipse * 0.14
        )
        angry_combo = float(
            np.sqrt(max(angry_brow, 0.0) * max(angry_mouth + angry_eye * 0.15, 0.0))
        )
        angry_peak = self._clamp(
            angry_brow_gate * 1.04
            + angry_brow * 0.38
            + brow_furrow * 0.32
            + brow_lowered_geo * 0.22
            + angry_eye * 0.02
            + angry_mouth * 0.04
            - friendly_signal * 0.72
            - 0.46
        )
        # RAW ANGRY Core: Aggressive suppression from Happy/Surprise/Sad
        raw_angry_score = self._clamp(
            angry_brow * 1.25
            + angry_brow_gate * 2.50
            + brow_furrow * 0.45
            + brow_lowered_geo * 0.35
            + mouth_press * 1.50
            + eye_squint * 0.82
            - smile_proxy * 2.50      # ULTIMATE suppression from smile
            - friendly_signal * 1.50  # ULTIMATE suppression from friendly
            - surprise_guard * 2.00   # ULTIMATE suppression from surprise
            - sad_brow * 2.50         # NEW: Ultimate suppression from sad brow
        )

        friendly_guard = self._clamp(
            friendly_signal * 0.88
            + mouth_smile * 0.28
            + happy_cheek * 0.24
            + mouth_ellipse * 0.16
            - mouth_press * 0.06
        )
        angry_posture = self._clamp(
            mouth_closed * 0.18
            + eye_squint * 0.02
            + (1.0 - eye_open) * 0.02
            + mouth_press * 0.14
            + angry_brow_gate * 0.48
            + brow_furrow * 0.38
            + brow_lowered_geo * 0.24
            - surprise_mouth * 0.42
            - surprise_eye * 0.18
            - mouth_roundness * 0.12
        )
        angry_posture = float(angry_posture**1.05)
        angry_gate = self._clamp(1.0 - friendly_guard * 0.88)
        angry_gate = float(angry_gate**1.3)
        surprise_block = self._clamp(1.0 - surprise_guard * 0.85)
        surprise_block = float(surprise_block**1.3)
        angry_brow_boost = self._clamp(
            angry_brow_gate * 1.58
            + angry_brow * 0.48
            + brow_furrow * 0.36
            + brow_lowered_geo * 0.28
            - friendly_guard * 0.32
            - surprise_guard * 0.18
            - 0.10
        )
        angry_support = angry_gate * surprise_block
        raw_final_angry = (
            raw_angry_score * angry_support * (0.15 + angry_brow_gate * 1.20 + angry_posture * 0.35)
            + angry_brow_boost * angry_support * (0.15 + angry_brow_gate * 0.62)
            + angry_brow_gate * angry_support * 0.12
        )
        # HARD GATING: If eyebrows aren't moving, score is forced to ZERO. 
        # This kills 100% of the leakage from mouth/jaw movements during other expressions.
        angry_score = self._clamp(max(raw_final_angry - 0.05, 0.0) * 80.0 * angry_brow_gate)

        
        # CUSTOM CALIBRATION (Absolute Final Overhaul: Neutral Floor 0.63-0.65)
        # Anything below 0.65 is treated as pure silence (0.0 score)
        w_frown = float(max(frown_curve - 0.65, 0.0))
        w_brow = float(max(sad_brow - 0.15, 0.0))
        w_mouth_frown = float(max(mouth_frown - 0.65, 0.0))
        w_pout = float(max(sad_pout_combo - 0.85, 0.0))

        # Intent requires passing the massive baseline or moving the eyebrows
        sad_intent = w_frown + w_mouth_frown + w_pout
        
        sad_score = 0.0
        sad_floor_pout = 0.0
        sad_floor_frown = 0.0

        if sad_intent > 0.005:
            sad_protection = self._clamp(w_pout * 1.0 + w_frown * 1.0)
            # Use exponents on suppressors so that tiny background noise (10-15%) 
            # doesn't obliterate the sad score, but actual strong emotions (80%) do.
            sad_suppressor = self._clamp(
                (happy_score ** 1.5) * 3.0 + 
                (surprised_score ** 1.5) * 2.0 + 
                (max(angry_score - 0.10, 0.0) ** 1.0) * 4.0 +  # Strong angry->sad block
                (angry_brow_gate ** 1.2) * 2.5 +  # Direct brow gate suppression on sad
                (smile_proxy ** 1.5) * 2.0 - 
                sad_protection * 0.4
            )
            
            # DELTA BOOST: Massive gain for the subtle sad signal
            raw_sad = (w_frown * 15.0 + w_mouth_frown * 15.0 + w_pout * 15.0)
            
            # Massive multipliers: Sad signal (15%) -> boosted to 75%+
            sad_base = self._clamp(raw_sad * 250.0)
            sad_floor_pout = self._clamp(w_pout * 400.0)
            sad_floor_frown = self._clamp(w_frown * 400.0)
            
            # Pick the strongest sad signal
            sad_score_unsuppressed = max(sad_base, sad_floor_pout, sad_floor_frown)
            
            # --- HEAD TILT PENALTY (Kafa Eğme / Sahte Ağız Cezası) ---
            # Tilting head down creates massive fake pout/frown signals.
            # If eyebrows aren't assisting, it deducts up to 25% from the raw score.
            brow_support = self._clamp(w_pout * 1.5 + w_frown * 1.5)
            tilt_penalty = max(0.0, 0.25 - brow_support)
            
            # Apply penalty and compensate legitimate high scores with a small multiplier (1.3x)
            sad_score_unsuppressed = self._clamp((sad_score_unsuppressed - tilt_penalty) * 1.3)
            
            # Apply the suppressor as an absolute final multiplier
            # If sad_suppressor is 1.0 (e.g. 100% angry), sad_score becomes 0
            # If sad_suppressor is 0.1 (e.g. 10% happy noise), sad_score keeps 90%
            sad_score = self._clamp(sad_score_unsuppressed * (1.0 - sad_suppressor))

        return {
            "tracking_confidence": 0.0,
            "eye_open": eye_open,
            "happy": happy_score,
            "surprised": surprised_score,
            "angry": angry_score,
            "sad": sad_score,
            "presence": 0.0,
        }

    @staticmethod
    def _blendshape_scores(categories) -> Dict[str, float]:
        scores: Dict[str, float] = {}
        for category in categories:
            key = category.category_name
            if not key and category.index is not None:
                key = face_landmarker_lib.Blendshapes(category.index).name.lower()
            if key:
                scores[key.lower()] = float(category.score or 0.0)
        return scores

    @staticmethod
    def _score(scores: Dict[str, float], key: str) -> float:
        return float(scores.get(key, 0.0))

    def _average_score(self, scores: Dict[str, float], *keys: str) -> float:
        if not keys:
            return 0.0
        return float(sum(self._score(scores, key) for key in keys) / len(keys))

    @staticmethod
    def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
        return float(max(minimum, min(maximum, value)))

    def close(self) -> None:
        self.landmarker.close()
