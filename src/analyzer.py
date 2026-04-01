from concurrent.futures import Future, ThreadPoolExecutor
import time
from dataclasses import dataclass
from typing import cast, Dict, List, Optional, Tuple, Any

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import face_landmarker as face_landmarker_lib
from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
    VisionTaskRunningMode,
)

try:
    from .age_estimator import age_label_for_years, create_age_estimator, normalize_age_years
    from .config import AppConfig
    from .emotion_estimator import create_emotion_estimator
    from .model_manager import ensure_face_landmarker_model
except ImportError:
    from age_estimator import age_label_for_years, create_age_estimator, normalize_age_years
    from config import AppConfig
    from emotion_estimator import create_emotion_estimator
    from model_manager import ensure_face_landmarker_model


@dataclass
class FaceAnalysis:
    bbox: tuple[int, int, int, int]
    metrics: Dict[str, float]
    points: np.ndarray
    face_id: int
    selection_score: float
    top_label: str = "Sakin"
    age_years: Optional[float] = None
    age_label: Optional[str] = None


@dataclass
class TrackedFace:
    track_id: int
    anchor: np.ndarray
    bbox: tuple[int, int, int, int]
    smoother: "MetricSmoother"
    emotion_scores: Optional[Dict[str, float]] = None
    emotion_confidence: float = 0.0
    emotion_margin: float = 0.0
    top_label_state: str = "Sakin"
    label_candidate: str = "Sakin"
    label_candidate_frames: int = 0
    last_emotion_inference_ms: int = 0
    pending_emotion_future: Optional[Future] = None
    age_smoother: Optional["ScalarSmoother"] = None
    age_years: Optional[float] = None
    age_label: Optional[str] = None
    age_history: Optional[List[float]] = None
    age_stable_frames: int = 0
    last_age_inference_ms: int = 0
    pending_age_future: Optional[Future] = None
    hits: int = 0
    misses: int = 0
    last_seen_ms: int = 0


class MetricSmoother:
    def __init__(self, alpha: float, emotion_backend: str = "rules") -> None:
        self.alpha = alpha
        self.emotion_backend = emotion_backend
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
        if self.emotion_backend != "rules":
            return self.alpha
        if key == "angry":
            return max(0.18, self.alpha * 0.55)
        if key == "sad":
            return max(0.16, self.alpha * 0.50)
        return self.alpha


class ScalarSmoother:
    def __init__(self, alpha: float) -> None:
        self.alpha = alpha
        self.state: Optional[float] = None

    def update(self, value: float) -> float:
        if self.state is None:
            self.state = value
        else:
            state = cast(float, self.state)
            self.state = (state * (1.0 - self.alpha)) + (value * self.alpha)
        return cast(float, self.state)


class FaceAnalyzer:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.age_estimator = create_age_estimator(self.config)
        self.emotion_estimator = create_emotion_estimator(self.config)
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
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="face-models")
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
        matched_track_ids: set[int] = set()

        for detection, track in self._match_tracks(detections, timestamp_ms):
            bbox = cast(Tuple[int, int, int, int], detection["bbox"])
            points = cast(np.ndarray, detection["points"])
            center = cast(np.ndarray, detection["center"])
            blendshape_scores = cast(Dict[str, float], detection["blendshape_scores"])

            metrics = self._calculate_metrics(points, bbox, blendshape_scores)
            metrics.update(
                self._update_emotion_estimate(
                    track=track,
                    frame_rgb=frame_rgb,
                    bbox=bbox,
                    points=points,
                    center=center,
                    base_metrics=metrics,
                    timestamp_ms=timestamp_ms,
                )
            )
            tracking_confidence = self._tracking_confidence(track, center, bbox)
            metrics["tracking_confidence"] = tracking_confidence
            metrics["presence"] = self._clamp((track.hits + 1) / 8.0)
            age_motion = self._age_motion_delta(track, center, bbox)
            age_yaw_ratio, age_roll_degrees = self._age_pose_metrics(points)
            age_frame_is_stable = self._is_age_frame_stable(
                track=track,
                metrics=metrics,
                motion_delta=age_motion,
                yaw_ratio=age_yaw_ratio,
                roll_degrees=age_roll_degrees,
            )

            smoothed_metrics = track.smoother.update(metrics)
            track.anchor = center
            track.bbox = bbox
            track.hits += 1
            track.misses = 0
            track.last_seen_ms = timestamp_ms
            track.age_stable_frames = track.age_stable_frames + 1 if age_frame_is_stable else 0
            matched_track_ids.add(track.track_id)

            age_years, age_label = self._update_age_estimate(
                track=track,
                frame_rgb=frame_rgb,
                bbox=bbox,
                points=points,
                frame_is_stable=age_frame_is_stable,
                timestamp_ms=timestamp_ms,
            )

            final_label = self._resolve_top_label(track, smoothed_metrics)

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
                    top_label=final_label,
                    age_years=age_years,
                    age_label=age_label,
                )
            )

        self._age_unmatched_tracks(matched_track_ids)
        analyses.sort(key=lambda analysis: analysis.selection_score, reverse=True)
        return analyses

    @staticmethod
    def _default_emotion_scores() -> Dict[str, float]:
        return {
            "happy": 0.0,
            "surprised": 0.0,
            "angry": 0.0,
            "sad": 0.0,
            "neutral": 1.0,
        }

    def _rule_emotion_scores(self, metrics: Dict[str, float]) -> Dict[str, float]:
        happy = self._clamp(metrics.get("happy", 0.0))
        surprised = self._clamp(metrics.get("surprised", 0.0))
        angry = self._clamp(metrics.get("angry", 0.0))
        sad = self._clamp(metrics.get("sad", 0.0))
        neutral = self._clamp(1.0 - max(happy, surprised, angry, sad))
        return {
            "happy": happy,
            "surprised": surprised,
            "angry": angry,
            "sad": sad,
            "neutral": neutral,
        }

    @staticmethod
    def select_primary_face(analyses: List[FaceAnalysis]) -> Optional[FaceAnalysis]:
        if not analyses:
            return None
        return analyses[0]

    def _update_age_estimate(
        self,
        track: TrackedFace,
        frame_rgb: np.ndarray,
        bbox: tuple[int, int, int, int],
        points: np.ndarray,
        frame_is_stable: bool,
        timestamp_ms: int,
    ) -> tuple[Optional[float], Optional[str]]:
        if self.age_estimator is None or track.age_smoother is None:
            return None, None

        self._consume_age_future(track)

        face_width = bbox[2] - bbox[0]
        if face_width < self.config.age_min_face_width:
            return track.age_years, track.age_label

        if (
            not frame_is_stable
            or track.age_stable_frames < self.config.age_stability_required_frames
        ):
            return track.age_years, track.age_label

        should_refresh = (
            track.age_years is None
            or timestamp_ms - track.last_age_inference_ms >= self.config.age_inference_interval_ms
        )
        if not should_refresh or track.pending_age_future is not None:
            return track.age_years, track.age_label

        aligned_crop = self._extract_aligned_age_crop(frame_rgb, points)
        face_crop = self._extract_age_crop(
            frame_rgb=frame_rgb,
            bbox=bbox,
            padding_ratio=self.config.age_crop_padding_ratio,
            top_bias_ratio=self.config.age_crop_top_bias_ratio,
        )
        body_crop = self._extract_body_crop(frame_rgb, bbox)
        if face_crop is None and aligned_crop is None:
            return track.age_years, track.age_label

        track.pending_age_future = self.executor.submit(
            self._run_age_inference,
            face_crop,
            body_crop,
            aligned_crop,
            timestamp_ms,
        )
        return track.age_years, track.age_label

    def _update_emotion_estimate(
        self,
        track: TrackedFace,
        frame_rgb: np.ndarray,
        bbox: tuple[int, int, int, int],
        points: np.ndarray,
        center: np.ndarray,
        base_metrics: Dict[str, float],
        timestamp_ms: int,
    ) -> Dict[str, float]:
        rule_scores = self._rule_emotion_scores(base_metrics)
        if self.emotion_estimator is None:
            return {"neutral": rule_scores["neutral"]}

        self._consume_emotion_future(track)
        cached_scores = dict(track.emotion_scores or rule_scores)
        face_width = bbox[2] - bbox[0]
        if face_width < self.config.emotion_min_face_width:
            return self._blend_emotion_sources(
                rule_scores=rule_scores,
                model_scores=cached_scores,
                confidence=track.emotion_confidence,
                margin=track.emotion_margin,
                frame_quality=0.0,
                mouth_open=base_metrics.get("mouth_open", 0.0),
                eye_open=base_metrics.get("eye_open", 0.0),
            )

        motion_delta = self._age_motion_delta(track, center, bbox)
        yaw_ratio, roll_degrees = self._age_pose_metrics(points)
        frame_quality = self._emotion_frame_quality(
            motion_delta=motion_delta,
            yaw_ratio=yaw_ratio,
            roll_degrees=roll_degrees,
        )

        should_refresh = (
            track.last_emotion_inference_ms == 0
            or timestamp_ms - track.last_emotion_inference_ms
            >= self.config.emotion_inference_interval_ms
        )
        if should_refresh and track.pending_emotion_future is None:
            aligned_crop = self._extract_aligned_age_crop(frame_rgb, points)
            face_crop = self._extract_age_crop(
                frame_rgb=frame_rgb,
                bbox=bbox,
                padding_ratio=self.config.emotion_crop_padding_ratio,
                top_bias_ratio=self.config.emotion_crop_top_bias_ratio,
            )
            context_crop = self._extract_age_crop(
                frame_rgb=frame_rgb,
                bbox=bbox,
                padding_ratio=self.config.emotion_context_padding_ratio,
                top_bias_ratio=self.config.emotion_context_top_bias_ratio,
            )
            track.pending_emotion_future = self.executor.submit(
                self._run_emotion_inference,
                face_crop,
                aligned_crop,
                context_crop,
                rule_scores,
                base_metrics.get("mouth_open", 0.0),
                base_metrics.get("eye_open", 0.0),
                frame_quality,
                timestamp_ms,
            )

        return self._blend_emotion_sources(
            rule_scores=rule_scores,
            model_scores=cached_scores,
            confidence=track.emotion_confidence,
            margin=track.emotion_margin,
            frame_quality=frame_quality,
            mouth_open=base_metrics.get("mouth_open", 0.0),
            eye_open=base_metrics.get("eye_open", 0.0),
        )

    def _consume_age_future(self, track: TrackedFace) -> None:
        future = track.pending_age_future
        if future is None or not future.done():
            return
        track.pending_age_future = None
        try:
            result = future.result()
        except Exception:
            return
        if result is None or track.age_smoother is None:
            return

        calibrated_age, timestamp_ms = cast(Tuple[float, int], result)
        robust_age = self._update_age_history(track, calibrated_age)
        age_smoother = cast(ScalarSmoother, track.age_smoother)
        smoothed_age = age_smoother.update(robust_age)
        track.age_years = smoothed_age
        track.age_label = age_label_for_years(smoothed_age)
        track.last_age_inference_ms = timestamp_ms

    def _consume_emotion_future(self, track: TrackedFace) -> None:
        future = track.pending_emotion_future
        if future is None or not future.done():
            return
        track.pending_emotion_future = None
        try:
            result = future.result()
        except Exception:
            return
        if result is None:
            return

        scores, confidence, margin, timestamp_ms = cast(
            Tuple[Dict[str, float], float, float, int],
            result,
        )
        track.emotion_scores = scores
        track.emotion_confidence = confidence
        track.emotion_margin = margin
        track.last_emotion_inference_ms = timestamp_ms

    def _run_age_inference(
        self,
        face_crop: Optional[np.ndarray],
        body_crop: Optional[np.ndarray],
        aligned_crop: Optional[np.ndarray],
        timestamp_ms: int,
    ) -> Optional[Tuple[float, int]]:
        if self.age_estimator is None:
            return None
        prediction = self.age_estimator.predict_from_crops(
            face_rgb=face_crop,
            body_rgb=body_crop,
            aligned_face_rgb=aligned_crop,
        )
        if prediction is None:
            return None
        calibrated_age = self._calibrate_age_prediction(
            age_years=prediction.age_years,
            aligned_crop=aligned_crop,
        )
        return calibrated_age, timestamp_ms

    def _run_emotion_inference(
        self,
        face_crop: Optional[np.ndarray],
        aligned_crop: Optional[np.ndarray],
        context_crop: Optional[np.ndarray],
        rule_scores: Dict[str, float],
        mouth_open: float,
        eye_open: float,
        frame_quality: float,
        timestamp_ms: int,
    ) -> Optional[Tuple[Dict[str, float], float, float, int]]:
        if self.emotion_estimator is None:
            return None
        prediction = self.emotion_estimator.predict(
            face_rgb=face_crop,
            aligned_face_rgb=aligned_crop,
            context_rgb=context_crop,
        )
        if prediction is None:
            return None
        scores = self._calibrate_emotion_prediction(
            model_scores=prediction.scores,
            rule_scores=rule_scores,
            mouth_open=mouth_open,
            eye_open=eye_open,
            frame_quality=frame_quality,
        )
        return scores, prediction.confidence, prediction.margin, timestamp_ms

    def _emotion_frame_quality(
        self,
        motion_delta: float,
        yaw_ratio: float,
        roll_degrees: float,
    ) -> float:
        motion_quality = 1.0 - self._clamp(
            motion_delta / max(self.config.emotion_stability_max_motion, 1e-6)
        )
        yaw_quality = 1.0 - self._clamp(
            yaw_ratio / max(self.config.emotion_stability_max_yaw_ratio, 1e-6)
        )
        roll_quality = 1.0 - self._clamp(
            roll_degrees / max(self.config.emotion_stability_max_roll_degrees, 1e-6)
        )
        return self._clamp(
            motion_quality * 0.42
            + yaw_quality * 0.33
            + roll_quality * 0.25
        )

    def _calibrate_emotion_prediction(
        self,
        model_scores: Dict[str, float],
        rule_scores: Dict[str, float],
        mouth_open: float,
        eye_open: float,
        frame_quality: float,
    ) -> Dict[str, float]:
        talking_penalty = self._clamp((mouth_open - 0.34) / 0.28) * self._clamp(
            (0.20 - rule_scores["surprised"]) / 0.20
        )
        narrow_eye_penalty = self._clamp((0.40 - eye_open) / 0.30)

        calibrated = {
            "happy": self._clamp(
                model_scores.get("happy", 0.0) * 0.94
                + rule_scores["happy"] * 0.06
                - rule_scores["angry"] * 0.04
            ),
            "surprised": self._clamp(
                model_scores.get("surprised", 0.0) * (1.0 - talking_penalty * 0.32)
                * (1.0 - narrow_eye_penalty * 0.12)
                + rule_scores["surprised"] * 0.10
            ),
            "angry": self._clamp(
                model_scores.get("angry", 0.0)
                + rule_scores["angry"] * 0.14
                - rule_scores["happy"] * 0.12
                - rule_scores["surprised"] * 0.05
            ),
            "sad": self._clamp(
                model_scores.get("sad", 0.0)
                + rule_scores["sad"] * 0.16
                - rule_scores["happy"] * 0.10
                - rule_scores["surprised"] * 0.08
            ),
            "neutral": self._clamp(
                model_scores.get("neutral", 0.0) * (0.90 + frame_quality * 0.10)
                + rule_scores["neutral"] * 0.10
            ),
        }
        calibrated["neutral"] = self._clamp(
            max(
                calibrated["neutral"],
                1.0
                - max(
                    calibrated["happy"],
                    calibrated["surprised"],
                    calibrated["angry"],
                    calibrated["sad"],
                ),
            )
        )
        return calibrated

    def _blend_emotion_sources(
        self,
        rule_scores: Dict[str, float],
        model_scores: Dict[str, float],
        confidence: float,
        margin: float,
        frame_quality: float,
        mouth_open: float,
        eye_open: float,
    ) -> Dict[str, float]:
        confidence_strength = self._clamp((confidence - 0.22) / 0.46)
        margin_strength = self._clamp(margin / 0.28)
        model_weight = self._clamp(
            self.config.emotion_model_base_weight * (0.58 + frame_quality * 0.42)
            + confidence_strength * 0.10
            + margin_strength * 0.06,
            self.config.emotion_model_min_weight,
            self.config.emotion_model_max_weight,
        )
        rule_weight = 1.0 - model_weight

        talking_penalty = self._clamp((mouth_open - 0.34) / 0.28) * self._clamp(
            (0.18 - rule_scores["surprised"]) / 0.18
        )
        low_eye_penalty = self._clamp((0.38 - eye_open) / 0.30)

        happy = self._clamp(
            model_scores.get("happy", 0.0) * (model_weight + 0.04)
            + rule_scores["happy"] * max(rule_weight - 0.04, 0.0)
            - rule_scores["angry"] * 0.05
        )
        surprised = self._clamp(
            (
                model_scores.get("surprised", 0.0) * max(model_weight - 0.02, 0.0)
                + rule_scores["surprised"] * min(rule_weight + 0.02, 1.0)
            )
            * (1.0 - talking_penalty * 0.34)
            * (1.0 - low_eye_penalty * 0.12)
        )
        angry = self._clamp(
            model_scores.get("angry", 0.0) * model_weight
            + rule_scores["angry"] * rule_weight
            - rule_scores["happy"] * 0.08
            - rule_scores["sad"] * 0.03
        )
        sad = self._clamp(
            model_scores.get("sad", 0.0) * model_weight
            + rule_scores["sad"] * rule_weight
            - rule_scores["happy"] * 0.12
            - rule_scores["surprised"] * 0.08
        )

        emotional_peak = max(happy, surprised, angry, sad)
        neutral = self._clamp(
            max(
                model_scores.get("neutral", 0.0) * model_weight
                + rule_scores["neutral"] * rule_weight,
                1.0 - emotional_peak * 1.08,
            )
        )
        return {
            "happy": happy,
            "surprised": surprised,
            "angry": angry,
            "sad": sad,
            "neutral": neutral,
        }

    def _resolve_top_label(
        self,
        track: TrackedFace,
        metrics: Dict[str, float],
    ) -> str:
        expressive_scores = {
            "Mutlu": metrics.get("happy", 0.0),
            "Şaşkın": metrics.get("surprised", 0.0),
            "Kızgın": metrics.get("angry", 0.0),
            "Üzgün": metrics.get("sad", 0.0),
        }
        neutral_score = float(metrics.get("neutral", 0.0))
        expressive_label, expressive_score = max(
            expressive_scores.items(),
            key=lambda item: item[1],
        )
        if expressive_score >= self.config.emotion_label_threshold:
            candidate_label = expressive_label
            candidate_score = float(expressive_score)
        else:
            candidate_label = "Sakin"
            candidate_score = neutral_score

        valid_expressions = {"Sakin": neutral_score, **expressive_scores}
        current_label = track.top_label_state or "Sakin"
        current_score = float(valid_expressions.get(current_label, 0.0))

        if candidate_label != "Sakin" and candidate_score < self.config.emotion_label_threshold:
            if current_label != "Sakin" and current_score >= self.config.emotion_label_hold_threshold:
                return current_label
            track.top_label_state = "Sakin"
            track.label_candidate = "Sakin"
            track.label_candidate_frames = 0
            return "Sakin"

        if candidate_label == current_label:
            track.label_candidate = candidate_label
            track.label_candidate_frames = 0
            track.top_label_state = candidate_label
            return candidate_label

        should_switch = (
            candidate_label == "Sakin"
            or current_label == "Sakin"
            or candidate_score >= current_score + self.config.emotion_label_switch_margin
            or current_score < self.config.emotion_label_hold_threshold
        )
        if not should_switch:
            return current_label

        if current_label == "Sakin" and candidate_label != "Sakin":
            track.top_label_state = candidate_label
            track.label_candidate = candidate_label
            track.label_candidate_frames = 0
            return candidate_label

        if track.label_candidate == candidate_label:
            track.label_candidate_frames += 1
        else:
            track.label_candidate = candidate_label
            track.label_candidate_frames = 1

        if track.label_candidate_frames >= self.config.emotion_label_switch_frames:
            track.top_label_state = candidate_label
            track.label_candidate = candidate_label
            track.label_candidate_frames = 0
            return candidate_label

        return current_label

    def _calibrate_age_prediction(
        self,
        age_years: float,
        aligned_crop: Optional[np.ndarray],
    ) -> float:
        calibrated = age_years + self.config.age_bias_years
        if getattr(self.age_estimator, "backend_name", "") == "openvino":
            calibrated += self._gray_hair_bonus(aligned_crop)
        return normalize_age_years(calibrated, min_age=0.0, max_age=100.0)

    def _gray_hair_bonus(self, aligned_crop: Optional[np.ndarray]) -> float:
        if aligned_crop is None or aligned_crop.size == 0:
            return 0.0

        crop = cast(np.ndarray, aligned_crop)
        height, width = crop.shape[:2]
        x_min = int(width * 0.24)
        x_max = int(width * 0.76)
        y_min = 0
        y_max = max(int(height * 0.18), 1)
        top_region = crop[y_min:y_max, x_min:x_max]
        if top_region.size == 0:
            return 0.0

        hsv = cv2.cvtColor(top_region, cv2.COLOR_RGB2HSV)
        saturation = hsv[:, :, 1].astype(np.float32)
        value = hsv[:, :, 2].astype(np.float32)

        silver_mask = (
            (saturation < 38.0)
            & (value > 55.0)
            & (value < 220.0)
        )
        silver_ratio = float(np.mean(silver_mask))
        if silver_ratio <= 0.16:
            return 0.0

        normalized = self._clamp((silver_ratio - 0.16) / 0.36)
        return normalized * self.config.gray_hair_bonus_max_years

    def _update_age_history(self, track: TrackedFace, age_years: float) -> float:
        if track.age_history is None:
            track.age_history = []

        age_history = cast(List[float], track.age_history)
        age_history.append(age_years)
        max_size = max(int(self.config.age_history_size), 1)
        if len(age_history) > max_size:
            # Using list comprehension to avoid slice indexing issues in some strict analyzers
            start_idx = len(age_history) - max_size
            age_history = [age_history[i] for i in range(start_idx, len(age_history))]
        track.age_history = age_history

        ordered = cast(List[float], sorted(age_history))
        trim = min(int(self.config.age_history_trim), max(0, (len(ordered) - 1) // 2))
        if trim > 0:
            # Using list comprehension to avoid slice indexing issues
            ordered = [ordered[i] for i in range(trim, len(ordered) - trim)]
        if not ordered:
            return age_years
        return float(np.mean(np.asarray(ordered, dtype=np.float32)))

    def _age_motion_delta(
        self,
        track: TrackedFace,
        center: np.ndarray,
        bbox: tuple[int, int, int, int],
    ) -> float:
        if track.hits == 0:
            return 1.0

        scale = max(float(bbox[2] - bbox[0]), float(track.bbox[2] - track.bbox[0]), 1.0)
        return float(np.linalg.norm(center - track.anchor) / scale)

    @staticmethod
    def _age_pose_metrics(points: np.ndarray) -> tuple[float, float]:
        left_eye = np.mean(points[[33, 133, 159, 145]], axis=0)
        right_eye = np.mean(points[[362, 263, 386, 374]], axis=0)
        nose_tip = points[1]

        left_span = float(np.linalg.norm(nose_tip - left_eye))
        right_span = float(np.linalg.norm(nose_tip - right_eye))
        yaw_ratio = abs(left_span - right_span) / max(left_span + right_span, 1e-6)

        eye_vector = right_eye - left_eye
        roll_degrees = abs(float(np.degrees(np.arctan2(eye_vector[1], eye_vector[0]))))
        return yaw_ratio, roll_degrees

    def _is_age_frame_stable(
        self,
        track: TrackedFace,
        metrics: Dict[str, float],
        motion_delta: float,
        yaw_ratio: float,
        roll_degrees: float,
    ) -> bool:
        if track.hits < 2:
            return False

        strongest_expression = max(
            metrics.get("happy", 0.0),
            metrics.get("surprised", 0.0),
            metrics.get("angry", 0.0),
            metrics.get("sad", 0.0),
        )
        return bool(
            metrics.get("tracking_confidence", 0.0)
            >= self.config.age_stability_min_tracking_confidence
            and metrics.get("presence", 0.0) >= self.config.age_stability_min_presence
            and strongest_expression <= self.config.age_stability_max_expression
            and metrics.get("mouth_open", 1.0) <= self.config.age_stability_max_mouth_open
            and motion_delta <= self.config.age_stability_max_motion
            and yaw_ratio <= self.config.age_stability_max_yaw_ratio
            and roll_degrees <= self.config.age_stability_max_roll_degrees
        )

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
                det_bbox = cast(Tuple[int, int, int, int], detection["bbox"])
                track = self._create_track(
                    anchor=center,
                    bbox=det_bbox,
                    timestamp_ms=timestamp_ms,
                )

            assignments.append((detection, track))

        return assignments

    def _extract_aligned_age_crop(
        self,
        frame_rgb: np.ndarray,
        points: np.ndarray,
    ) -> Optional[np.ndarray]:
        output_size = int(self.config.age_aligned_crop_size)
        if output_size < 64:
            output_size = 64

        left_eye = np.mean(points[[33, 133, 159, 145]], axis=0).astype(np.float32)
        right_eye = np.mean(points[[362, 263, 386, 374]], axis=0).astype(np.float32)
        mouth_center = np.mean(points[[61, 291, 13, 14]], axis=0).astype(np.float32)

        source = np.array([left_eye, right_eye, mouth_center], dtype=np.float32)
        if cv2.contourArea(source.reshape(-1, 1, 2)) < 1.0:
            return None

        destination = np.array(
            [
                [output_size * 0.32, output_size * 0.37],
                [output_size * 0.68, output_size * 0.37],
                [output_size * 0.50, output_size * 0.72],
            ],
            dtype=np.float32,
        )
        matrix = cv2.getAffineTransform(source, destination)
        return cv2.warpAffine(
            frame_rgb,
            matrix,
            (output_size, output_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

    def _extract_age_crop(
        self,
        frame_rgb: np.ndarray,
        bbox: tuple[int, int, int, int],
        padding_ratio: float,
        top_bias_ratio: float,
    ) -> Optional[np.ndarray]:
        frame_height, frame_width = frame_rgb.shape[:2]
        x_min, y_min, x_max, y_max = bbox

        width = max(float(x_max - x_min), 1.0)
        height = max(float(y_max - y_min), 1.0)

        crop_size = max(width * (1.0 + padding_ratio * 2.0), height * (1.0 + padding_ratio * 2.4))
        center_x = (x_min + x_max) / 2.0
        center_y = ((y_min + y_max) / 2.0) - (height * top_bias_ratio)
        half = crop_size / 2.0

        crop_x_min = int(np.clip(np.floor(center_x - half), 0, frame_width - 1))
        crop_y_min = int(np.clip(np.floor(center_y - half), 0, frame_height - 1))
        crop_x_max = int(np.clip(np.ceil(center_x + half), crop_x_min + 1, frame_width))
        crop_y_max = int(np.clip(np.ceil(center_y + half), crop_y_min + 1, frame_height))

        if crop_x_max <= crop_x_min or crop_y_max <= crop_y_min:
            return None

        return frame_rgb[crop_y_min:crop_y_max, crop_x_min:crop_x_max]

    def _extract_body_crop(
        self,
        frame_rgb: np.ndarray,
        bbox: tuple[int, int, int, int],
    ) -> Optional[np.ndarray]:
        frame_height, frame_width = frame_rgb.shape[:2]
        x_min, y_min, x_max, y_max = bbox

        face_width = max(float(x_max - x_min), 1.0)
        face_height = max(float(y_max - y_min), 1.0)
        center_x = (x_min + x_max) / 2.0

        crop_width = face_width * self.config.age_body_width_scale
        crop_height = face_height * (
            self.config.age_body_above_face_scale + self.config.age_body_below_face_scale
        )

        crop_x_min = int(np.clip(np.floor(center_x - crop_width / 2.0), 0, frame_width - 1))
        crop_x_max = int(np.clip(np.ceil(center_x + crop_width / 2.0), crop_x_min + 1, frame_width))
        crop_y_min = int(
            np.clip(
                np.floor(y_min - face_height * self.config.age_body_above_face_scale),
                0,
                frame_height - 1,
            )
        )
        crop_y_max = int(
            np.clip(
                np.ceil(y_max + face_height * self.config.age_body_below_face_scale),
                crop_y_min + 1,
                frame_height,
            )
        )

        if crop_x_max <= crop_x_min or crop_y_max <= crop_y_min:
            return None

        return frame_rgb[crop_y_min:crop_y_max, crop_x_min:crop_x_max]

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
            smoother=MetricSmoother(
                self.config.smoothing_alpha,
                emotion_backend=self.config.emotion_backend.lower(),
            ),
            emotion_scores=self._default_emotion_scores(),
            age_smoother=(
                ScalarSmoother(self.config.age_smoothing_alpha)
                if self.age_estimator is not None
                else None
            ),
            age_history=[],
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
            track = self.tracks.pop(track_id, None)
            if track is None:
                continue
            if track.pending_age_future is not None:
                track.pending_age_future.cancel()
            if track.pending_emotion_future is not None:
                track.pending_emotion_future.cancel()

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
            "mouth_open": mouth_open,
            "happy": happy_score,
            "surprised": surprised_score,
            "angry": angry_score,
            "sad": sad_score,
            "neutral": self._clamp(
                1.0 - max(happy_score, surprised_score, angry_score, sad_score)
            ),
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
        self.executor.shutdown(wait=False, cancel_futures=True)
        if self.emotion_estimator is not None:
            self.emotion_estimator.close()
        if self.age_estimator is not None:
            self.age_estimator.close()
        self.landmarker.close()
