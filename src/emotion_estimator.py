from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

try:
    from emotiefflib.facial_analysis import EmotiEffLibRecognizer
except ImportError:  # pragma: no cover - handled at runtime
    EmotiEffLibRecognizer = None

try:
    from .config import AppConfig
except ImportError:
    from config import AppConfig


@dataclass
class EmotionPrediction:
    scores: Dict[str, float]
    raw_label: Optional[str] = None
    raw_scores: Optional[Dict[str, float]] = None
    confidence: float = 0.0
    margin: float = 0.0


class BaseEmotionEstimator:
    backend_name = "base"

    def predict(
        self,
        face_rgb: Optional[np.ndarray],
        aligned_face_rgb: Optional[np.ndarray] = None,
        context_rgb: Optional[np.ndarray] = None,
    ) -> Optional[EmotionPrediction]:
        raise NotImplementedError

    def close(self) -> None:
        return None


class EmotiEffLibEmotionEstimator(BaseEmotionEstimator):
    backend_name = "emotiefflib"

    def __init__(self, config: AppConfig) -> None:
        if EmotiEffLibRecognizer is None:
            raise RuntimeError(
                "EmotiEffLib is not installed. Run `pip install -r requirements.txt` first."
            )

        self.recognizer = EmotiEffLibRecognizer(
            engine=config.emotion_emotiefflib_engine,
            model_name=config.emotion_emotiefflib_model_name,
            device="cpu",
        )
        self.idx_to_emotion_class = getattr(self.recognizer, "idx_to_emotion_class", {})

    @staticmethod
    def _clamp(value: float) -> float:
        return float(max(0.0, min(1.0, value)))

    def _map_scores(self, raw_scores: Dict[str, float]) -> Dict[str, float]:
        angry = raw_scores.get("anger", 0.0) + raw_scores.get("disgust", 0.0) * 0.30
        surprised = raw_scores.get("surprise", 0.0) + raw_scores.get("fear", 0.0) * 0.28
        happy = raw_scores.get("happiness", 0.0)
        sad = raw_scores.get("sadness", 0.0)
        neutral = raw_scores.get("neutral", 0.0) + raw_scores.get("contempt", 0.0) * 0.18

        return {
            "happy": self._clamp(happy),
            "surprised": self._clamp(surprised),
            "angry": self._clamp(angry),
            "sad": self._clamp(sad),
            "neutral": self._clamp(neutral),
        }

    def _prediction_inputs(
        self,
        face_rgb: Optional[np.ndarray],
        aligned_face_rgb: Optional[np.ndarray],
        context_rgb: Optional[np.ndarray],
    ) -> List[np.ndarray]:
        inputs: List[np.ndarray] = []
        for image in (aligned_face_rgb, face_rgb, context_rgb):
            if image is None or image.size == 0:
                continue
            if any(existing.shape == image.shape and np.array_equal(existing, image) for existing in inputs):
                continue
            inputs.append(image)
        return inputs

    def predict(
        self,
        face_rgb: Optional[np.ndarray],
        aligned_face_rgb: Optional[np.ndarray] = None,
        context_rgb: Optional[np.ndarray] = None,
    ) -> Optional[EmotionPrediction]:
        inputs = self._prediction_inputs(
            face_rgb=face_rgb,
            aligned_face_rgb=aligned_face_rgb,
            context_rgb=context_rgb,
        )
        if not inputs:
            return None

        model_input: object = inputs[0] if len(inputs) == 1 else inputs
        _, scores = self.recognizer.predict_emotions(model_input, logits=False)
        score_matrix = np.asarray(scores, dtype=np.float32)
        if score_matrix.ndim == 1:
            score_matrix = score_matrix[np.newaxis, ...]
        score_vector = score_matrix.mean(axis=0)

        raw_scores: Dict[str, float] = {}
        for index, value in enumerate(score_vector.tolist()):
            label = str(self.idx_to_emotion_class.get(index, index)).strip().lower()
            raw_scores[label] = float(value)

        mapped_scores = self._map_scores(raw_scores)
        sorted_scores = sorted(mapped_scores.items(), key=lambda item: item[1], reverse=True)
        raw_label = sorted_scores[0][0] if sorted_scores else None
        confidence = float(sorted_scores[0][1]) if sorted_scores else 0.0
        margin = (
            float(sorted_scores[0][1] - sorted_scores[1][1])
            if len(sorted_scores) > 1
            else confidence
        )
        return EmotionPrediction(
            scores=mapped_scores,
            raw_label=raw_label,
            raw_scores=raw_scores,
            confidence=confidence,
            margin=margin,
        )


def create_emotion_estimator(config: AppConfig) -> Optional[BaseEmotionEstimator]:
    backend = config.emotion_backend.lower()
    if backend == "rules":
        return None
    if backend == "emotiefflib":
        return EmotiEffLibEmotionEstimator(config)
    raise ValueError(f"Unsupported emotion backend '{config.emotion_backend}'")
