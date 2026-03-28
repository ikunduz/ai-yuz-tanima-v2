from dataclasses import dataclass
from typing import Dict, Optional

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


class BaseEmotionEstimator:
    backend_name = "base"

    def predict(self, face_rgb: np.ndarray) -> Optional[EmotionPrediction]:
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
        angry = raw_scores.get("anger", 0.0) + raw_scores.get("disgust", 0.0) * 0.42
        surprised = raw_scores.get("surprise", 0.0) + raw_scores.get("fear", 0.0) * 0.48
        happy = raw_scores.get("happiness", 0.0)
        sad = raw_scores.get("sadness", 0.0)
        neutral = raw_scores.get("neutral", 0.0) + raw_scores.get("contempt", 0.0) * 0.35

        return {
            "happy": self._clamp(happy),
            "surprised": self._clamp(surprised),
            "angry": self._clamp(angry),
            "sad": self._clamp(sad),
            "neutral": self._clamp(neutral),
        }

    def predict(self, face_rgb: np.ndarray) -> Optional[EmotionPrediction]:
        if face_rgb.size == 0:
            return None

        labels, scores = self.recognizer.predict_emotions(face_rgb, logits=False)
        score_vector = np.asarray(scores, dtype=np.float32)
        if score_vector.ndim == 2:
            score_vector = score_vector[0]

        raw_scores: Dict[str, float] = {}
        for index, value in enumerate(score_vector.tolist()):
            label = str(self.idx_to_emotion_class.get(index, index)).strip().lower()
            raw_scores[label] = float(value)

        mapped_scores = self._map_scores(raw_scores)
        raw_label = str(labels[0]) if labels else None
        return EmotionPrediction(
            scores=mapped_scores,
            raw_label=raw_label,
            raw_scores=raw_scores,
        )


def create_emotion_estimator(config: AppConfig) -> Optional[BaseEmotionEstimator]:
    backend = config.emotion_backend.lower()
    if backend == "rules":
        return None
    if backend == "emotiefflib":
        return EmotiEffLibEmotionEstimator(config)
    raise ValueError(f"Unsupported emotion backend '{config.emotion_backend}'")
