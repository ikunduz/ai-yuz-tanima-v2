from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Optional

import cv2
import numpy as np

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    import torch
except ImportError:  # pragma: no cover - handled at runtime
    torch = None

try:
    from transformers import AutoConfig, AutoModelForImageClassification
except ImportError:  # pragma: no cover - handled at runtime
    AutoConfig = None
    AutoModelForImageClassification = None

try:
    from openvino import Core
except ImportError:  # pragma: no cover - handled at runtime
    Core = None

try:
    from .config import AppConfig
    from .model_manager import ensure_openvino_age_gender_model
except ImportError:
    from config import AppConfig
    from model_manager import ensure_openvino_age_gender_model


IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass
class AgePrediction:
    age_years: float
    age_label: str
    gender_label: Optional[str] = None
    gender_confidence: Optional[float] = None


def age_label_for_years(age_years: float) -> str:
    rounded = normalize_age_years(age_years)
    if rounded < 6.0:
        return "0-5"
    if rounded < 10.0:
        return "6-9"
    if rounded < 13.0:
        return "10-12"
    if rounded < 18.0:
        return "13-17"
    if rounded < 25.0:
        return "18-24"
    if rounded < 30.0:
        return "25-29"
    if rounded < 35.0:
        return "30-34"
    if rounded < 40.0:
        return "35-39"
    if rounded < 45.0:
        return "40-44"
    if rounded < 50.0:
        return "45-49"
    if rounded < 55.0:
        return "50-54"
    if rounded < 60.0:
        return "55-59"
    if rounded < 65.0:
        return "60-64"
    if rounded < 70.0:
        return "65-69"
    if rounded < 80.0:
        return "70-79"
    return "80+"


def normalize_age_years(
    age_years: float,
    min_age: float = 0.0,
    max_age: float = 100.0,
) -> float:
    return float(np.clip(age_years, min_age, max_age))


def _letterbox_rgb(image_rgb: np.ndarray, target_size: int) -> np.ndarray:
    height, width = image_rgb.shape[:2]
    if height == 0 or width == 0:
        return np.zeros((target_size, target_size, 3), dtype=np.uint8)

    scale = min(target_size / float(height), target_size / float(width))
    resized_width = max(int(round(width * scale)), 1)
    resized_height = max(int(round(height * scale)), 1)
    resized = cv2.resize(image_rgb, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

    pad_x = target_size - resized_width
    pad_y = target_size - resized_height
    left = pad_x // 2
    right = pad_x - left
    top = pad_y // 2
    bottom = pad_y - top
    return cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )


class BaseAgeEstimator:
    backend_name = "base"

    def predict_from_crops(
        self,
        face_rgb: Optional[np.ndarray],
        body_rgb: Optional[np.ndarray] = None,
        aligned_face_rgb: Optional[np.ndarray] = None,
    ) -> Optional[AgePrediction]:
        raise NotImplementedError

    def close(self) -> None:
        return None


class OpenVinoAgeEstimator(BaseAgeEstimator):
    backend_name = "openvino"

    def __init__(self, config: AppConfig) -> None:
        if Core is None:
            raise RuntimeError(
                "OpenVINO is not installed. Run `pip install -r requirements.txt` first."
            )

        model_path = ensure_openvino_age_gender_model(
            config.age_model_path,
            precision=config.age_model_precision,
        )
        self.core = Core()
        self.compiled_model = self.core.compile_model(model=model_path, device_name="CPU")
        self.input_layer = self.compiled_model.input(0)
        self.age_output = None
        self.gender_output = None
        for output in self.compiled_model.outputs:
            name = output.any_name.lower()
            shape = tuple(int(dim) for dim in output.shape)
            if "age" in name or shape == (1, 1, 1, 1):
                self.age_output = output
                continue
            if "prob" in name or shape == (1, 2, 1, 1):
                self.gender_output = output

        if self.age_output is None or self.gender_output is None:
            raise RuntimeError("Age model outputs could not be identified correctly.")
        self.input_height = int(self.input_layer.shape[2])
        self.input_width = int(self.input_layer.shape[3])

    def _predict_single(self, face_rgb: np.ndarray) -> Optional[AgePrediction]:
        if face_rgb.size == 0:
            return None

        face_bgr = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR)
        resized = cv2.resize(
            face_bgr,
            (self.input_width, self.input_height),
            interpolation=cv2.INTER_AREA,
        )
        tensor = np.transpose(resized, (2, 0, 1))[np.newaxis, ...].astype(np.float32)
        outputs = self.compiled_model([tensor])

        raw_age = float(np.asarray(outputs[self.age_output]).reshape(-1)[0] * 100.0)
        age_years = normalize_age_years(raw_age, min_age=18.0, max_age=75.0)

        gender_scores = np.asarray(outputs[self.gender_output]).reshape(-1)
        gender_index = int(np.argmax(gender_scores)) if gender_scores.size else 0
        gender_label = "Erkek" if gender_index == 1 else "Kadin"
        gender_confidence = (
            float(gender_scores[gender_index]) if gender_scores.size > gender_index else None
        )

        return AgePrediction(
            age_years=age_years,
            age_label=age_label_for_years(age_years),
            gender_label=gender_label,
            gender_confidence=gender_confidence,
        )

    def predict_from_crops(
        self,
        face_rgb: Optional[np.ndarray],
        body_rgb: Optional[np.ndarray] = None,
        aligned_face_rgb: Optional[np.ndarray] = None,
    ) -> Optional[AgePrediction]:
        inputs = [
            image
            for image in (aligned_face_rgb, face_rgb)
            if image is not None and image.size != 0
        ]
        predictions = [prediction for prediction in (self._predict_single(image) for image in inputs) if prediction]
        if not predictions:
            return None
        if len(predictions) == 1:
            return predictions[0]

        weights = np.asarray([0.68, 0.32], dtype=np.float32)[: len(predictions)]
        weights /= weights.sum()
        age_years = normalize_age_years(
            float(sum(pred.age_years * weight for pred, weight in zip(predictions, weights)))
        )
        return AgePrediction(
            age_years=age_years,
            age_label=age_label_for_years(age_years),
        )


class MiVoloAgeEstimator(BaseAgeEstimator):
    backend_name = "mivolo"

    def __init__(self, config: AppConfig) -> None:
        if torch is None or AutoConfig is None or AutoModelForImageClassification is None:
            raise RuntimeError(
                "MiVOLO dependencies are missing. Run `pip install -r requirements.txt` first."
            )

        self.device_name = self._resolve_device_name(config.age_mivolo_device)
        self.device = torch.device(self.device_name)
        self.dtype = torch.float16 if self.device.type != "cpu" else torch.float32

        cache_dir = config.age_mivolo_cache_dir
        local_files_only = self._has_cached_model_files(cache_dir)
        self.config = AutoConfig.from_pretrained(
            config.age_mivolo_model_id,
            trust_remote_code=True,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )

        try:
            self.model = AutoModelForImageClassification.from_pretrained(
                config.age_mivolo_model_id,
                trust_remote_code=True,
                cache_dir=cache_dir,
                torch_dtype=self.dtype,
                local_files_only=local_files_only,
            )
        except Exception:
            if self.device.type == "mps" and self.dtype == torch.float16:
                self.dtype = torch.float32
                self.model = AutoModelForImageClassification.from_pretrained(
                    config.age_mivolo_model_id,
                    trust_remote_code=True,
                    cache_dir=cache_dir,
                    torch_dtype=self.dtype,
                    local_files_only=local_files_only,
                )
            else:
                raise

        self.model.to(self.device)
        self.model.eval()
        self.input_size = int(getattr(self.config, "input_size", 384))
        self.mean = IMAGENET_MEAN
        self.std = IMAGENET_STD

    @staticmethod
    def _has_cached_model_files(cache_dir: str) -> bool:
        cache_path = Path(cache_dir)
        if not cache_path.exists():
            return False
        return any(cache_path.rglob("config.json"))

    @staticmethod
    def _resolve_device_name(preference: str) -> str:
        normalized = preference.lower()
        if normalized != "auto":
            return normalized
        if torch is not None and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _prepare_inputs(self, images: list[Optional[np.ndarray]]) -> "torch.Tensor":
        prepared = []
        for image in images:
            if image is None or image.size == 0:
                tensor = torch.zeros((3, self.input_size, self.input_size), dtype=torch.float32)
            else:
                letterboxed = _letterbox_rgb(image, self.input_size).astype(np.float32) / 255.0
                normalized = (letterboxed - self.mean) / self.std
                tensor = torch.from_numpy(np.transpose(normalized, (2, 0, 1)).copy())
            prepared.append(tensor.unsqueeze(0))

        batch = torch.cat(prepared, dim=0)
        return batch.to(device=self.device, dtype=self.dtype)

    @staticmethod
    def _normalize_gender_label(label: Optional[str]) -> Optional[str]:
        if label is None:
            return None
        normalized = str(label).strip().lower()
        if normalized == "male":
            return "Erkek"
        if normalized == "female":
            return "Kadin"
        return label

    def predict_from_crops(
        self,
        face_rgb: Optional[np.ndarray],
        body_rgb: Optional[np.ndarray] = None,
        aligned_face_rgb: Optional[np.ndarray] = None,
    ) -> Optional[AgePrediction]:
        face_input_image = face_rgb if face_rgb is not None and face_rgb.size != 0 else aligned_face_rgb
        if face_input_image is None and body_rgb is None:
            return None

        faces_input = self._prepare_inputs([face_input_image])
        body_input = self._prepare_inputs([body_rgb])

        with torch.inference_mode():
            output = self.model(faces_input=faces_input, body_input=body_input)

        age_years = normalize_age_years(
            float(output.age_output[0].detach().float().cpu().item()),
            min_age=0.0,
            max_age=100.0,
        )
        gender_label = None
        gender_confidence = None
        if output.gender_class_idx is not None and output.gender_probs is not None:
            gender_index = int(output.gender_class_idx[0].detach().cpu().item())
            id2label = getattr(self.config, "gender_id2label", {0: "male", 1: "female"})
            gender_label = self._normalize_gender_label(id2label.get(gender_index))
            gender_confidence = float(output.gender_probs[0].detach().float().cpu().item())

        return AgePrediction(
            age_years=age_years,
            age_label=age_label_for_years(age_years),
            gender_label=gender_label,
            gender_confidence=gender_confidence,
        )

    def close(self) -> None:
        if torch is not None:
            if self.device.type == "mps":
                torch.mps.empty_cache()
            elif self.device.type == "cuda":
                torch.cuda.empty_cache()


def create_age_estimator(config: AppConfig) -> Optional[BaseAgeEstimator]:
    if not config.age_enabled:
        return None

    backend = config.age_backend.lower()
    if backend == "openvino":
        return OpenVinoAgeEstimator(config)
    if backend == "mivolo":
        return MiVoloAgeEstimator(config)
    raise ValueError(f"Unsupported age backend '{config.age_backend}'")
