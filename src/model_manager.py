from pathlib import Path
from urllib.request import urlretrieve


FACE_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)
OPENVINO_AGE_GENDER_MODEL_URLS = {
    "FP32": {
        "xml": (
            "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/"
            "models_bin/1/age-gender-recognition-retail-0013/FP32/"
            "age-gender-recognition-retail-0013.xml"
        ),
        "bin": (
            "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/"
            "models_bin/1/age-gender-recognition-retail-0013/FP32/"
            "age-gender-recognition-retail-0013.bin"
        ),
    },
    "FP16": {
        "xml": (
            "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/"
            "models_bin/1/age-gender-recognition-retail-0013/FP16/"
            "age-gender-recognition-retail-0013.xml"
        ),
        "bin": (
            "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/"
            "models_bin/1/age-gender-recognition-retail-0013/FP16/"
            "age-gender-recognition-retail-0013.bin"
        ),
    },
    "FP16-INT8": {
        "xml": (
            "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/"
            "models_bin/1/age-gender-recognition-retail-0013/FP16-INT8/"
            "age-gender-recognition-retail-0013.xml"
        ),
        "bin": (
            "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/"
            "models_bin/1/age-gender-recognition-retail-0013/FP16-INT8/"
            "age-gender-recognition-retail-0013.bin"
        ),
    },
}


def _download_file(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading model file to {target} ...")
    urlretrieve(url, target)


def ensure_face_landmarker_model(model_path: str) -> str:
    target = Path(model_path)

    if target.exists():
        return str(target)

    _download_file(FACE_LANDMARKER_MODEL_URL, target)
    return str(target)


def ensure_openvino_age_gender_model(model_path: str, precision: str = "FP16") -> str:
    normalized_precision = precision.upper()
    if normalized_precision not in OPENVINO_AGE_GENDER_MODEL_URLS:
        available = ", ".join(sorted(OPENVINO_AGE_GENDER_MODEL_URLS))
        raise ValueError(
            f"Unsupported age model precision '{precision}'. Available: {available}"
        )

    xml_target = Path(model_path)
    bin_target = xml_target.with_suffix(".bin")

    if xml_target.exists() and bin_target.exists():
        return str(xml_target)

    source_urls = OPENVINO_AGE_GENDER_MODEL_URLS[normalized_precision]
    if not xml_target.exists():
        _download_file(source_urls["xml"], xml_target)
    if not bin_target.exists():
        _download_file(source_urls["bin"], bin_target)

    return str(xml_target)
