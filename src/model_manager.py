from pathlib import Path
from urllib.request import urlretrieve


FACE_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)


def ensure_face_landmarker_model(model_path: str) -> str:
    target = Path(model_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists():
        return str(target)

    print(f"Downloading MediaPipe face landmarker model to {target} ...")
    urlretrieve(FACE_LANDMARKER_MODEL_URL, target)
    return str(target)
