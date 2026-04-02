import multiprocessing
import sys
from pathlib import Path

from src.config import DEFAULT_CONFIG


DEFAULT_CONFIG.window_name = "AI Yüz Tanıma"


def _apply_bundled_paths() -> None:
    bundle_root = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    bundled_face_model = bundle_root / "models" / "face_landmarker.task"
    bundled_mivolo_cache = bundle_root / "mivolo_v2"

    if bundled_face_model.exists():
        DEFAULT_CONFIG.model_path = str(bundled_face_model)
    if bundled_mivolo_cache.exists():
        DEFAULT_CONFIG.age_mivolo_cache_dir = str(bundled_mivolo_cache)


def main() -> None:
    multiprocessing.freeze_support()
    _apply_bundled_paths()
    from src.main import main as run_main

    run_main()


if __name__ == "__main__":
    main()
