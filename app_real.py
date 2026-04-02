import multiprocessing
import subprocess
import sys
import traceback
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


def _show_startup_error(exc: Exception) -> None:
    if not getattr(sys, "frozen", False):
        raise exc

    log_dir = Path.home() / "Library" / "Logs" / "AIYuzTanima"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "AI Yuz Tanima.log"
    log_path.write_text(traceback.format_exc(), encoding="utf-8")
    message = (
        "Uygulama acilirken hata olustu.\n\n"
        f"{exc}\n\n"
        f"Log: {log_path}"
    )
    subprocess.run(
        [
            "osascript",
            "-e",
            "on run argv",
            "-e",
            "display alert (item 2 of argv) message (item 1 of argv) as critical buttons {\"Tamam\"} default button \"Tamam\"",
            "-e",
            "end run",
            message,
            DEFAULT_CONFIG.window_name,
        ],
        check=False,
    )


def main() -> None:
    multiprocessing.freeze_support()
    _apply_bundled_paths()
    try:
        from src.main import main as run_main

        run_main()
    except Exception as exc:
        _show_startup_error(exc)
        if not getattr(sys, "frozen", False):
            raise


if __name__ == "__main__":
    main()
