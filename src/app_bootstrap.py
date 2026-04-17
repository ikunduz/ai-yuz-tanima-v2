import ctypes
import multiprocessing
import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Optional

from .config import DEFAULT_CONFIG


APP_LOG_DIR_NAME = "AIYuzTanima"


def _apply_bundled_paths() -> None:
    bundle_root = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
    bundled_face_model = bundle_root / "models" / "face_landmarker.task"
    bundled_mivolo_cache = bundle_root / "mivolo_v2"

    if bundled_face_model.exists():
        DEFAULT_CONFIG.model_path = str(bundled_face_model)
    if bundled_mivolo_cache.exists():
        DEFAULT_CONFIG.age_mivolo_cache_dir = str(bundled_mivolo_cache)


def _platform_log_dir() -> Path:
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Logs" / APP_LOG_DIR_NAME
    if os.name == "nt":
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            return Path(local_app_data) / APP_LOG_DIR_NAME / "Logs"
        return Path.home() / "AppData" / "Local" / APP_LOG_DIR_NAME / "Logs"

    xdg_state_home = os.environ.get("XDG_STATE_HOME")
    if xdg_state_home:
        return Path(xdg_state_home) / APP_LOG_DIR_NAME
    return Path.home() / ".local" / "state" / APP_LOG_DIR_NAME


def _show_message_box(message: str, title: str) -> None:
    if os.name == "nt":
        try:
            ctypes.windll.user32.MessageBoxW(None, message, title, 0x10)
            return
        except Exception:
            pass

    if sys.platform == "darwin":
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
                title,
            ],
            check=False,
        )
        return

    print(f"{title}\n\n{message}", file=sys.stderr)


def _show_startup_error(exc: Exception, window_name: str) -> None:
    if not getattr(sys, "frozen", False):
        raise exc

    log_dir = _platform_log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{window_name}.log"
    log_path.write_text(traceback.format_exc(), encoding="utf-8")
    message = (
        "Uygulama acilirken hata olustu.\n\n"
        f"{exc}\n\n"
        f"Log: {log_path}"
    )
    _show_message_box(message, window_name)


def run_app(window_name: str, age_bias_years: Optional[float] = None) -> None:
    DEFAULT_CONFIG.window_name = window_name
    if age_bias_years is not None:
        DEFAULT_CONFIG.age_bias_years = age_bias_years

    multiprocessing.freeze_support()
    _apply_bundled_paths()
    try:
        from src.main import main as run_main

        run_main()
    except Exception as exc:
        _show_startup_error(exc, window_name)
        if not getattr(sys, "frozen", False):
            raise
