import platform
import threading
import time
from typing import Optional

import cv2

try:
    from .config import AppConfig
except ImportError:
    from config import AppConfig


class CameraSource:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.capture = None
        self._reader_thread: Optional[threading.Thread] = None
        self._stop_requested = threading.Event()
        self._frame_lock = threading.Lock()
        self._frame_ready = threading.Event()
        self._latest_frame = None

    def open(self) -> None:
        deadline = time.monotonic() + 6.0
        while time.monotonic() < deadline:
            if self.capture is not None:
                self.capture.release()
                self.capture = None

            self.capture = self._open_capture()
            if self.capture.isOpened():
                self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.frame_width)
                self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.frame_height)
                self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self._start_reader()
                return

            time.sleep(0.25)

        raise RuntimeError("Camera could not be opened.")

    def _open_capture(self):
        backends = []
        if platform.system() == "Darwin" and hasattr(cv2, "CAP_AVFOUNDATION"):
            backends.append(cv2.CAP_AVFOUNDATION)
        backends.append(None)

        for backend in backends:
            capture = (
                cv2.VideoCapture(self.config.camera_index, backend)
                if backend is not None
                else cv2.VideoCapture(self.config.camera_index)
            )
            if capture.isOpened():
                return capture
            capture.release()

        return cv2.VideoCapture()

    def read(self):
        if self.capture is None:
            return None
        if not self._frame_ready.wait(timeout=1.0):
            return None
        with self._frame_lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy()

    def release(self) -> None:
        self._stop_reader()
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def _start_reader(self) -> None:
        if self._reader_thread is not None and self._reader_thread.is_alive():
            return
        self._stop_requested.clear()
        self._frame_ready.clear()
        self._reader_thread = threading.Thread(
            target=self._reader_loop,
            name="camera-reader",
            daemon=True,
        )
        self._reader_thread.start()

    def _stop_reader(self) -> None:
        self._stop_requested.set()
        if self._reader_thread is not None and self._reader_thread.is_alive():
            self._reader_thread.join(timeout=1.0)
        self._reader_thread = None
        self._frame_ready.clear()
        with self._frame_lock:
            self._latest_frame = None

    def _reader_loop(self) -> None:
        while not self._stop_requested.is_set():
            if self.capture is None:
                time.sleep(0.01)
                continue

            ok, frame = self.capture.read()
            if not ok:
                time.sleep(0.01)
                continue

            with self._frame_lock:
                self._latest_frame = frame
            self._frame_ready.set()
