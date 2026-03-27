import platform

import cv2

try:
    from .config import AppConfig
except ImportError:
    from config import AppConfig


class CameraSource:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.capture = None

    def open(self) -> None:
        if self.capture is None:
            self.capture = self._open_capture()

        if not self.capture.isOpened():
            raise RuntimeError("Camera could not be opened.")

        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.frame_width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.frame_height)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

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
        ok, frame = self.capture.read()
        if not ok:
            return None
        return frame

    def release(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None
