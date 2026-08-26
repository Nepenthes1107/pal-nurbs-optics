from __future__ import annotations

from PySide6.QtCore import QThread, Signal

from biot.domain import CancelToken, DistortionCurveRequest, DistortionGridRequest, ProgressEvent
from biot.services import compute_distortion_curve, compute_distortion_grid


class DistortionWorker(QThread):
    progress_message = Signal(str)
    log_message = Signal(str)
    finished_result = Signal(object)

    def __init__(self, request: DistortionCurveRequest | DistortionGridRequest) -> None:
        super().__init__()
        self.request = request
        self.cancel_token = CancelToken()

    def cancel(self) -> None:
        self.cancel_token.cancel()

    def run(self) -> None:
        def on_progress(event: ProgressEvent) -> None:
            self.progress_message.emit(f"{event.phase}: {event.current}/{event.total} {event.message}".strip())

        if isinstance(self.request, DistortionGridRequest):
            result = compute_distortion_grid(self.request, progress=on_progress, cancel=self.cancel_token)
        else:
            result = compute_distortion_curve(self.request, progress=on_progress, cancel=self.cancel_token)
        self.finished_result.emit(result)
