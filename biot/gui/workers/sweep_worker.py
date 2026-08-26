from __future__ import annotations

from PySide6.QtCore import QThread, Signal

from biot.domain import CancelToken, ProgressEvent, SweepRequest
from biot.services import compute_sweep


class SweepWorker(QThread):
    progress_event = Signal(object)
    progress_message = Signal(str)
    log_message = Signal(str)
    finished_result = Signal(object)

    def __init__(self, request: SweepRequest) -> None:
        super().__init__()
        self.request = request
        self.cancel_token = CancelToken()

    def cancel(self) -> None:
        self.cancel_token.cancel()

    def run(self) -> None:
        def on_progress(event: ProgressEvent) -> None:
            self.progress_event.emit(event)
            self.progress_message.emit(f"{event.phase}: {event.current}/{event.total} {event.message}".strip())

        result = compute_sweep(self.request, progress=on_progress, cancel=self.cancel_token)
        self.finished_result.emit(result)
