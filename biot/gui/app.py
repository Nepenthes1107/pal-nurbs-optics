from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    """Start the BIOT PySide6 GUI."""

    try:
        from PySide6.QtWidgets import QApplication
    except ImportError:
        print("缺少 GUI 依赖 PySide6。请先运行: pip install -r requirements-gui.txt")
        return 2

    from .main_window import MainWindow

    app = QApplication(argv or sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()
