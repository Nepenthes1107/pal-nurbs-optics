from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import QDockWidget, QMainWindow, QPlainTextEdit, QStatusBar, QTabWidget

from biot.domain import SystemConfig

from .tabs.distortion_tab import DistortionTab
from .tabs.export_tab import ExportTab
from .tabs.power_tab import PowerTab
from .tabs.single_field_tab import SingleFieldTab
from .tabs.sweep_tab import SweepTab
from .tabs.system_tab import SystemTab


class MainWindow(QMainWindow):
    """Main BIOT GUI window.

    The GUI layer depends only on domain models and services. Optical core
    modules remain behind the service layer.
    """

    def __init__(self, default_excel: Path | None = None) -> None:
        super().__init__()
        self.setWindowTitle("BIOT 光学仿真")
        self.resize(1280, 820)

        self._system_config: SystemConfig | None = None

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        log_dock = QDockWidget("日志", self)
        log_dock.setObjectName("logDock")
        log_dock.setWidget(self.log_view)
        self.addDockWidget(Qt.BottomDockWidgetArea, log_dock)
        self.resizeDocks([log_dock], [120], Qt.Vertical)

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("就绪")

        self.system_tab = SystemTab(default_excel=default_excel)
        self.single_field_tab = SingleFieldTab()
        self.sweep_tab = SweepTab()
        self.power_tab = PowerTab()
        self.distortion_tab = DistortionTab()
        self.export_tab = ExportTab()

        self.tabs.addTab(self.system_tab, "系统配置")
        self.tabs.addTab(self.single_field_tab, "单视角分析")
        self.tabs.addTab(self.sweep_tab, "范围扫描")
        self.tabs.addTab(self.power_tab, "光焦度像散")
        self.tabs.addTab(self.distortion_tab, "畸变分析")
        self.tabs.addTab(self.export_tab, "结果导出")

        self.system_tab.system_config_changed.connect(self._on_system_config_changed)
        self.system_tab.log_message.connect(self.append_log)
        self.single_field_tab.log_message.connect(self.append_log)
        self.single_field_tab.status_message.connect(self.status.showMessage)
        self.sweep_tab.log_message.connect(self.append_log)
        self.sweep_tab.status_message.connect(self.status.showMessage)
        self.power_tab.log_message.connect(self.append_log)
        self.power_tab.status_message.connect(self.status.showMessage)
        self.distortion_tab.log_message.connect(self.append_log)
        self.distortion_tab.status_message.connect(self.status.showMessage)
        self.export_tab.log_message.connect(self.append_log)
        self.export_tab.status_message.connect(self.status.showMessage)

        self.system_tab.load_default_if_available()

    @Slot(object)
    def _on_system_config_changed(self, config: SystemConfig) -> None:
        self._system_config = config
        self.single_field_tab.set_system_config(config)
        self.sweep_tab.set_system_config(config)
        self.power_tab.set_system_config(config)
        self.distortion_tab.set_system_config(config)
        self.status.showMessage(f"已加载系统配置: {config.excel_path}")

    @Slot(str)
    def append_log(self, message: str) -> None:
        self.log_view.appendPlainText(message)
