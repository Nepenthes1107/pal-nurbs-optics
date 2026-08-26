from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from biot.domain import PowerAstigmatismRequest, SystemConfig

from ..widgets.plot_canvas import PlotCanvas
from ..workers.power_worker import PowerWorker


class PowerTab(QWidget):
    log_message = Signal(str)
    status_message = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.system_config: SystemConfig | None = None
        self.worker: PowerWorker | None = None

        self.fov_edit = QLineEdit("50")
        self.field_num_edit = QLineEdit("51")
        self.lens_fov_edit = QLineEdit("50")
        self.aperture_edit = QLineEdit("2")
        self.wavelength_edit = QLineEdit("555")
        self.diff_aperture_edit = QLineEdit("0.01")
        self.target_power_edit = QLineEdit("0")
        self.averfang_crib_edit = QLineEdit("80")
        self.output_edit = QLineEdit(str(Path.cwd() / "results" / "gui_power_astigmatism"))
        browse_output = QPushButton("浏览")
        browse_output.clicked.connect(self.browse_output)

        output_row = QHBoxLayout()
        output_row.addWidget(self.output_edit, 1)
        output_row.addWidget(browse_output)

        form = QFormLayout()
        form.addRow("FOV 上限 (degree)", self.fov_edit)
        form.addRow("采样点数", self.field_num_edit)
        form.addRow("Lensdata FOV (degree)", self.lens_fov_edit)
        form.addRow("AverFang CRIB diameter (mm)", self.averfang_crib_edit)
        form.addRow("光阑半径 (mm)", self.aperture_edit)
        form.addRow("波长 (nm)", self.wavelength_edit)
        form.addRow("差分孔径 (mm)", self.diff_aperture_edit)
        form.addRow("目标光焦度 (D)", self.target_power_edit)
        form.addRow("输出目录", output_row)

        self.run_button = QPushButton("运行光焦度/像散")
        self.run_button.clicked.connect(self.run_power)
        self.cancel_button = QPushButton("取消")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel_current)
        button_row = QHBoxLayout()
        button_row.addWidget(self.run_button)
        button_row.addWidget(self.cancel_button)
        button_row.addStretch(1)

        self.curve_plot = PlotCanvas()
        self.metrics_table = QTableWidget(0, 2)
        self.metrics_table.setHorizontalHeaderLabels(["指标", "值"])
        self.metrics_table.horizontalHeader().setStretchLastSection(True)
        self.metadata_table = QTableWidget(0, 2)
        self.metadata_table.setHorizontalHeaderLabels(["元数据", "值"])
        self.metadata_table.horizontalHeader().setStretchLastSection(True)

        left = QWidget()
        left.setFixedWidth(360)
        left_layout = QVBoxLayout(left)
        left_layout.addLayout(form)
        left_layout.addLayout(button_row)
        left_layout.addStretch(1)

        right_tabs = QTabWidget()
        right_tabs.addTab(self.curve_plot, "曲线")
        right_tabs.addTab(self.metrics_table, "汇总指标")
        right_tabs.addTab(self.metadata_table, "元数据")

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(right_tabs)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([360, 920])

        layout = QVBoxLayout(self)
        layout.addWidget(splitter)

    def set_system_config(self, config: SystemConfig) -> None:
        self.system_config = config
        self.aperture_edit.setText(f"{config.pupil_radius_mm:g}")
        self.wavelength_edit.setText(f"{config.wavelength_nm:g}")

    def browse_output(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择输出目录", self.output_edit.text())
        if path:
            self.output_edit.setText(path)

    def run_power(self) -> None:
        if self.system_config is None:
            self.log_message.emit("请先在系统配置 tab 加载 Excel 配置。")
            return
        try:
            output_dir = Path(self.output_edit.text())
            output_dir.mkdir(parents=True, exist_ok=True)
            req = PowerAstigmatismRequest(
                system=self.system_config,
                fov_deg=float(self.fov_edit.text()),
                field_num=int(self.field_num_edit.text()),
                lens_fov_deg=float(self.lens_fov_edit.text()),
                aperture_mm=float(self.aperture_edit.text()),
                wavelength_nm=float(self.wavelength_edit.text()),
                differential_aperture_mm=float(self.diff_aperture_edit.text()),
                target_focal_power_d=float(self.target_power_edit.text()),
                averfang_crib_diameter_mm=float(self.averfang_crib_edit.text()),
                output_dir=output_dir,
            )
        except Exception as exc:
            self.log_message.emit(f"参数错误: {exc}")
            return

        self.worker = PowerWorker(req)
        self.worker.progress_message.connect(self._on_progress)
        self.worker.finished_result.connect(self._on_result)
        self.worker.log_message.connect(self.log_message)
        self.run_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.status_message.emit("光焦度/像散计算运行中")
        self.worker.start()

    def cancel_current(self) -> None:
        if self.worker is not None:
            self.worker.cancel()
            self.status_message.emit("正在取消光焦度/像散计算")

    def _on_progress(self, message: str) -> None:
        self.status_message.emit(message)
        self.log_message.emit(message)

    def _on_result(self, result: object) -> None:
        self.run_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.worker = None

        status = getattr(result, "status", None)
        self.status_message.emit(f"光焦度/像散结束: {status.value if status else 'unknown'}")
        if getattr(result, "error", None):
            self.log_message.emit(str(result.error))

        data = getattr(result, "table_data", None)
        columns = getattr(result, "table_columns", []) or []
        if isinstance(data, np.ndarray):
            self.curve_plot.show_power_astigmatism(columns, data)
        self._render_table(self.metrics_table, getattr(result, "metrics", {}) or {})
        self._render_table(self.metadata_table, getattr(result, "metadata", {}) or {})

    @staticmethod
    def _render_table(table: QTableWidget, values: dict) -> None:
        table.setRowCount(len(values))
        for row, (key, value) in enumerate(values.items()):
            table.setItem(row, 0, QTableWidgetItem(str(key)))
            table.setItem(row, 1, QTableWidgetItem(str(value)))
