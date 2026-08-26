from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from biot.domain import SingleFieldRequest, SystemConfig
from biot.services.visualization_utils import default_chart_path

from ..widgets.plot_canvas import PlotCanvas
from ..workers.single_field_worker import SingleFieldWorker


class SingleFieldTab(QWidget):
    log_message = Signal(str)
    status_message = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.system_config: SystemConfig | None = None
        self.worker: SingleFieldWorker | None = None

        self.object_distance_edit = QLineEdit("inf")
        self.field_x_edit = QLineEdit("0")
        self.field_y_edit = QLineEdit("0")
        self.cutoff_edit = QLineEdit("100")
        self.output_edit = QLineEdit(str(Path.cwd() / "results" / "gui_single_field"))
        browse_output = QPushButton("浏览")
        browse_output.clicked.connect(self.browse_output)
        self.with_mtf_check = QCheckBox("输出 MTF")
        self.with_mtf_check.setChecked(True)
        self.with_chart_check = QCheckBox("视标卷积")
        self.with_chart_check.setChecked(True)
        self.chart_edit = QLineEdit(str(default_chart_path()))
        browse_chart = QPushButton("浏览")
        browse_chart.clicked.connect(self.browse_chart)

        output_row = QHBoxLayout()
        output_row.addWidget(self.output_edit, 1)
        output_row.addWidget(browse_output)
        chart_row = QHBoxLayout()
        chart_row.addWidget(self.chart_edit, 1)
        chart_row.addWidget(browse_chart)

        form = QFormLayout()
        form.addRow("物距 (mm 或 inf)", self.object_distance_edit)
        form.addRow("视场角 X (degree)", self.field_x_edit)
        form.addRow("视场角 Y (degree)", self.field_y_edit)
        form.addRow("MTF cutoff (cycles/mm)", self.cutoff_edit)
        form.addRow("输出目录", output_row)
        form.addRow("", self.with_mtf_check)
        form.addRow("", self.with_chart_check)
        form.addRow("视标文件", chart_row)

        self.run_button = QPushButton("运行")
        self.run_button.clicked.connect(self.run_single_field)
        self.cancel_button = QPushButton("取消")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel_current)
        button_row = QHBoxLayout()
        button_row.addWidget(self.run_button)
        button_row.addWidget(self.cancel_button)
        button_row.addStretch(1)

        self.psf_plot = PlotCanvas()
        self.mtf_plot = PlotCanvas()
        self.chart_plot = PlotCanvas()
        self.metrics_table = QTableWidget(0, 2)
        self.metrics_table.setHorizontalHeaderLabels(["指标", "值"])
        self.metrics_table.horizontalHeader().setStretchLastSection(True)

        left = QWidget()
        left.setFixedWidth(360)
        left_layout = QVBoxLayout(left)
        left_layout.addLayout(form)
        left_layout.addLayout(button_row)
        left_layout.addStretch(1)

        right_tabs = QTabWidget()
        right_tabs.addTab(self.psf_plot, "PSF")
        right_tabs.addTab(self.mtf_plot, "MTF曲线")
        right_tabs.addTab(self.chart_plot, "视标卷积")
        right_tabs.addTab(self.metrics_table, "汇总指标")

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
        if config.object_distance_mm == float("inf"):
            self.object_distance_edit.setText("inf")
        else:
            self.object_distance_edit.setText(f"{config.object_distance_mm:g}")

    def browse_output(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择输出目录", self.output_edit.text())
        if path:
            self.output_edit.setText(path)

    def browse_chart(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择视标 XLSX", str(Path.cwd()), "Excel (*.xlsx)")
        if path:
            self.chart_edit.setText(path)

    def run_single_field(self) -> None:
        if self.system_config is None:
            self.log_message.emit("请先在系统配置 tab 加载 Excel 配置。")
            return

        try:
            field_x = float(self.field_x_edit.text())
            field_y = float(self.field_y_edit.text())
            cutoff = float(self.cutoff_edit.text())
            output_dir = Path(self.output_edit.text())
            output_dir.mkdir(parents=True, exist_ok=True)
            obj_text = self.object_distance_edit.text().strip().lower()
            object_distance = float("inf") if obj_text in {"inf", "infinity"} else float(obj_text)
            run_config = replace(
                self.system_config,
                object_distance_mm=object_distance,
                write_temp_excel=True,
            )
            req = SingleFieldRequest(
                system=run_config,
                field_x_deg=field_x,
                field_y_deg=field_y,
                cutoff_cyc_per_mm=cutoff,
                with_mtf=self.with_mtf_check.isChecked(),
                with_chart_convolution=self.with_chart_check.isChecked(),
                chart_path=Path(self.chart_edit.text()) if self.with_chart_check.isChecked() else None,
                output_dir=output_dir,
            )
        except Exception as exc:
            self.log_message.emit(f"参数错误: {exc}")
            return

        self.worker = SingleFieldWorker(req)
        self.worker.progress_message.connect(self._on_progress)
        self.worker.finished_result.connect(self._on_result)
        self.worker.log_message.connect(self.log_message)
        self.run_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.status_message.emit("单视角计算运行中")
        self.worker.start()

    def cancel_current(self) -> None:
        if self.worker is not None:
            self.worker.cancel()
            self.status_message.emit("正在取消")

    def _on_progress(self, message: str) -> None:
        self.status_message.emit(message)
        self.log_message.emit(message)

    def _on_result(self, result: object) -> None:
        self.run_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.worker = None

        status = getattr(result, "status", None)
        self.status_message.emit(f"单视角计算结束: {status.value if status else 'unknown'}")
        if getattr(result, "error", None):
            self.log_message.emit(str(result.error))

        psf = getattr(result, "psf", None)
        if isinstance(psf, np.ndarray):
            self.psf_plot.show_psf(psf)
        mtf_curve = getattr(result, "mtf_curve", None)
        if isinstance(mtf_curve, np.ndarray):
            self.mtf_plot.show_mtf_curve(mtf_curve)
        chart_image = getattr(result, "chart_image", None)
        if isinstance(chart_image, np.ndarray):
            self.chart_plot.show_image(chart_image, "Chart convolution", cmap="gray")
        self._render_metrics(getattr(result, "metrics", {}) or {})

    def _render_metrics(self, metrics: dict) -> None:
        self.metrics_table.setRowCount(len(metrics))
        for row, (key, value) in enumerate(metrics.items()):
            self.metrics_table.setItem(row, 0, QTableWidgetItem(str(key)))
            self.metrics_table.setItem(row, 1, QTableWidgetItem(str(value)))
