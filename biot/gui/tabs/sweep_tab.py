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
    QProgressBar,
    QPushButton,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from biot.domain import SweepRequest, SystemConfig
from biot.services.visualization_utils import default_chart_path

from ..widgets.plot_canvas import PlotCanvas
from ..workers.sweep_worker import SweepWorker


class SweepTab(QWidget):
    log_message = Signal(str)
    status_message = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.system_config: SystemConfig | None = None
        self.worker: SweepWorker | None = None

        self.object_distance_edit = QLineEdit("inf")
        self.x_min_edit = QLineEdit("-10")
        self.x_max_edit = QLineEdit("10")
        self.x_step_edit = QLineEdit("5")
        self.y_min_edit = QLineEdit("-10")
        self.y_max_edit = QLineEdit("10")
        self.y_step_edit = QLineEdit("5")
        self.cutoff_edit = QLineEdit("100")
        self.output_edit = QLineEdit(str(Path.cwd() / "results" / "gui_sweep"))
        browse_output = QPushButton("浏览")
        browse_output.clicked.connect(self.browse_output)
        self.with_mtf_check = QCheckBox("输出每点 MTF")
        self.with_chart_check = QCheckBox("视标卷积拼接")
        self.with_chart_check.setChecked(True)
        self.with_mtf_grid_check = QCheckBox("二维 MTF 指标网格")
        self.chart_edit = QLineEdit(str(default_chart_path()))
        browse_chart = QPushButton("浏览")
        browse_chart.clicked.connect(self.browse_chart)
        self.use_cache_check = QCheckBox("使用缓存")
        self.use_cache_check.setChecked(True)

        output_row = QHBoxLayout()
        output_row.addWidget(self.output_edit, 1)
        output_row.addWidget(browse_output)
        chart_row = QHBoxLayout()
        chart_row.addWidget(self.chart_edit, 1)
        chart_row.addWidget(browse_chart)

        form = QFormLayout()
        form.addRow("物距 (mm 或 inf)", self.object_distance_edit)
        form.addRow("X min (degree)", self.x_min_edit)
        form.addRow("X max (degree)", self.x_max_edit)
        form.addRow("X step (degree)", self.x_step_edit)
        form.addRow("Y min (degree)", self.y_min_edit)
        form.addRow("Y max (degree)", self.y_max_edit)
        form.addRow("Y step (degree)", self.y_step_edit)
        form.addRow("MTF cutoff (cycles/mm)", self.cutoff_edit)
        form.addRow("输出目录", output_row)
        form.addRow("", self.with_mtf_check)
        form.addRow("", self.with_chart_check)
        form.addRow("", self.with_mtf_grid_check)
        form.addRow("视标文件", chart_row)
        form.addRow("", self.use_cache_check)

        self.run_button = QPushButton("运行扫描")
        self.run_button.clicked.connect(self.run_sweep)
        self.cancel_button = QPushButton("取消")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel_current)
        button_row = QHBoxLayout()
        button_row.addWidget(self.run_button)
        button_row.addWidget(self.cancel_button)
        button_row.addStretch(1)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)

        self.psf_plot = PlotCanvas()
        self.chart_plot = PlotCanvas()
        self.mtf_grid_plot = PlotCanvas()
        self.metrics_table = QTableWidget(0, 2)
        self.metrics_table.setHorizontalHeaderLabels(["指标", "值"])
        self.metrics_table.horizontalHeader().setStretchLastSection(True)

        left = QWidget()
        left.setFixedWidth(360)
        left_layout = QVBoxLayout(left)
        left_layout.addLayout(form)
        left_layout.addLayout(button_row)
        left_layout.addWidget(self.progress)
        left_layout.addStretch(1)

        right_tabs = QTabWidget()
        right_tabs.addTab(self.psf_plot, "PSF拼接")
        right_tabs.addTab(self.chart_plot, "视标卷积")
        right_tabs.addTab(self.mtf_grid_plot, "MTF网格")
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

    def run_sweep(self) -> None:
        if self.system_config is None:
            self.log_message.emit("请先在系统配置 tab 加载 Excel 配置。")
            return

        try:
            obj_text = self.object_distance_edit.text().strip().lower()
            object_distance = float("inf") if obj_text in {"inf", "infinity"} else float(obj_text)
            output_dir = Path(self.output_edit.text())
            output_dir.mkdir(parents=True, exist_ok=True)
            run_config = replace(
                self.system_config,
                object_distance_mm=object_distance,
                write_temp_excel=True,
            )
            req = SweepRequest(
                system=run_config,
                field_x_min_deg=float(self.x_min_edit.text()),
                field_x_max_deg=float(self.x_max_edit.text()),
                field_x_step_deg=float(self.x_step_edit.text()),
                field_y_min_deg=float(self.y_min_edit.text()),
                field_y_max_deg=float(self.y_max_edit.text()),
                field_y_step_deg=float(self.y_step_edit.text()),
                cutoff_cyc_per_mm=float(self.cutoff_edit.text()),
                with_mtf=self.with_mtf_check.isChecked(),
                with_chart_stitch=self.with_chart_check.isChecked(),
                with_mtf_grid=self.with_mtf_grid_check.isChecked(),
                chart_path=Path(self.chart_edit.text()) if self.with_chart_check.isChecked() else None,
                output_dir=output_dir,
                use_cache=self.use_cache_check.isChecked(),
            )
        except Exception as exc:
            self.log_message.emit(f"参数错误: {exc}")
            return

        self.progress.setValue(0)
        self.worker = SweepWorker(req)
        self.worker.progress_event.connect(self._on_progress_event)
        self.worker.progress_message.connect(self._on_progress_message)
        self.worker.finished_result.connect(self._on_result)
        self.worker.log_message.connect(self.log_message)
        self.run_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.status_message.emit("范围扫描运行中")
        self.worker.start()

    def cancel_current(self) -> None:
        if self.worker is not None:
            self.worker.cancel()
            self.status_message.emit("正在取消扫描")

    def _on_progress_event(self, event: object) -> None:
        total = max(int(getattr(event, "total", 0)), 1)
        current = int(getattr(event, "current", 0))
        self.progress.setValue(int(round(current / total * 100)))

    def _on_progress_message(self, message: str) -> None:
        self.status_message.emit(message)
        self.log_message.emit(message)

    def _on_result(self, result: object) -> None:
        self.run_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.worker = None
        self.progress.setValue(100)

        status = getattr(result, "status", None)
        self.status_message.emit(f"范围扫描结束: {status.value if status else 'unknown'}")
        if getattr(result, "error", None):
            self.log_message.emit(str(result.error))

        stitched = getattr(result, "stitched_psf", None)
        field_x_values, field_y_values = self._field_values_from_result(result)
        if isinstance(stitched, np.ndarray) and stitched.size:
            self.psf_plot.show_field_stitch(stitched, field_x_values, field_y_values, kind="psf")
        stitched_chart = getattr(result, "stitched_chart", None)
        if isinstance(stitched_chart, np.ndarray) and stitched_chart.size:
            self.chart_plot.show_field_stitch(stitched_chart, field_x_values, field_y_values, kind="chart")
        mtf_grid = getattr(result, "mtf_grid", None)
        if isinstance(mtf_grid, np.ndarray) and mtf_grid.size:
            self.mtf_grid_plot.show_mtf_value_grid(mtf_grid, field_x_values, field_y_values)
        self._render_metrics(getattr(result, "metrics", {}) or {})

    def _field_values_from_result(self, result: object) -> tuple[np.ndarray, np.ndarray]:
        field_grid = getattr(result, "field_grid", None) or []
        if field_grid:
            arr = np.asarray(field_grid, dtype=float)
            return np.unique(arr[:, 0]), np.unique(arr[:, 1])
        try:
            return (
                np.arange(float(self.x_min_edit.text()), float(self.x_max_edit.text()) + 0.5 * float(self.x_step_edit.text()), float(self.x_step_edit.text())),
                np.arange(float(self.y_min_edit.text()), float(self.y_max_edit.text()) + 0.5 * float(self.y_step_edit.text()), float(self.y_step_edit.text())),
            )
        except Exception:
            return np.asarray([], dtype=float), np.asarray([], dtype=float)

    def _render_metrics(self, metrics: dict) -> None:
        self.metrics_table.setRowCount(len(metrics))
        for row, (key, value) in enumerate(metrics.items()):
            self.metrics_table.setItem(row, 0, QTableWidgetItem(str(key)))
            self.metrics_table.setItem(row, 1, QTableWidgetItem(str(value)))
