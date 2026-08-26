from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QAbstractItemView,
    QLabel,
    QLineEdit,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from biot.services.result_service import diff_results, export_result, list_results, load_result_manifest


class ExportTab(QWidget):
    log_message = Signal(str)
    status_message = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.results: list[dict] = []

        self.root_edit = QLineEdit(str(Path.cwd() / "results"))
        browse_root = QPushButton("浏览")
        browse_root.clicked.connect(self.browse_root)
        scan_button = QPushButton("扫描结果")
        scan_button.clicked.connect(self.scan_results)
        export_button = QPushButton("导出所选")
        export_button.clicked.connect(self.export_selected)
        compare_button = QPushButton("对比两项")
        compare_button.clicked.connect(self.compare_selected)

        root_row = QHBoxLayout()
        root_row.addWidget(QLabel("结果根目录"))
        root_row.addWidget(self.root_edit, 1)
        root_row.addWidget(browse_root)

        action_row = QHBoxLayout()
        action_row.addWidget(scan_button)
        action_row.addWidget(compare_button)
        action_row.addWidget(export_button)
        action_row.addStretch(1)

        self.results_table = QTableWidget(0, 5)
        self.results_table.setHorizontalHeaderLabels(["类型", "状态", "完成时间", "输出目录", "manifest"])
        self.results_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.results_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.results_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.results_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.results_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.Stretch)
        self.results_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.results_table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.results_table.itemSelectionChanged.connect(self.render_selected)

        left = QWidget()
        left.setMinimumWidth(520)
        left_layout = QVBoxLayout(left)
        left_layout.addLayout(root_row)
        left_layout.addLayout(action_row)
        left_layout.addWidget(self.results_table, 1)

        self.metrics_table = self._new_table(["指标", "值"])
        self.request_table = self._new_table(["参数", "值"])
        self.artifacts_table = self._new_table(["产物", "路径"])
        self.diff_table = self._new_table(["分组", "键", "左侧", "右侧", "差值"])

        right_tabs = QTabWidget()
        right_tabs.addTab(self.metrics_table, "指标")
        right_tabs.addTab(self.request_table, "参数")
        right_tabs.addTab(self.artifacts_table, "产物")
        right_tabs.addTab(self.diff_table, "对比")

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(right_tabs)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)

        layout = QVBoxLayout(self)
        layout.addWidget(splitter)

    @staticmethod
    def _new_table(headers: list[str]) -> QTableWidget:
        table = QTableWidget(0, len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.horizontalHeader().setStretchLastSection(True)
        table.setWordWrap(False)
        return table

    def browse_root(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择结果根目录", self.root_edit.text())
        if path:
            self.root_edit.setText(path)

    def scan_results(self) -> None:
        root = Path(self.root_edit.text())
        self.results = list_results(root)
        self.results_table.setRowCount(len(self.results))
        for row, item in enumerate(self.results):
            values = [
                item.get("result_type", ""),
                item.get("status", ""),
                item.get("finished_at", ""),
                item.get("output_dir", ""),
                item.get("manifest_path", ""),
            ]
            for col, value in enumerate(values):
                self.results_table.setItem(row, col, QTableWidgetItem(str(value)))
        self.status_message.emit(f"扫描到 {len(self.results)} 个结果")
        self.log_message.emit(f"结果扫描完成: {root} ({len(self.results)} 个 manifest)")

    def _selected_indices(self) -> list[int]:
        rows = self.results_table.selectionModel().selectedRows()
        return sorted({row.row() for row in rows})

    def _selected_manifest_paths(self) -> list[Path]:
        paths: list[Path] = []
        for index in self._selected_indices():
            if 0 <= index < len(self.results):
                paths.append(Path(self.results[index]["manifest_path"]))
        return paths

    def render_selected(self) -> None:
        paths = self._selected_manifest_paths()
        if not paths:
            return
        manifest = load_result_manifest(paths[0])
        self._render_key_value(self.metrics_table, manifest.get("metrics", {}) or {})
        self._render_key_value(self.request_table, manifest.get("request_snapshot", {}) or {})
        self._render_key_value(self.artifacts_table, manifest.get("artifacts", {}) or {})

    def compare_selected(self) -> None:
        paths = self._selected_manifest_paths()
        if len(paths) != 2:
            self.log_message.emit("请选择两个结果后再对比。")
            return
        diff = diff_results(paths[0], paths[1])
        rows: list[list[object]] = []
        for section, section_rows in diff.items():
            for row in section_rows:
                rows.append(
                    [
                        section,
                        row.get("key", ""),
                        row.get("left", ""),
                        row.get("right", ""),
                        row.get("absolute_delta", ""),
                    ]
                )
        self.diff_table.setRowCount(len(rows))
        for row_index, row_values in enumerate(rows):
            for col, value in enumerate(row_values):
                self.diff_table.setItem(row_index, col, QTableWidgetItem(str(value)))
        self.status_message.emit(f"对比完成: {len(rows)} 项差异")
        self.log_message.emit(f"结果对比完成: {paths[0]} <-> {paths[1]}")

    def export_selected(self) -> None:
        paths = self._selected_manifest_paths()
        if len(paths) != 1:
            self.log_message.emit("请选择一个结果后再导出。")
            return
        destination = QFileDialog.getExistingDirectory(self, "选择导出目录", str(Path.cwd() / "results"))
        if not destination:
            return
        target = export_result(paths[0], Path(destination), include_artifacts=True)
        self.status_message.emit(f"已导出到 {target}")
        self.log_message.emit(f"结果导出完成: {target}")

    @staticmethod
    def _render_key_value(table: QTableWidget, values: dict) -> None:
        rows: list[tuple[str, object]] = []

        def append(prefix: str, value: object) -> None:
            if isinstance(value, dict):
                for key, child in value.items():
                    append(f"{prefix}.{key}" if prefix else str(key), child)
            else:
                rows.append((prefix, value))

        append("", values)
        table.setRowCount(len(rows))
        for row, (key, value) in enumerate(rows):
            table.setItem(row, 0, QTableWidgetItem(str(key)))
            table.setItem(row, 1, QTableWidgetItem(str(value)))
