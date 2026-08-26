from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QPlainTextEdit,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from biot.domain import SystemConfig
from biot.services import load_system_config, load_system_from_excel, save_system_config, summarize_system, validate_system


class SystemTab(QWidget):
    system_config_changed = Signal(object)
    log_message = Signal(str)

    def __init__(self, default_excel: Path | None = None) -> None:
        super().__init__()
        repo_default = Path.cwd() / "eye_image_glass.xlsx"
        self.default_excel = default_excel or repo_default
        self.config: SystemConfig | None = None

        self.excel_edit = QLineEdit(str(self.default_excel))
        browse_button = QPushButton("浏览")
        browse_button.clicked.connect(self.browse_excel)
        load_button = QPushButton("加载 Excel")
        load_button.clicked.connect(self.load_excel)
        save_json_button = QPushButton("保存配置")
        save_json_button.clicked.connect(self.save_config_json)
        load_json_button = QPushButton("加载配置")
        load_json_button.clicked.connect(self.load_config_json)

        path_row = QHBoxLayout()
        path_row.addWidget(self.excel_edit, 1)
        path_row.addWidget(browse_button)
        path_row.addWidget(load_button)

        json_row = QHBoxLayout()
        json_row.addWidget(save_json_button)
        json_row.addWidget(load_json_button)
        json_row.addStretch(1)

        self.np_spin = QSpinBox()
        self.np_spin.setRange(8, 4096)
        self.np_spin.setValue(256)
        self.ni_spin = QSpinBox()
        self.ni_spin.setRange(8, 4096)
        self.ni_spin.setValue(512)

        sampling_form = QFormLayout()
        sampling_form.addRow("pupil 采样数", self.np_spin)
        sampling_form.addRow("像面采样数", self.ni_spin)

        self.summary = QPlainTextEdit()
        self.summary.setReadOnly(True)
        self.issues = QPlainTextEdit()
        self.issues.setReadOnly(True)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Excel 配置文件"))
        layout.addLayout(path_row)
        layout.addLayout(sampling_form)
        layout.addLayout(json_row)
        layout.addWidget(QLabel("系统摘要"))
        layout.addWidget(self.summary, 2)
        layout.addWidget(QLabel("校验结果"))
        layout.addWidget(self.issues, 1)

    def load_default_if_available(self) -> None:
        if self.default_excel.exists():
            self.load_excel()

    def browse_excel(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择 Excel 配置", str(Path.cwd()), "Excel (*.xlsx)")
        if path:
            self.excel_edit.setText(path)

    def load_excel(self) -> None:
        try:
            self.config = load_system_from_excel(
                Path(self.excel_edit.text()),
                np_pupil=self.np_spin.value(),
                ni_image=self.ni_spin.value(),
            )
            self._render_config()
            self.system_config_changed.emit(self.config)
            self.log_message.emit(f"已加载 Excel 配置: {self.config.excel_path}")
        except Exception as exc:
            self.issues.setPlainText(str(exc))
            self.log_message.emit(f"加载 Excel 配置失败: {exc}")

    def save_config_json(self) -> None:
        if self.config is None:
            self.load_excel()
        if self.config is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "保存系统配置", str(Path.cwd() / "system_config.json"), "JSON (*.json)")
        if path:
            save_system_config(self.config, Path(path))
            self.log_message.emit(f"已保存系统配置: {path}")

    def load_config_json(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "加载系统配置", str(Path.cwd()), "JSON (*.json)")
        if not path:
            return
        try:
            self.config = load_system_config(Path(path))
            self.excel_edit.setText(str(self.config.excel_path))
            self.np_spin.setValue(int(self.config.np_pupil))
            self.ni_spin.setValue(int(self.config.ni_image))
            self._render_config()
            self.system_config_changed.emit(self.config)
            self.log_message.emit(f"已加载系统配置 JSON: {path}")
        except Exception as exc:
            self.issues.setPlainText(str(exc))
            self.log_message.emit(f"加载系统配置 JSON 失败: {exc}")

    def _render_config(self) -> None:
        if self.config is None:
            return
        summary_lines = [f"{key}: {value}" for key, value in summarize_system(self.config).items()]
        self.summary.setPlainText("\n".join(summary_lines))
        issues = validate_system(self.config)
        self.issues.setPlainText("\n".join(issues) if issues else "未发现配置问题")
