"""
launcher.py - PyQt6 GUI frontend for CIDeconvolve config.yaml.

Dynamically reads config.yaml and builds a form with appropriate widgets for
each parameter. On "Run" it executes the Docker container in the console that
launched this script.

Usage:
    python launcher.py
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

# Windows taskbar: set AppUserModelID so the taskbar shows our icon, not Python's
if sys.platform == "win32":
    import ctypes
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("ci.w_cideconvolve.bilayers_launcher")

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QIcon
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "config.yaml"
ICON_PATH = SCRIPT_DIR / "gui" / "icon.svg"
LAST_SETTINGS_PATH = SCRIPT_DIR / ".last_launcher_settings.json"


class ToggleSwitch(QCheckBox):
    """Styled toggle switch using a QCheckBox with a stylesheet."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(
            """
            QCheckBox {
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 40px;
                height: 22px;
                border-radius: 11px;
                background-color: #888;
            }
            QCheckBox::indicator:checked {
                background-color: #4CAF50;
            }
            QCheckBox::indicator:unchecked {
                background-color: #888;
            }
            """
        )


class CollapsiblePanel(QWidget):
    """Simple collapsible panel with a checkable header button."""

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self._toggle = QToolButton()
        self._toggle.setText(title)
        self._toggle.setCheckable(True)
        self._toggle.setChecked(False)
        self._toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._toggle.setArrowType(Qt.ArrowType.RightArrow)
        self._toggle.clicked.connect(self._on_toggled)

        self.content = QWidget()
        self.content.setVisible(False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(self._toggle)
        layout.addWidget(self.content)

    def _on_toggled(self, checked: bool):
        self._toggle.setArrowType(
            Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow
        )
        self.content.setVisible(checked)
        window = self.window()
        if window is not None:
            window.adjustSize()


def load_bilayers_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _docker_image_name(config: dict) -> str:
    image = config.get("docker_image", {})
    name = str(image.get("name", "w_cideconvolve"))
    tag = str(image.get("tag", "latest"))
    return name if tag in ("", "latest") else f"{name}:{tag}"


def _cli_items(config: dict) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for source in ("inputs", "outputs", "parameters"):
        for spec in config.get(source, []) or []:
            item = dict(spec)
            item["source"] = source[:-1]
            items.append(item)
    for spec in config.get("exec_function", {}).get("hidden_args", []) or []:
        item = dict(spec)
        item["source"] = "hidden"
        items.append(item)
    return sorted(items, key=lambda item: int(item.get("cli_order", 0)))


def _append_cli_value(cmd: list[str], cli_tag: str, value: Any) -> None:
    if cli_tag in ("", None, "None") or value in (None, ""):
        return
    cmd.extend([str(cli_tag), str(value)])


def build_docker_command(
    config: dict,
    values: dict,
    folders: dict,
    docker_options: dict | None = None,
) -> list[str]:
    """Build the docker run command from config.yaml and current values."""
    docker_options = docker_options or {}
    cmd = ["docker", "run", "--rm"]
    if bool(docker_options.get("use_gpus", True)):
        cmd.extend(["--gpus", "all"])
    cmd.extend([
        "-v", f"{folders['infolder']}:/data/in",
        "-v", f"{folders['outfolder']}:/data/out",
        _docker_image_name(config),
    ])

    for item in _cli_items(config):
        cli_tag = item.get("cli_tag")
        if cli_tag in (None, "None"):
            continue
        source = item.get("source")
        if source == "hidden":
            value = item.get("value")
            append_value = bool(item.get("append_value", True))
        elif source == "input":
            value = item.get("folder_name") or values.get(item["name"])
            append_value = True
        else:
            value = values.get(item["name"], item.get("default"))
            append_value = bool(item.get("append_value", False))

        if item.get("type") == "checkbox" or isinstance(value, bool):
            if append_value:
                _append_cli_value(cmd, str(cli_tag), value)
            elif value:
                cmd.append(str(cli_tag))
            continue
        _append_cli_value(cmd, str(cli_tag), value)

    return cmd


class LauncherWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.config = load_bilayers_config()
        self.widgets: dict[str, QWidget] = {}
        self._build_ui()

    @staticmethod
    def _add_two_column_row(grid: QGridLayout, row_index: int, label: QLabel, widget: QWidget):
        col = 0 if row_index % 2 == 0 else 2
        row = row_index // 2
        grid.addWidget(label, row, col)
        grid.addWidget(widget, row, col + 1)

    def _build_ui(self):
        image = self.config.get("docker_image", {})
        name = str(image.get("name", "W_CIDeconvolve"))
        self.setWindowTitle(f"{name} - Bilayers Launcher")
        self.setWindowIcon(QIcon(str(ICON_PATH)))
        self.setMinimumWidth(920)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(12)

        title = QLabel(f"{name} - Bilayers")
        title.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        layout.addWidget(title)

        desc_label = QLabel("Docker launcher generated from config.yaml.")
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("color: #666; margin-bottom: 8px;")
        layout.addWidget(desc_label)

        folder_group = QGroupBox("Data Folders")
        folder_layout = QFormLayout()
        folder_group.setLayout(folder_layout)

        self.folder_widgets: dict[str, QLineEdit] = {}
        for key, label_text, default_path in [
            ("infolder", "Input folder", str(SCRIPT_DIR / "infolder")),
            ("outfolder", "Output folder", str(SCRIPT_DIR / "outfolder")),
        ]:
            row = QHBoxLayout()
            line = QLineEdit(default_path)
            line.setMinimumWidth(360)
            line.textChanged.connect(self._update_preview)
            browse_btn = QPushButton("Browse...")
            browse_btn.setFixedWidth(80)
            browse_btn.clicked.connect(lambda checked, le=line: self._browse_folder(le))
            row.addWidget(line)
            row.addWidget(browse_btn)
            folder_layout.addRow(label_text + ":", row)
            self.folder_widgets[key] = line

        layout.addWidget(folder_group)

        docker_group = QGroupBox("Docker Runtime")
        docker_layout = QFormLayout()
        docker_group.setLayout(docker_layout)

        self.use_gpus_checkbox = QCheckBox("Expose NVIDIA GPU to container")
        self.use_gpus_checkbox.setChecked(True)
        self.use_gpus_checkbox.setToolTip(
            "When enabled, Docker is run with '--gpus all'. "
            "Turn this off to test CPU fallback behavior."
        )
        self.use_gpus_checkbox.stateChanged.connect(self._update_preview)
        docker_layout.addRow("GPU:", self.use_gpus_checkbox)
        layout.addWidget(docker_group)

        param_group = QGroupBox("Parameters")
        param_layout = QVBoxLayout()
        param_layout.setSpacing(8)
        param_group.setLayout(param_layout)

        main_params = QWidget()
        main_grid = QGridLayout(main_params)
        main_grid.setContentsMargins(0, 0, 0, 0)
        main_grid.setHorizontalSpacing(18)
        main_grid.setVerticalSpacing(6)
        main_grid.setColumnStretch(1, 1)
        main_grid.setColumnStretch(3, 1)

        advanced_panel = CollapsiblePanel("Advanced parameters")
        advanced_grid = QGridLayout(advanced_panel.content)
        advanced_grid.setContentsMargins(18, 0, 0, 0)
        advanced_grid.setHorizontalSpacing(18)
        advanced_grid.setVerticalSpacing(6)
        advanced_grid.setColumnStretch(1, 1)
        advanced_grid.setColumnStretch(3, 1)

        main_count = 0
        advanced_count = 0
        for param in self.config.get("parameters", []):
            widget = self._create_widget(param)
            if widget is None:
                continue
            tooltip = param.get("description", "")
            widget.setToolTip(tooltip)
            label = QLabel(param.get("label", param["name"]))
            label.setToolTip(tooltip)
            label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            if param.get("mode") == "advanced":
                self._add_two_column_row(advanced_grid, advanced_count, label, widget)
                advanced_count += 1
            else:
                self._add_two_column_row(main_grid, main_count, label, widget)
                main_count += 1
            self.widgets[param["name"]] = widget

        param_layout.addWidget(main_params)
        if advanced_count:
            param_layout.addWidget(advanced_panel)
        layout.addWidget(param_group)

        self.cmd_preview = QTextEdit()
        self.cmd_preview.setReadOnly(True)
        self.cmd_preview.setMaximumHeight(163)
        self.cmd_preview.setFont(QFont("Consolas", 9))
        self.cmd_preview.setStyleSheet("background: #1e1e1e; color: #dcdcdc;")
        layout.addWidget(QLabel("Command preview:"))
        layout.addWidget(self.cmd_preview)

        btn_layout = QHBoxLayout()
        restore_btn = QPushButton("Restore Last Settings")
        restore_btn.setStyleSheet("padding: 8px 16px;")
        restore_btn.setToolTip("Restore parameter values from the previous run")
        restore_btn.setEnabled(LAST_SETTINGS_PATH.exists())
        restore_btn.clicked.connect(self._on_restore)
        btn_layout.addWidget(restore_btn)

        load_btn = QPushButton("Load Settings")
        load_btn.setStyleSheet("padding: 8px 16px;")
        load_btn.setToolTip("Load parameter values from a JSON file")
        load_btn.clicked.connect(self._on_load_settings)
        btn_layout.addWidget(load_btn)

        save_btn = QPushButton("Save Settings")
        save_btn.setStyleSheet("padding: 8px 16px;")
        save_btn.setToolTip("Save current parameter values to a JSON file")
        save_btn.clicked.connect(self._on_save_settings)
        btn_layout.addWidget(save_btn)

        btn_layout.addStretch()

        run_btn = QPushButton("Run")
        run_btn.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; "
            "font-weight: bold; padding: 8px 24px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #45a049; }"
        )
        run_btn.clicked.connect(self._on_run)

        close_btn = QPushButton("Close")
        close_btn.setStyleSheet("padding: 8px 24px;")
        close_btn.clicked.connect(self.close)

        btn_layout.addWidget(run_btn)
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)

        self._update_preview()
        self._connect_preview_signals()

    def _connect_preview_signals(self):
        for param in self.config.get("parameters", []):
            w = self.widgets.get(param["name"])
            if w is None:
                continue
            if isinstance(w, (QSpinBox, QDoubleSpinBox)):
                w.valueChanged.connect(self._update_preview)
            elif isinstance(w, QComboBox):
                w.currentTextChanged.connect(self._update_preview)
            elif isinstance(w, QCheckBox):
                w.stateChanged.connect(self._update_preview)
            elif isinstance(w, QLineEdit):
                w.textChanged.connect(self._update_preview)

    def _create_widget(self, param: dict) -> QWidget | None:
        ptype = param.get("type", "textbox")
        default = param.get("default")

        if ptype == "checkbox":
            toggle = ToggleSwitch()
            toggle.setChecked(bool(default))
            return toggle

        options = param.get("options") or []
        if ptype == "dropdown" or options:
            combo = QComboBox()
            for option in options:
                combo.addItem(str(option.get("label", option.get("value"))), option.get("value"))
            if default is not None:
                idx = combo.findData(default)
                if idx < 0:
                    idx = combo.findText(str(default))
                if idx >= 0:
                    combo.setCurrentIndex(idx)
            return combo

        if ptype in ("integer", "int"):
            spin = QSpinBox()
            spin.setMinimum(int(param.get("minimum", 0)))
            spin.setMaximum(int(param.get("maximum", 999999)))
            if default is not None:
                spin.setValue(int(default))
            return spin

        if ptype == "float":
            spin = QDoubleSpinBox()
            spin.setDecimals(8)
            spin.setMinimum(float(param.get("minimum", -999999.0)))
            spin.setMaximum(float(param.get("maximum", 999999.0)))
            if default is not None:
                spin.setValue(float(default))
            return spin

        line = QLineEdit()
        if default is not None:
            line.setText(str(default))
        return line

    def _get_values(self) -> dict:
        values = {}
        for param in self.config.get("parameters", []):
            w = self.widgets.get(param["name"])
            if w is None:
                continue
            if isinstance(w, QCheckBox):
                values[param["name"]] = w.isChecked()
            elif isinstance(w, (QSpinBox, QDoubleSpinBox)):
                values[param["name"]] = w.value()
            elif isinstance(w, QComboBox):
                data = w.currentData()
                values[param["name"]] = data if data is not None else w.currentText()
            elif isinstance(w, QLineEdit):
                values[param["name"]] = w.text()
        return values

    def _get_folders(self) -> dict:
        return {
            "infolder": self.folder_widgets["infolder"].text(),
            "outfolder": self.folder_widgets["outfolder"].text(),
        }

    def _get_docker_options(self) -> dict:
        return {"use_gpus": self.use_gpus_checkbox.isChecked()}

    def _browse_folder(self, line_edit: QLineEdit):
        current = line_edit.text()
        start = current if Path(current).is_dir() else str(SCRIPT_DIR)
        folder = QFileDialog.getExistingDirectory(self, "Select Folder", start)
        if folder:
            line_edit.setText(folder)

    def _update_preview(self):
        cmd = build_docker_command(
            self.config,
            self._get_values(),
            self._get_folders(),
            self._get_docker_options(),
        )
        self.cmd_preview.setPlainText(" ".join(cmd))

    def _settings_payload(self) -> dict:
        return {
            "values": self._get_values(),
            "folders": self._get_folders(),
            "docker_options": self._get_docker_options(),
        }

    def _save_settings(self):
        try:
            with open(LAST_SETTINGS_PATH, "w", encoding="utf-8") as f:
                json.dump(self._settings_payload(), f, indent=2)
        except OSError:
            pass

    def _apply_settings(self, data: dict):
        for key, line in self.folder_widgets.items():
            saved = data.get("folders", {}).get(key)
            if saved is not None:
                line.setText(str(saved))

        docker_options = data.get("docker_options", {})
        self.use_gpus_checkbox.setChecked(bool(docker_options.get("use_gpus", True)))

        saved_vals = data.get("values", {})
        for param in self.config.get("parameters", []):
            w = self.widgets.get(param["name"])
            val = saved_vals.get(param["name"])
            if w is None or val is None:
                continue
            if isinstance(w, QCheckBox):
                w.setChecked(bool(val))
            elif isinstance(w, QDoubleSpinBox):
                w.setValue(float(val))
            elif isinstance(w, QSpinBox):
                w.setValue(int(val))
            elif isinstance(w, QComboBox):
                idx = w.findData(val)
                if idx < 0:
                    idx = w.findText(str(val))
                if idx >= 0:
                    w.setCurrentIndex(idx)
            elif isinstance(w, QLineEdit):
                w.setText(str(val))

    def _on_restore(self):
        try:
            with open(LAST_SETTINGS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            return
        self._apply_settings(data)

    def _on_save_settings(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Settings", str(SCRIPT_DIR / "launcher_settings.json"),
            "JSON files (*.json);;All files (*)",
        )
        if path:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self._settings_payload(), f, indent=2)

    def _on_load_settings(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Settings", str(SCRIPT_DIR),
            "JSON files (*.json);;All files (*)",
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            return
        self._apply_settings(data)

    def _on_run(self):
        self._save_settings()
        cmd = build_docker_command(
            self.config,
            self._get_values(),
            self._get_folders(),
            self._get_docker_options(),
        )
        print("\n" + "=" * 70)
        print("Running:")
        print(" ".join(cmd))
        print("=" * 70 + "\n")

        self.close()
        subprocess.run(cmd)


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setWindowIcon(QIcon(str(ICON_PATH)))
    window = LauncherWindow()
    window.show()
    screen = app.primaryScreen().availableGeometry()
    window.move(
        (screen.width() - window.frameGeometry().width()) // 2,
        (screen.height() - window.frameGeometry().height()) // 2,
    )
    app.exec()


if __name__ == "__main__":
    main()
