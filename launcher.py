"""
launcher.py — PyQt6 GUI frontend for W_CIDeconvolve descriptor.json.

Dynamically reads descriptor.json and builds a form with appropriate
widgets for each parameter. On "Run" it executes the Docker container
in the console that launched this script.

Usage:
    python launcher.py
"""

import json
import os
import subprocess
import sys
from pathlib import Path

# Windows taskbar: set AppUserModelID so the taskbar shows our icon, not Python's
if sys.platform == "win32":
    import ctypes
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("ci.w_cideconvolve.launcher")

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QIcon
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSpinBox,
    QLineEdit,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

# Resolve paths relative to this script
SCRIPT_DIR = Path(__file__).resolve().parent
DESCRIPTOR_PATH = SCRIPT_DIR / "descriptor.json"
ICON_PATH = SCRIPT_DIR / "icon.svg"


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


def load_descriptor() -> dict:
    with open(DESCRIPTOR_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def build_docker_command(descriptor: dict, values: dict, folders: dict) -> list[str]:
    """Build the docker run command from descriptor and current widget values."""
    # Derive from descriptor, strip namespace (e.g. "cellularimagingcf/") for local run
    full_image = descriptor.get("container-image", {}).get("image", "w_cideconvolve")
    image = full_image.rsplit("/", 1)[-1]
    name = descriptor.get("name", image)

    cmd = [
        "docker", "run", "--rm", "--gpus", "all",
        "-v", f"{folders['infolder']}:/data/in",
        "-v", f"{folders['outfolder']}:/data/out",
        "-v", f"{folders['gtfolder']}:/data/gt",
        image,
        "--infolder", "/data/in",
        "--outfolder", "/data/out",
        "--gtfolder", "/data/gt",
        "--local",
    ]

    for inp in descriptor.get("inputs", []):
        param_id = inp["id"]
        flag = inp.get("command-line-flag", f"--{param_id}")
        val = values.get(param_id)
        if val is None:
            continue
        if inp["type"] == "Boolean":
            # argparse store_true: only pass the flag when True, omit when False
            if val:
                cmd.append(flag)
        else:
            cmd.extend([flag, str(val)])

    return cmd


class LauncherWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.descriptor = load_descriptor()
        self.widgets: dict[str, QWidget] = {}
        self._build_ui()

    def _build_ui(self):
        name = self.descriptor.get("name", "W_CIDeconvolve")
        self.setWindowTitle(f"{name} — Launcher")
        self.setWindowIcon(QIcon(str(ICON_PATH)))
        self.setMinimumWidth(676)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(12)

        # -- Header --
        title = QLabel(name)
        title.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        layout.addWidget(title)

        desc = self.descriptor.get("description", "")
        if desc:
            desc_label = QLabel(desc)
            desc_label.setWordWrap(True)
            desc_label.setStyleSheet("color: #666; margin-bottom: 8px;")
            layout.addWidget(desc_label)

        # -- Data folder info (read-only) --
        folder_group = QGroupBox("Data Folders")
        folder_layout = QFormLayout()
        folder_group.setLayout(folder_layout)

        for label_text, folder_path in [
            ("Input folder", str(SCRIPT_DIR / "infolder")),
            ("Output folder", str(SCRIPT_DIR / "outfolder")),
        ]:
            path_label = QLabel(folder_path)
            path_label.setStyleSheet("color: #555;")
            path_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            folder_layout.addRow(label_text + ":", path_label)

        layout.addWidget(folder_group)

        # -- Parameters from descriptor --
        param_group = QGroupBox("Parameters")
        param_layout = QFormLayout()
        param_group.setLayout(param_layout)

        for inp in self.descriptor.get("inputs", []):
            if inp.get("set-by-server", False):
                continue
            widget = self._create_widget(inp)
            if widget is not None:
                tooltip = inp.get("description", "")
                widget.setToolTip(tooltip)
                label = QLabel(inp.get("name", inp["id"]))
                label.setToolTip(tooltip)
                param_layout.addRow(label, widget)
                self.widgets[inp["id"]] = widget

        layout.addWidget(param_group)

        # -- Command preview --
        self.cmd_preview = QTextEdit()
        self.cmd_preview.setReadOnly(True)
        self.cmd_preview.setMaximumHeight(104)
        self.cmd_preview.setFont(QFont("Consolas", 9))
        self.cmd_preview.setStyleSheet("background: #1e1e1e; color: #dcdcdc;")
        layout.addWidget(QLabel("Command preview:"))
        layout.addWidget(self.cmd_preview)

        # -- Buttons --
        btn_layout = QHBoxLayout()
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

        btn_layout.addStretch()
        btn_layout.addWidget(run_btn)
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)

        # Initial preview
        self._update_preview()

        # Connect all widgets for live preview updates
        for inp in self.descriptor.get("inputs", []):
            w = self.widgets.get(inp["id"])
            if w is None:
                continue
            if isinstance(w, QSpinBox):
                w.valueChanged.connect(self._update_preview)
            elif isinstance(w, QComboBox):
                w.currentTextChanged.connect(self._update_preview)
            elif isinstance(w, QCheckBox):
                w.stateChanged.connect(self._update_preview)
            elif isinstance(w, QLineEdit):
                w.textChanged.connect(self._update_preview)

    def _create_widget(self, inp: dict) -> QWidget | None:
        ptype = inp.get("type", "String")
        default = inp.get("default-value")
        choices = inp.get("value-choices")

        if ptype == "Boolean":
            toggle = ToggleSwitch()
            toggle.setChecked(bool(default))
            return toggle

        if choices:
            combo = QComboBox()
            combo.addItems([str(c) for c in choices])
            if default is not None and str(default) in [str(c) for c in choices]:
                combo.setCurrentText(str(default))
            return combo

        if ptype == "Number":
            spin = QSpinBox()
            spin.setMinimum(inp.get("minimum", 0))
            spin.setMaximum(inp.get("maximum", 99999))
            if default is not None:
                spin.setValue(int(default))
            return spin

        # Fallback: plain text
        line = QLineEdit()
        if default is not None:
            line.setText(str(default))
        return line

    def _get_values(self) -> dict:
        values = {}
        for inp in self.descriptor.get("inputs", []):
            w = self.widgets.get(inp["id"])
            if w is None:
                continue
            if isinstance(w, QCheckBox):
                values[inp["id"]] = w.isChecked()
            elif isinstance(w, QSpinBox):
                values[inp["id"]] = w.value()
            elif isinstance(w, QComboBox):
                values[inp["id"]] = w.currentText()
            elif isinstance(w, QLineEdit):
                values[inp["id"]] = w.text()
        return values

    def _get_folders(self) -> dict:
        return {
            "infolder": str(SCRIPT_DIR / "infolder"),
            "outfolder": str(SCRIPT_DIR / "outfolder"),
            "gtfolder": str(SCRIPT_DIR / "gtfolder"),
        }

    def _update_preview(self):
        cmd = build_docker_command(
            self.descriptor, self._get_values(), self._get_folders()
        )
        self.cmd_preview.setPlainText(" ".join(cmd))

    def _on_run(self):
        cmd = build_docker_command(
            self.descriptor, self._get_values(), self._get_folders()
        )
        print("\n" + "=" * 70)
        print("Running:")
        print(" ".join(cmd))
        print("=" * 70 + "\n")

        self.close()

        # Run the docker command in the current console (inherits stdin/stdout)
        subprocess.run(cmd)


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setWindowIcon(QIcon(str(ICON_PATH)))
    window = LauncherWindow()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
