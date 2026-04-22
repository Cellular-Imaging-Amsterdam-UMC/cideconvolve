"""Dual-pane XYZT / 3D viewer for the CI deconvolution GUI.

Adapted from the omero-browser-qt 0.2.2 viewer concepts, but packaged as an
embeddable widget with fixed Original / Deconvolved panes and shared controls.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from PyQt6.QtCore import QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QImage, QPainter, QPixmap, QWheelEvent
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

try:
    from vispy import scene as vispy_scene
    from vispy.color import BaseColormap
    from vispy.visuals.transforms import STTransform

    _HAS_VISPY = True
except ImportError:
    _HAS_VISPY = False


_FALLBACK_PALETTE = [
    (0, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (255, 0, 0),
    (0, 0, 255),
    (255, 255, 0),
]

_RGB_CHANNEL_NAMES = ("R", "G", "B")
_RGB_CHANNEL_COLORS = (
    (220, 68, 68),
    (56, 184, 104),
    (72, 136, 255),
)

_PROJECTION_MODES = ["Slice", "MIP", "SUM"]
_VIEW_SELECTOR_MODES = ["Both", "Original", "Deconvolved"]
_TWO_D_MODE = "2D"
_THREE_D_MODE = "3D"
_VOLUME_METHODS = [
    "mip",
    "attenuated_mip",
    "minip",
    "translucent",
    "average",
    "iso",
    "additive",
]
_VOLUME_METHOD_LABELS = {
    "mip": "MIP",
    "attenuated_mip": "Attenuated MIP",
    "minip": "MinIP",
    "translucent": "Translucent",
    "average": "Average",
    "iso": "Isosurface",
    "additive": "Additive",
}
_VOLUME_METHOD_UI = {
    "mip": {"label": "Gain:", "range": (1, 200), "default": 100, "role": "gain"},
    "attenuated_mip": {"label": "Atten.:", "range": (1, 300), "default": 100, "role": "attenuation"},
    "minip": {"label": "Cutoff:", "range": (0, 100), "default": 100, "role": "minip_cutoff"},
    "translucent": {"label": "Gain:", "range": (1, 500), "default": 200, "role": "gain"},
    "average": {"label": "Gain:", "range": (1, 600), "default": 180, "role": "gain"},
    "iso": {"label": "Threshold:", "range": (0, 100), "default": 22, "role": "threshold"},
    "additive": {"label": "Gain:", "range": (1, 200), "default": 28, "role": "gain"},
}
_INTERPOLATION_TOGGLE_METHODS = set(_VOLUME_METHODS) - {"iso"}


def _emission_to_rgb(wavelength_nm: Optional[float]) -> tuple[int, int, int]:
    if wavelength_nm is None:
        return (255, 255, 255)
    wl = float(wavelength_nm)
    r = g = b = 0.0
    if 380 <= wl < 440:
        r = -(wl - 440) / (440 - 380)
        b = 1.0
    elif 440 <= wl < 490:
        g = (wl - 440) / (490 - 440)
        b = 1.0
    elif 490 <= wl < 510:
        g = 1.0
        b = -(wl - 510) / (510 - 490)
    elif 510 <= wl < 580:
        r = (wl - 510) / (580 - 510)
        g = 1.0
    elif 580 <= wl < 645:
        r = 1.0
        g = -(wl - 645) / (645 - 580)
    elif 645 <= wl <= 780:
        r = 1.0
    else:
        return (255, 255, 255)
    return (int(r * 255), int(g * 255), int(b * 255))


def _channels_look_like_rgb(channels: list[dict]) -> bool:
    if len(channels) != 3:
        return False
    names = [str(ch.get("name", "")).strip().lower() for ch in channels]
    if names in (["r", "g", "b"], ["red", "green", "blue"]):
        return True
    colors = [tuple(ch.get("color", (0, 0, 0))) for ch in channels]
    return set(colors) == set(_RGB_CHANNEL_COLORS)


def _channels_look_fluorescence_like(channels: list[dict]) -> bool:
    if not channels or _channels_look_like_rgb(channels):
        return False
    fluor_markers = (
        "dapi", "fitc", "gfp", "yfp", "cfp", "rfp", "mcherry", "tdtomato",
        "tritc", "cy3", "cy5", "alexa", "hoechst", "far red",
    )
    for ch in channels:
        emission = ch.get("emission_wavelength")
        if emission is not None:
            try:
                if float(emission) > 0:
                    return True
            except (TypeError, ValueError):
                pass
        name = str(ch.get("name", "")).strip().lower()
        if any(marker in name for marker in fluor_markers):
            return True
    return False


def _resolve_channel_colors(channels: list[dict]) -> list[tuple[int, int, int]]:
    colors: list[tuple[int, int, int] | None] = []
    for ch in channels:
        color = ch.get("color")
        if isinstance(color, tuple) and len(color) == 3 and color != (255, 255, 255):
            colors.append(tuple(int(v) for v in color))
            continue
        emission = ch.get("emission_wavelength")
        colors.append(_emission_to_rgb(emission))
    if len(colors) > 1 and len(set(colors)) == 1:
        return [_FALLBACK_PALETTE[i % len(_FALLBACK_PALETTE)] for i in range(len(colors))]
    return [c if c is not None else _FALLBACK_PALETTE[i % len(_FALLBACK_PALETTE)] for i, c in enumerate(colors)]


def _project_stack(stack: np.ndarray, mode: str, z_index: int) -> np.ndarray:
    if stack.ndim != 3:
        raise ValueError(f"Expected ZYX stack, got {stack.shape}")
    if mode == "Slice":
        z = max(0, min(int(z_index), stack.shape[0] - 1))
        return stack[z]
    if mode == "MIP":
        return stack.max(axis=0)
    if mode == "SUM":
        return stack.sum(axis=0).astype(np.float64)
    raise ValueError(f"Unsupported projection mode: {mode}")


def _composite_to_pixmap(
    slices: list[tuple[np.ndarray, tuple[int, int, int], tuple[float, float]]],
) -> QPixmap:
    if not slices:
        return QPixmap()
    height, width = slices[0][0].shape
    canvas = np.zeros((height, width, 3), dtype=np.float64)
    for arr, (cr, cg, cb), (lo, hi) in slices:
        if hi <= lo:
            hi = lo + 1.0
        norm = (arr.astype(np.float64) - lo) / (hi - lo)
        np.clip(norm, 0.0, 1.0, out=norm)
        canvas[..., 0] += norm * (cr / 255.0)
        canvas[..., 1] += norm * (cg / 255.0)
        canvas[..., 2] += norm * (cb / 255.0)
    np.clip(canvas, 0.0, 1.0, out=canvas)
    rgb = np.ascontiguousarray((canvas * 255).astype(np.uint8))
    qimg = QImage(rgb.data, width, height, 3 * width, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


class ZoomableImageView(QGraphicsView):
    """Simple pannable / zoomable QGraphicsView with optional linking."""

    _ZOOM_FACTOR = 1.15

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pix_item: Optional[QGraphicsPixmapItem] = None
        self._linked: list["ZoomableImageView"] = []
        self._syncing = False
        self.setRenderHints(
            self.renderHints()
            | QPainter.RenderHint.Antialiasing
            | QPainter.RenderHint.SmoothPixmapTransform
        )
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setMinimumSize(260, 260)
        self.setStyleSheet(
            "QGraphicsView { background: #17191c; border: 1px solid #343a40; border-radius: 8px; }"
        )

    def link_to(self, other: "ZoomableImageView") -> None:
        if other not in self._linked:
            self._linked.append(other)
        if self not in other._linked:
            other._linked.append(self)

    def set_pixmap(self, pixmap: Optional[QPixmap]) -> None:
        self._scene.clear()
        self._pix_item = None
        if pixmap is not None and not pixmap.isNull():
            self._pix_item = self._scene.addPixmap(pixmap)
            self._scene.setSceneRect(QRectF(pixmap.rect()))

    def clear(self) -> None:
        self._scene.clear()
        self._pix_item = None

    def fit_in_view(self) -> None:
        if self._pix_item is not None:
            self.resetTransform()
            self.fitInView(self._pix_item, Qt.AspectRatioMode.KeepAspectRatio)

    def copy_view_state_from(self, other: "ZoomableImageView") -> None:
        self.setTransform(other.transform())
        self.horizontalScrollBar().setValue(other.horizontalScrollBar().value())
        self.verticalScrollBar().setValue(other.verticalScrollBar().value())

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        factor = self._ZOOM_FACTOR if event.angleDelta().y() > 0 else 1.0 / self._ZOOM_FACTOR
        self.scale(factor, factor)
        self._sync_transform()

    def scrollContentsBy(self, dx: int, dy: int) -> None:  # noqa: N802
        super().scrollContentsBy(dx, dy)
        if not self._syncing:
            self._sync_transform()

    def _sync_transform(self) -> None:
        if self._syncing:
            return
        self._syncing = True
        transform = self.transform()
        x_value = self.horizontalScrollBar().value()
        y_value = self.verticalScrollBar().value()
        for other in self._linked:
            other._syncing = True
            other.setTransform(transform)
            other.horizontalScrollBar().setValue(x_value)
            other.verticalScrollBar().setValue(y_value)
            other._syncing = False
        self._syncing = False


@dataclass
class _PaneMessage:
    text: str


class _PaneWidget(QWidget):
    cameraStateChanged = pyqtSignal(object)

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self._syncing_camera = False
        self._volumes: list[object] = []
        self._vol_camera_ranges: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None = None
        self._camera_emit_timer = QTimer(self)
        self._camera_emit_timer.setSingleShot(True)
        self._camera_emit_timer.setInterval(30)
        self._camera_emit_timer.timeout.connect(self._emit_camera_state)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        title_label = QLabel(title)
        title_label.setStyleSheet("font-weight: 700; color: #f3f4f6;")
        layout.addWidget(title_label)

        self._display_stack = QStackedWidget()
        layout.addWidget(self._display_stack, stretch=1)

        self._mode_stack = QStackedWidget()
        self._display_stack.addWidget(self._mode_stack)

        self.view2d = ZoomableImageView()
        self._mode_stack.addWidget(self.view2d)

        if _HAS_VISPY:
            self._vispy_canvas = vispy_scene.SceneCanvas(keys="interactive", show=False, bgcolor="#000000")
            self._vispy_view = self._vispy_canvas.central_widget.add_view()
            self._vispy_view.camera = vispy_scene.ArcballCamera(fov=60, distance=None)
            self._connect_camera_events(self._vispy_view.camera)
            self._mode_stack.addWidget(self._vispy_canvas.native)
        else:
            self._vispy_canvas = None
            self._vispy_view = None
            missing = QLabel("3D viewer unavailable.\nInstall vispy and PyOpenGL.")
            missing.setAlignment(Qt.AlignmentFlag.AlignCenter)
            missing.setStyleSheet("color: #c0c4c8; background: #111315; border: 1px solid #343a40; border-radius: 8px;")
            self._mode_stack.addWidget(missing)

        self._placeholder = QLabel("")
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setWordWrap(True)
        self._placeholder.setStyleSheet(
            "color: #d0d3d7; background: #111315; border: 1px dashed #43484d; border-radius: 8px; padding: 18px;"
        )
        self._display_stack.addWidget(self._placeholder)
        self._display_stack.setCurrentIndex(0)

    def set_mode(self, mode: str) -> None:
        if mode == _THREE_D_MODE:
            self._mode_stack.setCurrentIndex(1)
        else:
            self._mode_stack.setCurrentIndex(0)

    def show_placeholder(self, text: str) -> None:
        self._placeholder.setText(text)
        self._display_stack.setCurrentIndex(1)
        self.view2d.clear()
        self.clear_3d()

    def show_content(self) -> None:
        self._display_stack.setCurrentIndex(0)

    def set_pixmap(self, pixmap: Optional[QPixmap]) -> None:
        self.show_content()
        self._mode_stack.setCurrentIndex(0)
        self.view2d.set_pixmap(pixmap)

    def fit_2d(self) -> None:
        self.view2d.fit_in_view()

    def copy_2d_view_from(self, other: "_PaneWidget") -> None:
        self.view2d.copy_view_state_from(other.view2d)

    def clear_3d(self) -> None:
        if not _HAS_VISPY or self._vispy_view is None:
            return
        for volume in self._volumes:
            try:
                volume.parent = None
            except Exception:
                pass
        self._volumes.clear()
        self._vol_camera_ranges = None
        self._vispy_canvas.update()

    def load_3d(
        self,
        stacks: list[tuple[np.ndarray, tuple[int, int, int], tuple[float, float]]],
        *,
        method: str,
        slider_val: float,
        interpolation: str,
        downsample: int,
        pixel_size_x: Optional[float],
        pixel_size_z: Optional[float],
        preserve_camera_state: object = None,
    ) -> None:
        self.show_content()
        self._mode_stack.setCurrentIndex(1)
        if not _HAS_VISPY or self._vispy_view is None:
            return
        self.clear_3d()
        z_scale = 1.0
        if pixel_size_x and pixel_size_z and pixel_size_x > 0:
            z_scale = float(pixel_size_z) / float(pixel_size_x)

        reference_shape: tuple[int, int, int] | None = None
        for stack, color, contrast in stacks:
            if reference_shape is None:
                reference_shape = tuple(int(v) for v in stack.shape)
            work = stack[::downsample, ::downsample, ::downsample] if downsample > 1 else stack
            lo, hi = contrast
            gain = slider_val if _VOLUME_METHOD_UI[method]["role"] == "gain" else 1.0
            data = self._prepare_volume_data(work, lo, hi, method, gain)
            cmap = _ChannelColormap(color, translucent_boost=(method == "translucent"))
            volume = vispy_scene.visuals.Volume(
                data,
                parent=self._vispy_view.scene,
                method=method,
                threshold=slider_val if _VOLUME_METHOD_UI[method]["role"] == "threshold" else 0.0,
                attenuation=slider_val if _VOLUME_METHOD_UI[method]["role"] == "attenuation" else 1.0,
                mip_cutoff=slider_val if _VOLUME_METHOD_UI[method]["role"] == "mip_cutoff" else None,
                minip_cutoff=slider_val if _VOLUME_METHOD_UI[method]["role"] == "minip_cutoff" else None,
                cmap=cmap,
                interpolation=interpolation,
            )
            volume.transform = STTransform(scale=(downsample, downsample, z_scale * downsample))
            volume.set_gl_state("additive", depth_test=False)
            self._volumes.append(volume)

        if reference_shape is not None:
            oz, oy, ox = reference_shape
            self._vol_camera_ranges = (
                (0.0, float(max(ox - 1, 0))),
                (0.0, float(max(oy - 1, 0))),
                (0.0, float(max(oz - 1, 0)) * z_scale),
            )
        if preserve_camera_state is not None:
            self.reset_camera()
            self.apply_camera_state(preserve_camera_state)
        else:
            self.reset_camera()

    def apply_camera_state(self, state: object) -> None:
        if not _HAS_VISPY or self._vispy_view is None or state is None:
            return
        camera = self._vispy_view.camera
        if not hasattr(camera, "set_state"):
            return
        self._syncing_camera = True
        try:
            camera.set_state(state)
            self._vispy_canvas.update()
        finally:
            self._syncing_camera = False

    def camera_state(self) -> object:
        if not _HAS_VISPY or self._vispy_view is None:
            return None
        camera = self._vispy_view.camera
        if hasattr(camera, "get_state"):
            return dict(camera.get_state())
        return None

    def reset_camera(self) -> None:
        if not _HAS_VISPY or self._vispy_view is None:
            return
        camera = vispy_scene.ArcballCamera(fov=60, distance=None)
        self._vispy_view.camera = camera
        self._connect_camera_events(camera)
        if self._vol_camera_ranges is not None:
            x_range, y_range, z_range = self._vol_camera_ranges
            camera.set_range(x=x_range, y=y_range, z=z_range, margin=0.05)
            camera.center = (
                0.5 * (x_range[0] + x_range[1]),
                0.5 * (y_range[0] + y_range[1]),
                0.5 * (z_range[0] + z_range[1]),
            )
            if hasattr(camera, "set_default_state"):
                camera.set_default_state()
        self._vispy_canvas.update()

    def _connect_camera_events(self, camera) -> None:
        """Hook camera-change notifications in a vispy-version-tolerant way."""
        camera_events = getattr(camera, "events", None)
        if camera_events is not None:
            changed = getattr(camera_events, "changed", None)
            if changed is not None and hasattr(changed, "connect"):
                changed.connect(self._on_camera_changed)
                return

        if self._vispy_canvas is None:
            return
        canvas_events = getattr(self._vispy_canvas, "events", None)
        if canvas_events is None:
            return
        for name in ("mouse_move", "mouse_wheel", "mouse_release", "resize"):
            emitter = getattr(canvas_events, name, None)
            if emitter is not None and hasattr(emitter, "connect"):
                emitter.connect(self._schedule_camera_emit)

    def _schedule_camera_emit(self, _event=None) -> None:
        if self._syncing_camera:
            return
        self._camera_emit_timer.start()

    def _emit_camera_state(self) -> None:
        state = self.camera_state()
        if state is not None:
            self.cameraStateChanged.emit(state)

    def _prepare_volume_data(
        self,
        stack: np.ndarray,
        lo: float,
        hi: float,
        method: str,
        gain: float,
    ) -> np.ndarray:
        volume = stack.astype(np.float32)
        denom = hi - lo if hi > lo else 1.0
        volume = (volume - lo) / denom
        np.clip(volume, 0.0, 1.0, out=volume)
        if method == "translucent":
            np.power(volume, 0.5, out=volume)
            volume *= gain
        elif method == "average":
            np.power(volume, 0.7, out=volume)
            volume = 1.0 - np.exp(-(volume * gain * 1.8))
        else:
            volume *= gain
        np.clip(volume, 0.0, 1.0, out=volume)
        return volume

    def _on_camera_changed(self, _event=None) -> None:
        if self._syncing_camera:
            return
        self._schedule_camera_emit()


class DualViewerWidget(QWidget):
    timepointChanged = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._metadata: dict = {}
        self._input_channels: list[np.ndarray] = []
        self._loaded_input_timepoint: Optional[int] = None
        self._preview_by_t: dict[int, list[np.ndarray]] = {}
        self._channel_buttons: list[QPushButton] = []
        self._channel_colors: list[tuple[int, int, int]] = []
        self._available_volume_methods = list(_VOLUME_METHODS)
        self._active_volume_method = _VOLUME_METHODS[0]
        self._volume_method_values = {
            method: spec["default"] for method, spec in _VOLUME_METHOD_UI.items()
        }
        self._syncing_camera = False
        self._reset_3d_on_next_render = False
        self._fit_on_next_render = False
        self._refresh_3d_timer = QTimer(self)
        self._refresh_3d_timer.setSingleShot(True)
        self._refresh_3d_timer.setInterval(150)
        self._refresh_3d_timer.timeout.connect(self._refresh_view)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)

        self._build_toolbar(root)
        self._build_panes(root)
        self._refresh_view()

    def _build_toolbar(self, root: QVBoxLayout) -> None:
        self._channel_bar = QHBoxLayout()
        self._channel_bar.addWidget(QLabel("Channels:"))
        self._channel_bar.addStretch()
        root.addLayout(self._channel_bar)

        top = QHBoxLayout()
        top.setSpacing(8)
        mode_group = QWidget()
        mode_layout = QHBoxLayout(mode_group)
        mode_layout.setContentsMargins(0, 0, 0, 0)
        mode_layout.setSpacing(6)
        mode_layout.addWidget(QLabel("Mode:"))
        self._mode_combo = QComboBox()
        self._mode_combo.addItems([_TWO_D_MODE, _THREE_D_MODE])
        self._mode_combo.currentTextChanged.connect(self._on_mode_changed)
        mode_layout.addWidget(self._mode_combo)
        top.addWidget(mode_group)

        show_group = QWidget()
        show_layout = QHBoxLayout(show_group)
        show_layout.setContentsMargins(0, 0, 0, 0)
        show_layout.setSpacing(6)
        show_layout.addWidget(QLabel("Show:"))
        self._view_selector = QComboBox()
        self._view_selector.addItems(_VIEW_SELECTOR_MODES)
        self._view_selector.currentTextChanged.connect(self._apply_view_selector)
        show_layout.addWidget(self._view_selector)
        top.addWidget(show_group)

        view_group = QWidget()
        view_layout = QHBoxLayout(view_group)
        view_layout.setContentsMargins(0, 0, 0, 0)
        view_layout.setSpacing(6)
        view_layout.addWidget(QLabel("View:"))
        self._projection_combo = QComboBox()
        self._projection_combo.addItems(_PROJECTION_MODES)
        self._projection_combo.currentTextChanged.connect(self._refresh_view)
        view_layout.addWidget(self._projection_combo)
        top.addWidget(view_group)

        self._fit_button = QPushButton("Fit")
        self._fit_button.clicked.connect(self.fit_views)
        top.addWidget(self._fit_button)

        lo_group = QWidget()
        lo_layout = QHBoxLayout(lo_group)
        lo_layout.setContentsMargins(0, 0, 0, 0)
        lo_layout.setSpacing(6)
        lo_layout.addWidget(QLabel("Lo%:"))
        self._lo_spin = QDoubleSpinBox()
        self._lo_spin.setRange(0.0, 50.0)
        self._lo_spin.setDecimals(3)
        self._lo_spin.setSingleStep(0.01)
        self._lo_spin.setValue(0.1)
        self._lo_spin.valueChanged.connect(self._on_contrast_changed)
        lo_layout.addWidget(self._lo_spin)
        top.addWidget(lo_group)

        hi_group = QWidget()
        hi_layout = QHBoxLayout(hi_group)
        hi_layout.setContentsMargins(0, 0, 0, 0)
        hi_layout.setSpacing(6)
        hi_layout.addWidget(QLabel("Hi%:"))
        self._hi_spin = QDoubleSpinBox()
        self._hi_spin.setRange(50.0, 100.0)
        self._hi_spin.setDecimals(3)
        self._hi_spin.setSingleStep(0.01)
        self._hi_spin.setValue(100.0)
        self._hi_spin.valueChanged.connect(self._on_contrast_changed)
        hi_layout.addWidget(self._hi_spin)
        top.addWidget(hi_group)
        top.addStretch()

        root.addLayout(top)

        self._bar_3d = QWidget()
        bar_3d_layout = QHBoxLayout(self._bar_3d)
        bar_3d_layout.setContentsMargins(0, 0, 0, 0)
        bar_3d_layout.setSpacing(8)
        bar_3d_layout.addWidget(QLabel("Render:"))
        self._volume_method_combo = QComboBox()
        self._volume_method_combo.currentIndexChanged.connect(self._on_volume_method_changed)
        bar_3d_layout.addWidget(self._volume_method_combo)
        self._volume_slider_label = QLabel("Gain:")
        bar_3d_layout.addWidget(self._volume_slider_label)
        self._volume_slider = QSlider(Qt.Orientation.Horizontal)
        self._volume_slider.valueChanged.connect(self._on_volume_slider_changed)
        bar_3d_layout.addWidget(self._volume_slider)
        self._volume_slider_value = QLabel("1.00")
        self._volume_slider_value.setMinimumWidth(40)
        bar_3d_layout.addWidget(self._volume_slider_value)
        bar_3d_layout.addWidget(QLabel("Downsample:"))
        self._downsample_combo = QComboBox()
        self._downsample_combo.addItems(["1x", "2x", "4x"])
        self._downsample_combo.currentIndexChanged.connect(self._schedule_3d_refresh)
        bar_3d_layout.addWidget(self._downsample_combo)
        self._smooth_check = QCheckBox("Smooth")
        self._smooth_check.setChecked(True)
        self._smooth_check.toggled.connect(self._schedule_3d_refresh)
        bar_3d_layout.addWidget(self._smooth_check)
        self._reset_3d_button = QPushButton("Reset View")
        self._reset_3d_button.clicked.connect(self._reset_3d_views)
        bar_3d_layout.addWidget(self._reset_3d_button)
        bar_3d_layout.addStretch()
        root.addWidget(self._bar_3d)

        self._refresh_volume_method_options()
        self._bar_3d.setVisible(False)

    def _build_panes(self, root: QVBoxLayout) -> None:
        body = QHBoxLayout()
        body.setSpacing(10)

        panes = QHBoxLayout()
        panes.setSpacing(10)
        self._input_pane = _PaneWidget("Original")
        self._output_pane = _PaneWidget("Deconvolved")
        self._input_pane.view2d.link_to(self._output_pane.view2d)
        self._input_pane.cameraStateChanged.connect(self._sync_3d_from_input)
        self._output_pane.cameraStateChanged.connect(self._sync_3d_from_output)
        panes.addWidget(self._input_pane, stretch=1)
        panes.addWidget(self._output_pane, stretch=1)
        body.addLayout(panes, stretch=1)

        z_panel = QWidget()
        z_layout = QVBoxLayout(z_panel)
        z_layout.setContentsMargins(0, 0, 0, 0)
        z_layout.setSpacing(6)
        z_title = QLabel("Z:")
        z_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        z_layout.addWidget(z_title)
        self._z_slider = QSlider(Qt.Orientation.Vertical)
        self._z_slider.setMinimum(0)
        self._z_slider.setMaximum(0)
        self._z_slider.valueChanged.connect(self._refresh_view)
        z_layout.addWidget(self._z_slider, stretch=1)
        self._z_label = QLabel("0/0")
        self._z_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        z_layout.addWidget(self._z_label)
        body.addWidget(z_panel)

        root.addLayout(body, stretch=1)

        self._time_bar = QWidget()
        time_layout = QHBoxLayout(self._time_bar)
        time_layout.setContentsMargins(0, 0, 0, 0)
        time_layout.setSpacing(8)
        time_layout.addWidget(QLabel("T:"))
        self._t_slider = QSlider(Qt.Orientation.Horizontal)
        self._t_slider.setMinimum(0)
        self._t_slider.setMaximum(0)
        self._t_slider.valueChanged.connect(self._on_time_changed)
        time_layout.addWidget(self._t_slider, stretch=1)
        self._t_label = QLabel("0/0")
        self._t_label.setMinimumWidth(50)
        time_layout.addWidget(self._t_label)
        root.addWidget(self._time_bar)

    def set_input_data(self, channels: list[np.ndarray], metadata: dict) -> None:
        self._input_channels = channels
        self._loaded_input_timepoint = None
        self._metadata = metadata
        self._preview_by_t.clear()
        channels_meta = self._display_channels()
        self._channel_colors = _resolve_channel_colors(channels_meta)
        self._rebuild_channel_buttons(channels_meta)
        self._refresh_volume_method_options(channels_meta)
        size_t = max(int(metadata.get("size_t", 1)), 1)
        size_z = max(int(metadata.get("size_z", 1)), 1)
        default_t = max(0, min(int(metadata.get("default_t", 0)), size_t - 1))
        default_z = max(0, min(int(metadata.get("default_z", size_z // 2 if size_z > 1 else 0)), size_z - 1))
        self._t_slider.blockSignals(True)
        self._t_slider.setMaximum(size_t - 1)
        self._t_slider.setValue(default_t)
        self._t_slider.blockSignals(False)
        self._z_slider.blockSignals(True)
        self._z_slider.setMaximum(size_z - 1)
        self._z_slider.setValue(default_z)
        self._z_slider.blockSignals(False)
        self._time_bar.setVisible(size_t > 1)
        self._update_labels()
        self._fit_on_next_render = True
        self._ensure_mode_valid()
        self._refresh_view()

    def set_input_timepoint_data(self, timepoint: int, channels_zyx: list[np.ndarray]) -> None:
        self._input_channels = list(channels_zyx)
        self._loaded_input_timepoint = int(timepoint)
        self._ensure_mode_valid()
        self._refresh_view()

    def clear_preview_results(self) -> None:
        self._preview_by_t.clear()
        self._refresh_view()

    def set_preview_result(self, timepoint: int, channels_zyx: list[np.ndarray]) -> None:
        self._preview_by_t[int(timepoint)] = channels_zyx
        self._refresh_view()

    def current_timepoint(self) -> int:
        return self._t_slider.value()

    def has_time_axis(self) -> bool:
        return self._t_slider.maximum() > 0

    def set_timepoint(self, timepoint: int) -> None:
        timepoint = max(0, min(timepoint, self._t_slider.maximum()))
        self._t_slider.setValue(timepoint)

    def has_preview_for_timepoint(self, timepoint: int) -> bool:
        return int(timepoint) in self._preview_by_t

    def current_preview_channels(self) -> list[np.ndarray]:
        return list(self._preview_by_t.get(self.current_timepoint(), []))

    def lo_percentile(self) -> float:
        return float(self._lo_spin.value())

    def hi_percentile(self) -> float:
        return float(self._hi_spin.value())

    def set_lo_percentile(self, value: float) -> None:
        self._lo_spin.setValue(value)

    def set_hi_percentile(self, value: float) -> None:
        self._hi_spin.setValue(value)

    def fit_views(self) -> None:
        if self._mode_combo.currentText() == _THREE_D_MODE:
            self._reset_3d_views()
            return
        self._input_pane.fit_2d()
        self._output_pane.fit_2d()

    def refresh_view(self) -> None:
        self._refresh_view()

    def _display_channels(self) -> list[dict]:
        channels = list(self._metadata.get("channels", []))
        names = self._metadata.get("channel_names", [])
        result: list[dict] = []
        declared_count = max(int(self._metadata.get("size_c", 0)), 0)
        for i in range(max(len(self._input_channels), len(channels), declared_count)):
            src = dict(channels[i]) if i < len(channels) else {}
            if "name" not in src:
                src["name"] = names[i] if i < len(names) else f"Ch {i}"
            src.setdefault("active", True)
            result.append(src)
        if _channels_look_like_rgb(result):
            for i, ch in enumerate(result[:3]):
                ch["name"] = _RGB_CHANNEL_NAMES[i]
                ch["color"] = _RGB_CHANNEL_COLORS[i]
        return result

    def _rebuild_channel_buttons(self, channels: list[dict]) -> None:
        for btn in self._channel_buttons:
            self._channel_bar.removeWidget(btn)
            btn.deleteLater()
        self._channel_buttons.clear()
        for i, channel in enumerate(channels):
            name = channel.get("name", f"Ch {i}")
            r, g, b = self._channel_colors[i]
            btn = QPushButton(name)
            btn.setCheckable(True)
            btn.setChecked(bool(channel.get("active", True)))
            btn.setStyleSheet(
                f"QPushButton {{ color: rgb({r},{g},{b}); font-weight: bold; border: 2px solid rgb({r},{g},{b}); padding: 2px 8px; }}"
                f"QPushButton:checked {{ background-color: rgba({r},{g},{b},60); }}"
            )
            btn.toggled.connect(self._refresh_view)
            self._channel_bar.insertWidget(self._channel_bar.count() - 1, btn)
            self._channel_buttons.append(btn)

    def _active_channel_indices(self) -> list[int]:
        return [
            i for i, btn in enumerate(self._channel_buttons)
            if btn.isChecked()
        ]

    def _on_mode_changed(self, mode: str) -> None:
        if mode == _THREE_D_MODE and not self._can_show_3d():
            self._mode_combo.blockSignals(True)
            self._mode_combo.setCurrentText(_TWO_D_MODE)
            self._mode_combo.blockSignals(False)
            mode = _TWO_D_MODE
        if mode == _THREE_D_MODE:
            self._reset_3d_on_next_render = True
        self._bar_3d.setVisible(mode == _THREE_D_MODE)
        self._projection_combo.setEnabled(mode == _TWO_D_MODE)
        self._z_slider.setEnabled(mode == _TWO_D_MODE and self._z_slider.maximum() > 0 and self._projection_combo.currentText() == "Slice")
        self._refresh_view()

    def _on_time_changed(self, value: int) -> None:
        self._update_labels()
        self.timepointChanged.emit(value)
        self._refresh_view()

    def _on_contrast_changed(self) -> None:
        if self._mode_combo.currentText() == _THREE_D_MODE:
            self._schedule_3d_refresh()
        else:
            self._refresh_view()

    def _on_volume_method_changed(self, index: int) -> None:
        if index < 0 or index >= len(self._available_volume_methods):
            return
        self._active_volume_method = self._available_volume_methods[index]
        self._set_volume_slider_ui(
            self._active_volume_method,
            self._volume_method_values[self._active_volume_method],
        )
        self._schedule_3d_refresh()

    def _on_volume_slider_changed(self, value: int) -> None:
        method = self._active_volume_method
        self._volume_method_values[method] = value
        self._volume_slider_value.setText(f"{value / 100:.2f}")
        self._schedule_3d_refresh()

    def _schedule_3d_refresh(self) -> None:
        if self._mode_combo.currentText() == _THREE_D_MODE:
            self._refresh_3d_timer.start()

    def _refresh_volume_method_options(self, channels: Optional[list[dict]] = None) -> None:
        current = self._active_volume_method
        fluorescence_like = _channels_look_fluorescence_like(channels or self._display_channels())
        methods = [m for m in _VOLUME_METHODS if m != "minip" or not fluorescence_like]
        if current not in methods:
            current = methods[0]
        self._available_volume_methods = methods
        self._active_volume_method = current
        self._volume_method_combo.blockSignals(True)
        self._volume_method_combo.clear()
        self._volume_method_combo.addItems([_VOLUME_METHOD_LABELS[m] for m in methods])
        self._volume_method_combo.setCurrentIndex(methods.index(current))
        self._volume_method_combo.blockSignals(False)
        self._set_volume_slider_ui(current, self._volume_method_values[current])

    def _set_volume_slider_ui(self, method: str, value: int) -> None:
        spec = _VOLUME_METHOD_UI[method]
        min_val, max_val = spec["range"]
        clamped = max(min_val, min(int(value), max_val))
        self._volume_slider_label.setText(spec["label"])
        self._volume_slider.blockSignals(True)
        self._volume_slider.setRange(min_val, max_val)
        self._volume_slider.setValue(clamped)
        self._volume_slider.blockSignals(False)
        self._volume_slider_value.setText(f"{clamped / 100:.2f}")
        self._smooth_check.setVisible(method in _INTERPOLATION_TOGGLE_METHODS)

    def _apply_view_selector(self) -> None:
        mode = self._view_selector.currentText()
        self._input_pane.setVisible(mode in ("Both", "Original"))
        self._output_pane.setVisible(mode in ("Both", "Deconvolved"))

    def _update_labels(self) -> None:
        t = self._t_slider.value()
        z = self._z_slider.value()
        self._t_label.setText(f"{t}/{max(self._t_slider.maximum(), 0)}")
        self._z_label.setText(f"{z}/{max(self._z_slider.maximum(), 0)}")

    def _ensure_mode_valid(self) -> None:
        can_show_3d = self._can_show_3d()
        if self._mode_combo.currentText() == _THREE_D_MODE and not can_show_3d:
            self._mode_combo.setCurrentText(_TWO_D_MODE)

    def _can_show_3d(self) -> bool:
        return _HAS_VISPY and bool(self._input_channels) and max(int(self._metadata.get("size_z", 1)), 1) > 1

    def _refresh_view(self) -> None:
        self._update_labels()
        self._apply_view_selector()
        if not self._input_channels:
            self._input_pane.show_placeholder("Open an image to view it.")
            self._output_pane.show_placeholder("Run deconvolution to preview results.")
            return
        if self._loaded_input_timepoint is not None and self._loaded_input_timepoint != self.current_timepoint():
            self._input_pane.show_placeholder(f"Loading T={self.current_timepoint()}…")
            preview = self._preview_by_t.get(self.current_timepoint())
            if preview:
                self._output_pane.show_content()
            else:
                self._output_pane.show_placeholder(
                    f"No deconvolved preview for T={self.current_timepoint()}.\nRun Deconvolution to preview this timepoint."
                )
            return

        mode = self._mode_combo.currentText()
        if mode == _THREE_D_MODE and self._can_show_3d():
            self._refresh_view_3d()
        else:
            self._refresh_view_2d()

    def _refresh_view_2d(self) -> None:
        timepoint = self.current_timepoint()
        projection = self._projection_combo.currentText()
        self._z_slider.setEnabled(self._z_slider.maximum() > 0 and projection == "Slice")
        self._input_pane.set_mode(_TWO_D_MODE)
        self._output_pane.set_mode(_TWO_D_MODE)

        input_pixmap = self._build_2d_pixmap(self._input_channels, projection)
        self._input_pane.set_pixmap(input_pixmap)

        preview = self._preview_by_t.get(timepoint)
        if preview:
            output_was_placeholder = self._output_pane._display_stack.currentIndex() == 1
            output_pixmap = self._build_2d_pixmap(preview, projection)
            self._output_pane.set_pixmap(output_pixmap)
            if output_was_placeholder:
                self._output_pane.copy_2d_view_from(self._input_pane)
        else:
            self._output_pane.show_placeholder(
                f"No deconvolved preview for T={timepoint}.\nRun Deconvolution to preview this timepoint."
            )

        if self._fit_on_next_render:
            self.fit_views()
            self._fit_on_next_render = False

    def _refresh_view_3d(self) -> None:
        timepoint = self.current_timepoint()
        self._input_pane.set_mode(_THREE_D_MODE)
        self._output_pane.set_mode(_THREE_D_MODE)
        self._z_slider.setEnabled(False)
        if self._reset_3d_on_next_render:
            input_camera_state = None
            output_camera_state = None
        else:
            input_camera_state = self._input_pane.camera_state()
            output_camera_state = self._output_pane.camera_state()

        input_stacks = self._build_3d_channel_payload(self._input_channels)
        self._input_pane.load_3d(
            input_stacks,
            method=self._active_volume_method,
            slider_val=self._volume_method_values[self._active_volume_method] / 100.0,
            interpolation="linear" if self._smooth_check.isChecked() else "nearest",
            downsample=[1, 2, 4][self._downsample_combo.currentIndex()],
            pixel_size_x=self._metadata.get("pixel_size_x"),
            pixel_size_z=self._metadata.get("pixel_size_z"),
            preserve_camera_state=input_camera_state,
        )

        preview = self._preview_by_t.get(timepoint)
        if preview:
            output_was_placeholder = self._output_pane._display_stack.currentIndex() == 1
            output_stacks = self._build_3d_channel_payload(preview)
            self._output_pane.load_3d(
                output_stacks,
                method=self._active_volume_method,
                slider_val=self._volume_method_values[self._active_volume_method] / 100.0,
                interpolation="linear" if self._smooth_check.isChecked() else "nearest",
                downsample=[1, 2, 4][self._downsample_combo.currentIndex()],
                pixel_size_x=self._metadata.get("pixel_size_x"),
                pixel_size_z=self._metadata.get("pixel_size_z"),
                preserve_camera_state=output_camera_state if not output_was_placeholder else input_camera_state,
            )
            if output_was_placeholder:
                self._sync_3d_from_input(self._input_pane.camera_state())
        else:
            self._output_pane.show_placeholder(
                f"No deconvolved preview for T={timepoint}.\nRun Deconvolution to preview this timepoint."
            )
        self._reset_3d_on_next_render = False

    def _build_2d_pixmap(
        self,
        channels_zyx: list[np.ndarray],
        projection: str,
    ) -> QPixmap:
        slices: list[tuple[np.ndarray, tuple[int, int, int], tuple[float, float]]] = []
        lo_pct = self._lo_spin.value()
        hi_pct = self._hi_spin.value()
        for idx in self._active_channel_indices():
            if idx >= len(channels_zyx):
                continue
            stack = channels_zyx[idx]
            plane = _project_stack(stack, projection, self._z_slider.value())
            contrast_src = stack if projection == "Slice" else plane
            lo = float(np.percentile(contrast_src, lo_pct))
            hi = float(np.percentile(contrast_src, hi_pct))
            slices.append((plane, self._channel_colors[idx], (lo, hi)))
        return _composite_to_pixmap(slices)

    def _build_3d_channel_payload(
        self,
        channels_zyx: list[np.ndarray],
    ) -> list[tuple[np.ndarray, tuple[int, int, int], tuple[float, float]]]:
        result: list[tuple[np.ndarray, tuple[int, int, int], tuple[float, float]]] = []
        lo_pct = self._lo_spin.value()
        hi_pct = self._hi_spin.value()
        for idx in self._active_channel_indices():
            if idx >= len(channels_zyx):
                continue
            stack = channels_zyx[idx]
            mid_z = min(stack.shape[0] // 2, stack.shape[0] - 1)
            lo = float(np.percentile(stack[mid_z], lo_pct))
            hi = float(np.percentile(stack[mid_z], hi_pct))
            result.append((stack, self._channel_colors[idx], (lo, hi)))
        return result

    def _reset_3d_views(self) -> None:
        self._input_pane.reset_camera()
        self._output_pane.reset_camera()
        self._sync_3d_from_input(self._input_pane.camera_state())

    def _sync_3d_from_input(self, state: object) -> None:
        if self._syncing_camera:
            return
        self._syncing_camera = True
        try:
            self._output_pane.apply_camera_state(state)
        finally:
            self._syncing_camera = False

    def _sync_3d_from_output(self, state: object) -> None:
        if self._syncing_camera:
            return
        self._syncing_camera = True
        try:
            self._input_pane.apply_camera_state(state)
        finally:
            self._syncing_camera = False


if _HAS_VISPY:

    class _ChannelColormap(BaseColormap):
        glsl_map = ""

        def __init__(self, rgb: tuple[int, int, int], *, translucent_boost: bool = False):
            r, g, b = rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0
            if translucent_boost:
                self.glsl_map = (
                    "vec4 channel_cmap(float t) {\n"
                    "    float c = clamp(pow(t, 0.55) * 1.18, 0.0, 1.0);\n"
                    "    float a = clamp(pow(t, 1.35) * 0.82, 0.0, 1.0);\n"
                    f"    return vec4({r:.4f} * c, {g:.4f} * c, {b:.4f} * c, a);\n"
                    "}\n"
                )
            else:
                self.glsl_map = (
                    "vec4 channel_cmap(float t) {\n"
                    f"    return vec4({r:.4f} * t, {g:.4f} * t, {b:.4f} * t, clamp(t * 1.2, 0.0, 1.0));\n"
                    "}\n"
                )
            super().__init__()
