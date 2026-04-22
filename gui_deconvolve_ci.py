"""gui_deconvolve_ci.py — Standalone PyQt6 GUI for CI deconvolution.

Provides a graphical interface to the ``ci_rl_deconvolve``,
``ci_sparse_hessian_deconvolve``, and ``ci_generate_psf`` functions from
``deconvolve_ci.py``. Supports
multi-channel 3-D OME-TIFF input with per-channel PSF generation,
side-by-side input/output viewing with a shared Z-slider, and
MIP / SUM projection toggle.

Usage:
    python gui_deconvolve_ci.py
"""

from __future__ import annotations

import gc
import json
import logging
import os
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

# Windows taskbar: set AppUserModelID so the taskbar shows our icon
if sys.platform == "win32":
    import ctypes
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
        "ci.gui_deconvolve_ci"
    )

from PyQt6.QtCore import Qt, QRectF, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QIcon, QImage, QPainter, QPixmap, QWheelEvent
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ci_dual_viewer import DualViewerWidget

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
ICON_PATH = SCRIPT_DIR / "icon.svg"
LAST_SETTINGS_PATH = SCRIPT_DIR / ".last_settings.json"
OMERO_SESSION_REUSE_S = 10 * 60

def _default_settings_dir() -> str:
    """Return Documents folder (Windows) or HOME (Linux) as default dir."""
    if sys.platform == "win32":
        docs = Path.home() / "Documents"
        if docs.is_dir():
            return str(docs)
    return str(Path.home())

# ---------------------------------------------------------------------------
# Channel colour helpers (same scheme as deconvolve.save_mip_png)
# ---------------------------------------------------------------------------

_FALLBACK_COLORS = [
    (0, 255, 0),      # Green
    (255, 0, 255),     # Magenta
    (0, 255, 255),     # Cyan
    (255, 0, 0),       # Red
    (0, 0, 255),       # Blue
    (255, 255, 0),     # Yellow
]

_BGRCYM = [
    (0, 0, 255),      # Blue
    (0, 255, 0),      # Green
    (255, 0, 0),      # Red
    (0, 255, 255),    # Cyan
    (255, 255, 0),    # Yellow
    (255, 0, 255),    # Magenta
]


def _emission_to_rgb(wavelength_nm: Optional[float]) -> tuple[int, int, int]:
    """Map an emission wavelength (nm) to an approximate RGB colour."""
    if wavelength_nm is None:
        return (255, 255, 255)
    wl = wavelength_nm
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


def _channel_color(metadata: dict, ch_idx: int) -> tuple[int, int, int]:
    """Determine display colour for a channel from metadata."""
    channels = metadata.get("channels", [])
    em = None
    if ch_idx < len(channels):
        em = channels[ch_idx].get("emission_wavelength")
    rgb = _emission_to_rgb(em)
    if rgb == (255, 255, 255):
        rgb = _FALLBACK_COLORS[ch_idx % len(_FALLBACK_COLORS)]
    return rgb


def _resolve_channel_colors(
    metadata: dict, n_ch: int
) -> list[tuple[int, int, int]]:
    """Return a list of display colours, de-duplicating if needed."""
    colors = [_channel_color(metadata, i) for i in range(n_ch)]
    if n_ch > 1 and len(set(colors)) == 1:
        colors = [_BGRCYM[i % len(_BGRCYM)] for i in range(n_ch)]
    return colors


# ---------------------------------------------------------------------------
# Zoomable image view (QGraphicsView with wheel-zoom and pan)
# ---------------------------------------------------------------------------

class ZoomableImageView(QGraphicsView):
    """QGraphicsView that supports mouse-wheel zoom and middle-button pan.

    Multiple views can be linked via ``link_to()`` so that zoom level
    and scroll position stay synchronised.
    """

    _ZOOM_FACTOR = 1.15

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item: QGraphicsPixmapItem | None = None
        self._linked: list[ZoomableImageView] = []
        self._syncing = False

        self.setRenderHints(
            self.renderHints()
            | QPainter.RenderHint.Antialiasing
            | QPainter.RenderHint.SmoothPixmapTransform
        )
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setMinimumSize(256, 256)
        self.setStyleSheet("background-color: #1a1a1a; border: none;")

    # --- linking ---
    def link_to(self, other: "ZoomableImageView"):
        """Bidirectionally link two views so zoom/pan stay in sync."""
        if other not in self._linked:
            self._linked.append(other)
        if self not in other._linked:
            other._linked.append(self)

    # --- public API ---
    def set_pixmap(self, pix: QPixmap | None):
        self._scene.clear()
        self._pixmap_item = None
        if pix is not None and not pix.isNull():
            self._pixmap_item = self._scene.addPixmap(pix)
            self._scene.setSceneRect(QRectF(pix.rect()))

    def clear(self):
        self._scene.clear()
        self._pixmap_item = None

    def fit_in_view(self):
        if self._pixmap_item is not None:
            self.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

    # --- zoom ---
    def wheelEvent(self, event: QWheelEvent):
        degrees = event.angleDelta().y() / 8
        steps = degrees / 15
        factor = self._ZOOM_FACTOR ** steps
        self.scale(factor, factor)
        self._sync_transform()

    def _sync_transform(self):
        if self._syncing:
            return
        self._syncing = True
        t = self.transform()
        cx = self.horizontalScrollBar().value()
        cy = self.verticalScrollBar().value()
        for v in self._linked:
            v._syncing = True
            v.setTransform(t)
            v.horizontalScrollBar().setValue(cx)
            v.verticalScrollBar().setValue(cy)
            v._syncing = False
        self._syncing = False

    def scrollContentsBy(self, dx, dy):
        super().scrollContentsBy(dx, dy)
        if not self._syncing:
            self._sync_transform()


# ---------------------------------------------------------------------------
# Helpers: numpy 2-D arrays → QPixmap
# ---------------------------------------------------------------------------

def _composite_to_pixmap(
    slices: list[tuple[np.ndarray, tuple[int, int, int], tuple[float, float]]],
    width: int = 0,
) -> QPixmap:
    """Build an RGB composite from (2-D array, colour, (lo, hi)) triples → QPixmap.

    The (lo, hi) pair should be the global min/max across the full 3-D
    volume so that contrast is consistent across Z-slices.
    """
    if not slices:
        return QPixmap()
    h, w_ = slices[0][0].shape
    canvas = np.zeros((h, w_, 3), dtype=np.float64)
    for arr, rgb, (lo, hi) in slices:
        ch_img = arr.astype(np.float64)
        if hi - lo > 0:
            ch_img = (ch_img - lo) / (hi - lo)
        else:
            ch_img = np.zeros_like(ch_img)
        for c in range(3):
            canvas[:, :, c] += ch_img * (rgb[c] / 255.0)
    canvas = np.clip(canvas, 0, 1)
    canvas = (canvas * 255).astype(np.uint8)
    canvas = np.ascontiguousarray(canvas)
    qimg = QImage(canvas.data, w_, h, w_ * 3, QImage.Format.Format_RGB888)
    pix = QPixmap.fromImage(qimg.copy())
    if width > 0:
        pix = pix.scaledToWidth(width, Qt.TransformationMode.SmoothTransformation)
    return pix


# ---------------------------------------------------------------------------
# Lightweight image loader (avoids importing deconvolve.py / torch)
# ---------------------------------------------------------------------------


def _apply_metadata_defaults(images: list, meta: dict) -> dict:
    """Track metadata source and apply defaults for missing fields."""
    meta_keys_from_file = set(meta.keys())
    ch_list = meta.get("channels", [])
    if ch_list:
        meta_keys_from_file.add("channels")
        if all("emission_wavelength" in c for c in ch_list):
            meta_keys_from_file.add("emission_wavelength")
        if all("excitation_wavelength" in c for c in ch_list):
            meta_keys_from_file.add("excitation_wavelength")

    meta.setdefault("na", 1.4)
    meta.setdefault("pixel_size_x", 0.065)
    meta.setdefault("pixel_size_z", 0.2)
    meta.setdefault("refractive_index", 1.515)
    meta.setdefault("microscope_type", "widefield")
    if "channels" not in meta:
        meta["channels"] = [{"emission_wavelength": 520.0}] * len(images)
    if "channel_names" not in meta:
        meta["channel_names"] = [f"Ch {i}" for i in range(len(images))]
    meta["n_channels"] = len(images)
    meta["_from_file"] = meta_keys_from_file
    return meta


def _normalize_channel_to_tzyx(channel: np.ndarray) -> np.ndarray:
    """Normalize supported image arrays to TZYX."""
    arr = np.asarray(channel, dtype=np.float32)
    if arr.ndim == 2:
        return arr[np.newaxis, np.newaxis, :, :]
    if arr.ndim == 3:
        return arr[np.newaxis, :, :, :]
    if arr.ndim == 4:
        return arr
    raise ValueError(f"Unsupported channel shape {arr.shape}; expected 2D, 3D, or 4D.")


def _normalize_image_bundle(images: list[np.ndarray], meta: dict) -> dict:
    """Normalize per-channel image volumes to TZYX and enrich metadata."""
    channels = [_normalize_channel_to_tzyx(img) for img in images]
    if not channels:
        raise ValueError("No image channels were loaded.")
    size_t, size_z, size_y, size_x = channels[0].shape
    for idx, channel in enumerate(channels[1:], start=1):
        if channel.shape != channels[0].shape:
            raise ValueError(
                f"Channel {idx} shape {channel.shape} does not match channel 0 shape {channels[0].shape}."
            )
    meta = _apply_metadata_defaults(channels, meta)
    meta["size_t"] = size_t
    meta["size_z"] = size_z
    meta["size_y"] = size_y
    meta["size_x"] = size_x
    meta["size_c"] = len(channels)
    meta["default_t"] = 0
    meta["default_z"] = size_z // 2 if size_z > 1 else 0
    if "channel_names" not in meta or len(meta["channel_names"]) < len(channels):
        existing = list(meta.get("channel_names", []))
        meta["channel_names"] = existing + [
            f"Ch {i}" for i in range(len(existing), len(channels))
        ]
    for idx, channel_meta in enumerate(meta.get("channels", [])):
        if "name" not in channel_meta and idx < len(meta["channel_names"]):
            channel_meta["name"] = meta["channel_names"][idx]
    return {"images": channels, "metadata": meta}


def _channel_stack_to_solver_input(channel_zyx: np.ndarray) -> np.ndarray:
    """Convert a normalized ZYX stack into 2D or 3D solver input."""
    volume = np.asarray(channel_zyx, dtype=np.float32)
    if volume.ndim != 3:
        raise ValueError(f"Expected ZYX input, got {volume.shape}.")
    if volume.shape[0] == 1:
        return volume[0]
    return volume


def _solver_output_to_zyx(result: np.ndarray) -> np.ndarray:
    """Normalize solver output back to ZYX for preview/export storage."""
    arr = np.asarray(result, dtype=np.float32)
    if arr.ndim == 2:
        return arr[np.newaxis, :, :]
    if arr.ndim == 3:
        return arr
    raise ValueError(f"Unexpected solver result shape {arr.shape}.")


def _current_channel_names(meta: dict, n_channels: int) -> list[str]:
    names = list(meta.get("channel_names") or [])
    if names:
        return names[:n_channels]
    channels = meta.get("channels", [])
    if channels:
        return [
            str(ch.get("name") or f"Ch{i} em={ch.get('emission_wavelength', '?')}")
            for i, ch in enumerate(channels[:n_channels])
        ]
    return [f"Ch{i}" for i in range(n_channels)]


def _extract_bioio_metadata(img, path_str: str) -> dict:
    _ACQ_MODE_MAP = {
        "LASER_SCANNING_CONFOCAL_MICROSCOPY": "confocal",
        "SPINNING_DISK_CONFOCAL": "confocal",
        "SLIT_SCAN_CONFOCAL": "confocal",
        "MULTI_PHOTON_MICROSCOPY": "confocal",
        "WIDE_FIELD": "widefield",
        "OTHER": "widefield",
    }
    _IMM_RI = {
        "OIL": 1.515,
        "WATER": 1.333,
        "GLYCEROL": 1.47,
        "AIR": 1.0,
        "MULTI": 1.515,
    }
    meta: dict = {}

    # Physical pixel sizes (µm)
    pps = img.physical_pixel_sizes
    if pps.X:
        meta["pixel_size_x"] = pps.X
    if pps.Z:
        meta["pixel_size_z"] = pps.Z

    # OME metadata (unified across formats via ome-types)
    try:
        ome = img.ome_metadata
        if ome and hasattr(ome, "images") and ome.images:
            im0 = ome.images[0]

            # Per-channel metadata
            ch_list = []
            for c in (im0.pixels.channels or []):
                ch_d: dict = {}
                if c.emission_wavelength is not None:
                    ch_d["emission_wavelength"] = float(c.emission_wavelength)
                if c.excitation_wavelength is not None:
                    ch_d["excitation_wavelength"] = float(c.excitation_wavelength)
                ch_list.append(ch_d)
                # Acquisition mode (use first channel's)
                if c.acquisition_mode and "microscope_type" not in meta:
                    name = getattr(c.acquisition_mode, "name",
                                   str(c.acquisition_mode))
                    meta["microscope_type"] = _ACQ_MODE_MAP.get(
                        name, "widefield")
            if ch_list:
                meta["channels"] = ch_list

            # Objective (NA, immersion → RI)
            if ome.instruments:
                for inst in ome.instruments:
                    for obj in (inst.objectives or []):
                        if obj.lens_na and "na" not in meta:
                            meta["na"] = float(obj.lens_na)
                        if obj.immersion and "refractive_index" not in meta:
                            imm_name = getattr(
                                obj.immersion, "name",
                                str(obj.immersion)).upper()
                            if imm_name in _IMM_RI:
                                meta["refractive_index"] = _IMM_RI[imm_name]

            # ObjectiveSettings — may contain explicit RI (overrides above)
            if hasattr(im0, "objective_settings") and im0.objective_settings:
                os_ = im0.objective_settings
                if os_.refractive_index:
                    meta["refractive_index"] = float(os_.refractive_index)
    except Exception:
        pass

    try:
        channel_names = list(img.channel_names or [])
        if channel_names:
            meta["channel_names"] = [str(name) for name in channel_names]
    except Exception:
        pass

    # Fallback: try to parse channel names as emission wavelengths
    # (e.g. OME-Zarr stores channel names like '520.0', '600.0')
    if "channels" not in meta:
        try:
            ch_names = img.channel_names or []
            ch_list = []
            for nm in ch_names:
                val = float(str(nm))
                if 300 < val < 900:
                    ch_list.append({"emission_wavelength": val})
            if ch_list:
                meta["channels"] = ch_list
        except (ValueError, TypeError):
            pass

    # ND2 fallback: native nd2 metadata for RI when ome_metadata lacks it
    ext = Path(path_str).suffix.lower()
    if ext == ".nd2" and "refractive_index" not in meta:
        try:
            import nd2
            with nd2.ND2File(str(path_str)) as f:
                chs = f.metadata.channels if f.metadata else []
                if chs and chs[0].microscope:
                    ri = chs[0].microscope.immersionRefractiveIndex
                    if ri is not None and ri > 0:
                        meta["refractive_index"] = ri
        except Exception:
            pass

    return meta


def _bioio_dim_size(img, dim_name: str, default: int = 1) -> int:
    try:
        value = getattr(img.dims, dim_name)
        if value is not None:
            return max(int(value), 1)
    except Exception:
        pass
    try:
        order = str(getattr(img.dims, "order", ""))
        shape = tuple(int(v) for v in getattr(img, "shape", ()))
        if dim_name in order and len(order) == len(shape):
            return max(int(shape[order.index(dim_name)]), 1)
    except Exception:
        pass
    return max(int(default), 1)


def _normalize_stack_to_zyx(arr: np.ndarray) -> np.ndarray:
    stack = np.asarray(arr, dtype=np.float32)
    if stack.ndim == 2:
        return stack[np.newaxis, :, :]
    if stack.ndim == 3:
        return stack
    raise ValueError(f"Unsupported stack shape {stack.shape}; expected YX or ZYX.")


def _parse_float_list(text: str) -> list[float]:
    values: list[float] = []
    for raw in text.split(","):
        token = raw.strip()
        if not token:
            continue
        if token.lower() in {"none", "nan", "null"}:
            continue
        try:
            values.append(float(token))
        except ValueError:
            continue
    return values


def _format_channel_values(
    channels: list[dict],
    key: str,
    default: float,
) -> str:
    values: list[str] = []
    for channel in channels:
        value = channel.get(key)
        try:
            values.append(str(float(value if value is not None else default)))
        except (TypeError, ValueError):
            values.append(str(float(default)))
    return ", ".join(values)


class _BaseTimepointSource:
    def __init__(self, metadata: dict):
        self.metadata = metadata

    def load_timepoint(
        self,
        t_index: int,
        progress_cb: Optional[Callable[[int, int, str], None]] = None,
    ) -> list[np.ndarray]:
        raise NotImplementedError


class _BioImageTimepointSource(_BaseTimepointSource):
    def __init__(self, path_str: str):
        from bioio import BioImage

        self._path_str = str(path_str)
        self._img = BioImage(self._path_str)
        meta = _extract_bioio_metadata(self._img, self._path_str)
        size_c = _bioio_dim_size(self._img, "C", default=1)
        meta = _apply_metadata_defaults([None] * size_c, meta)
        meta["size_t"] = _bioio_dim_size(self._img, "T", default=1)
        meta["size_z"] = _bioio_dim_size(self._img, "Z", default=1)
        meta["size_y"] = _bioio_dim_size(self._img, "Y", default=1)
        meta["size_x"] = _bioio_dim_size(self._img, "X", default=1)
        meta["size_c"] = size_c
        meta["default_t"] = 0
        meta["default_z"] = meta["size_z"] // 2 if meta["size_z"] > 1 else 0
        super().__init__(meta)

    def load_timepoint(
        self,
        t_index: int,
        progress_cb: Optional[Callable[[int, int, str], None]] = None,
    ) -> list[np.ndarray]:
        size_c = max(int(self.metadata.get("size_c", 1)), 1)
        size_t = max(int(self.metadata.get("size_t", 1)), 1)
        target_t = max(0, min(int(t_index), size_t - 1))
        channels: list[np.ndarray] = []
        total = size_c
        for c_index in range(size_c):
            if progress_cb is not None:
                progress_cb(c_index, total, f"Loading stack… channel {c_index + 1}/{total}")
            selector: dict[str, Any] = {}
            if size_t > 1:
                selector["T"] = target_t
            if size_c > 1:
                selector["C"] = c_index
            if hasattr(self._img, "get_image_dask_data"):
                arr = self._img.get_image_dask_data("ZYX", **selector).compute()
            else:
                arr = self._img.get_image_data("ZYX", **selector)
            channels.append(_normalize_stack_to_zyx(arr))
            if progress_cb is not None:
                progress_cb(c_index + 1, total, f"Loading stack… channel {c_index + 1}/{total}")
        return channels


class _OmeroTimepointSource(_BaseTimepointSource):
    def __init__(self, image):
        from omero_browser_qt import RegularImagePlaneProvider, get_image_metadata

        self._image = image
        self._provider = RegularImagePlaneProvider(image)
        meta = get_image_metadata(image)
        size_c = max(int(meta.get("size_c", 1)), 1)
        meta = _apply_metadata_defaults([None] * size_c, meta)
        meta["size_c"] = size_c
        meta["default_t"] = 0
        meta["default_z"] = max(int(meta.get("size_z", 1)) // 2, 0)
        super().__init__(meta)

    def load_timepoint(
        self,
        t_index: int,
        progress_cb: Optional[Callable[[int, int, str], None]] = None,
    ) -> list[np.ndarray]:
        size_c = max(int(self.metadata.get("size_c", 1)), 1)
        size_t = max(int(self.metadata.get("size_t", 1)), 1)
        size_z = max(int(self.metadata.get("size_z", 1)), 1)
        target_t = max(0, min(int(t_index), size_t - 1))
        channels: list[np.ndarray] = []
        total = max(size_c * size_z, 1)
        for c_index in range(size_c):
            base = c_index * size_z

            def _progress(done: int, channel_total: int, *, _c=c_index, _base=base) -> None:
                if progress_cb is None:
                    return
                progress_cb(
                    _base + done,
                    total,
                    f"Loading stack… channel {_c + 1}/{size_c}, plane {done}/{channel_total}",
                )

            stack = self._provider.get_stack(c_index, target_t, progress=_progress)
            channels.append(_normalize_stack_to_zyx(stack))
            if progress_cb is not None:
                progress_cb(base + size_z, total, f"Loading stack… channel {c_index + 1}/{size_c}")
        return channels


def _build_file_source(path_str: str) -> _BaseTimepointSource:
    return _BioImageTimepointSource(path_str)


def _build_omero_source(image) -> _BaseTimepointSource:
    return _OmeroTimepointSource(image)


# ---------------------------------------------------------------------------
# Live resource monitor (CPU / RAM / GPU / VRAM)
# ---------------------------------------------------------------------------

class _ResourceMonitor(QThread):
    """Background QThread that polls CPU/RAM/GPU metrics every 500 ms.

    Emits ``metrics_updated`` with a dict containing:
        cpu_pct, ram_used_gb, ram_total_gb, ram_swap_gb,
        gpu_pct, vram_used_gb, vram_total_gb, vram_spill_gb, has_gpu
    """

    metrics_updated = pyqtSignal(dict)
    _POLL_S = 0.5

    def __init__(self, parent=None):
        super().__init__(parent)
        self._stop = threading.Event()

        # psutil
        self._psutil = None
        self._proc = None
        try:
            import psutil as _ps
            self._psutil = _ps
            self._proc = _ps.Process()
        except ImportError:
            pass

        # pynvml
        self._pynvml = None
        self._nvml_handle = None
        try:
            import pynvml as _nv
            _nv.nvmlInit()
            self._nvml_handle = _nv.nvmlDeviceGetHandleByIndex(0)
            self._pynvml = _nv
        except Exception:
            pass

        # nvidia-smi fallback for GPU utilisation on Windows
        import shutil
        self._nvsmi = shutil.which("nvidia-smi")

        # torch GPU available?
        self._torch_cuda = False
        try:
            import torch
            self._torch_cuda = torch.cuda.is_available()
        except Exception:
            pass

    def run(self):
        self._stop.clear()
        # Prime cpu_percent (first call always returns 0)
        if self._proc:
            try:
                self._proc.cpu_percent()
            except Exception:
                pass
        while not self._stop.is_set():
            self.metrics_updated.emit(self._poll())
            self._stop.wait(self._POLL_S)

    def _poll(self) -> dict:
        m: dict = {
            "cpu_pct": 0.0,
            "ram_used_gb": 0.0,
            "ram_total_gb": 0.0,
            "ram_swap_gb": 0.0,
            "ram_swap_total_gb": 0.0,
            "gpu_pct": 0.0,
            "vram_used_gb": 0.0,
            "vram_total_gb": 0.0,
            "vram_spill_gb": 0.0,
            "has_gpu": False,
        }
        if self._psutil:
            try:
                # System-wide RAM (more meaningful than process RSS)
                vm = self._psutil.virtual_memory()
                m["ram_used_gb"] = vm.used / (1024 ** 3)
                m["ram_total_gb"] = vm.total / (1024 ** 3)
                # Swap / pagefile
                sw = self._psutil.swap_memory()
                m["ram_swap_gb"] = sw.used / (1024 ** 3)
                m["ram_swap_total_gb"] = sw.total / (1024 ** 3)
            except Exception:
                pass
            if self._proc:
                try:
                    m["cpu_pct"] = self._proc.cpu_percent()
                except Exception:
                    pass
        # --- GPU utilisation & VRAM ---
        gpu_pct_set = False

        # 1) Try pynvml (most reliable when available)
        if self._nvml_handle:
            try:
                util = self._pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
                m["gpu_pct"] = float(util.gpu)
                gpu_pct_set = True
                mi = self._pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
                m["vram_used_gb"] = mi.used / (1024 ** 3)
                m["vram_total_gb"] = mi.total / (1024 ** 3)
                m["has_gpu"] = True
            except Exception:
                pass

        # 2) nvidia-smi subprocess fallback for GPU % (works on Windows
        #    even when pynvml fails to report utilisation)
        if not gpu_pct_set and self._nvsmi:
            try:
                import subprocess
                out = subprocess.check_output(
                    [self._nvsmi,
                     "--query-gpu=utilization.gpu,memory.used,memory.total",
                     "--format=csv,noheader,nounits"],
                    timeout=2,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                parts = out.decode().strip().split(",")
                if len(parts) >= 3:
                    m["gpu_pct"] = float(parts[0].strip())
                    gpu_pct_set = True
                    m["vram_used_gb"] = float(parts[1].strip()) / 1024
                    m["vram_total_gb"] = float(parts[2].strip()) / 1024
                    m["has_gpu"] = True
            except Exception:
                pass

        # 3) torch.cuda fallback (VRAM only, no utilisation %)
        if self._torch_cuda:
            try:
                import torch
                if not m["has_gpu"]:
                    m["has_gpu"] = True
                    props = torch.cuda.get_device_properties(0)
                    m["vram_total_gb"] = props.total_memory / (1024 ** 3)
                    m["vram_used_gb"] = torch.cuda.memory_allocated() / (1024 ** 3)
                if not gpu_pct_set:
                    m["gpu_pct"] = -1.0  # sentinel: "unknown"

                # VRAM spill: torch reserved memory exceeds physical VRAM
                # (Windows spills into pagefile/shared GPU memory)
                reserved_gb = torch.cuda.memory_reserved() / (1024 ** 3)
                if m["vram_total_gb"] > 0 and reserved_gb > m["vram_total_gb"]:
                    m["vram_spill_gb"] = reserved_gb - m["vram_total_gb"]
            except Exception:
                pass

        return m

    def request_stop(self):
        """Signal the polling loop to stop and wait up to 2 s."""
        self._stop.set()
        self.wait(2000)


# ---------------------------------------------------------------------------
# Single-metric horizontal bar widget (label | progress | value text)
# ---------------------------------------------------------------------------

class _MetricWidget(QWidget):
    """Compact bar displaying one resource metric."""

    _COLORS = {"ok": "#4CAF50", "warn": "#FF9800", "crit": "#F44336"}

    def __init__(self, label: str, bar_width: int = 80, val_width: int = 88, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(1, 0, 1, 0)
        layout.setSpacing(2)

        lbl = QLabel(label)
        lbl.setFixedWidth(36)
        lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        f = lbl.font()
        f.setPointSize(8)
        lbl.setFont(f)
        layout.addWidget(lbl)

        self._bar = QProgressBar()
        self._bar.setRange(0, 100)
        self._bar.setValue(0)
        self._bar.setFixedWidth(bar_width)
        self._bar.setFixedHeight(13)
        self._bar.setTextVisible(False)
        self._bar.setStyleSheet(self._style("#4CAF50"))
        layout.addWidget(self._bar)

        self._val = QLabel("—")
        self._val.setFixedWidth(val_width)
        f2 = self._val.font()
        f2.setPointSize(8)
        self._val.setFont(f2)
        layout.addWidget(self._val)

    @staticmethod
    def _style(color: str) -> str:
        return (
            f"QProgressBar {{ border: 1px solid #555; border-radius: 3px; "
            f"background-color: #2a2a2a; }}"
            f"QProgressBar::chunk {{ background-color: {color}; border-radius: 2px; }}"
        )

    def set_value(self, pct: float, text: str):
        self._bar.setValue(int(min(max(pct, 0), 100)))
        self._val.setText(text)
        color = (
            self._COLORS["crit"] if pct >= 90
            else self._COLORS["warn"] if pct >= 70
            else self._COLORS["ok"]
        )
        self._bar.setStyleSheet(self._style(color))


# ---------------------------------------------------------------------------
# Composite status-bar panel showing all resource metrics
# ---------------------------------------------------------------------------

class ResourceMonitorBar(QWidget):
    """Status-bar widget with live CPU / RAM / GPU / VRAM bars."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 1, 2, 1)
        layout.setSpacing(2)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(sep)

        self._cpu_bar = _MetricWidget("CPU", bar_width=70, val_width=44)
        layout.addWidget(self._cpu_bar)

        self._ram_bar = _MetricWidget("RAM", bar_width=70, val_width=100)
        layout.addWidget(self._ram_bar)

        self._swap_bar = _MetricWidget("SWAP", bar_width=50, val_width=58)
        self._swap_bar.setToolTip("System swap / pagefile in use (RAM spillover)")
        layout.addWidget(self._swap_bar)

        # GPU / VRAM — always visible; show N/A when not available
        self._gpu_bar = _MetricWidget("GPU", bar_width=70, val_width=44)
        layout.addWidget(self._gpu_bar)

        self._vram_bar = _MetricWidget("VRAM", bar_width=70, val_width=100)
        layout.addWidget(self._vram_bar)

        self._vspill_bar = _MetricWidget("SPILL", bar_width=50, val_width=58)
        self._vspill_bar.setToolTip(
            "VRAM spillover — GPU memory reserved by PyTorch that exceeds\n"
            "physical VRAM (spills into system RAM / pagefile on Windows)"
        )
        layout.addWidget(self._vspill_bar)

        self._dot = QLabel("●")
        self._dot.setStyleSheet("color: #4CAF50; font-size: 9pt; padding: 0 2px;")
        self._dot.setToolTip("Deconvolution running")
        self._dot.setVisible(False)
        layout.addWidget(self._dot)

        self._has_gpu: bool = False

    def update_metrics(self, m: dict):
        """Slot — called from main thread via signal."""
        # CPU
        self._cpu_bar.set_value(m["cpu_pct"], f"{m['cpu_pct']:.0f}%")

        # System RAM
        if m["ram_total_gb"] > 0:
            ram_pct = m["ram_used_gb"] / m["ram_total_gb"] * 100
            self._ram_bar.set_value(
                ram_pct,
                f"{m['ram_used_gb']:.1f}/{m['ram_total_gb']:.0f} GB",
            )

        # RAM swap / pagefile (always visible)
        swap = m.get("ram_swap_gb", 0.0)
        swap_total = m.get("ram_swap_total_gb", 0.0)
        if swap_total > 0:
            swap_pct = swap / swap_total * 100
            self._swap_bar.set_value(swap_pct, f"{swap:.1f} GB")
        else:
            self._swap_bar.set_value(0, f"{swap:.1f} GB")

        # GPU / VRAM
        has_gpu = bool(m.get("has_gpu"))
        if has_gpu != self._has_gpu:
            self._has_gpu = has_gpu
        gpu_pct = m.get("gpu_pct", 0.0)
        if has_gpu:
            if gpu_pct < 0:
                self._gpu_bar.set_value(0, "N/A %")
            else:
                self._gpu_bar.set_value(gpu_pct, f"{gpu_pct:.0f}%")
            if m["vram_total_gb"] > 0:
                vram_pct = m["vram_used_gb"] / m["vram_total_gb"] * 100
                self._vram_bar.set_value(
                    vram_pct,
                    f"{m['vram_used_gb']:.1f}/{m['vram_total_gb']:.0f} GB",
                )
        else:
            self._gpu_bar.set_value(0, "no GPU")
            self._vram_bar.set_value(0, "no GPU")

        # VRAM spill (always visible) — max is half system RAM on Windows
        vspill = m.get("vram_spill_gb", 0.0)
        spill_max = m.get("ram_total_gb", 0) / 2
        if spill_max > 0:
            spill_pct = min(vspill / spill_max * 100, 100)
            self._vspill_bar.set_value(spill_pct, f"{vspill:.1f} GB")
        else:
            self._vspill_bar.set_value(0, f"{vspill:.1f} GB")

    def set_active(self, active: bool):
        """Show/hide the green activity dot (deconvolution running)."""
        self._dot.setVisible(active)


class CollapsibleSection(QWidget):
    """Simple inline section that can show or hide its child content."""

    def __init__(self, title: str, expanded: bool = False, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self._toggle = QToolButton()
        self._toggle.setText(title)
        self._toggle.setCheckable(True)
        self._toggle.setChecked(expanded)
        self._toggle.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        self._toggle.toggled.connect(self._on_toggled)
        layout.addWidget(self._toggle)

        self._content = QWidget()
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(0, 0, 0, 0)
        self._content_layout.setSpacing(6)
        layout.addWidget(self._content)

        self._on_toggled(expanded)

    def content_layout(self) -> QVBoxLayout:
        """Return the layout that should receive the collapsible content."""
        return self._content_layout

    def _on_toggled(self, expanded: bool):
        self._toggle.setArrowType(
            Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow
        )
        self._content.setVisible(expanded)


class NoWheelComboBox(QComboBox):
    """Combo box that ignores mouse-wheel changes."""

    def wheelEvent(self, event: QWheelEvent):
        event.ignore()


class NoWheelSpinBox(QSpinBox):
    """Spin box that ignores mouse-wheel changes."""

    def wheelEvent(self, event: QWheelEvent):
        event.ignore()


class NoWheelDoubleSpinBox(QDoubleSpinBox):
    """Double spin box that ignores mouse-wheel changes."""

    def wheelEvent(self, event: QWheelEvent):
        event.ignore()


# ---------------------------------------------------------------------------
# Worker thread for deconvolution
# ---------------------------------------------------------------------------

def _deconvolve_channel_stacks(
    channels_zyx: list[np.ndarray],
    metadata: dict,
    params: dict,
    t_index: int,
    *,
    progress_cb: Optional[Callable[[str], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> list[np.ndarray]:
    """Deconvolve one already-loaded timepoint and return per-channel ZYX output."""
    del metadata  # Metadata is carried indirectly via params and normalized shapes.

    def _progress(message: str) -> None:
        if progress_cb is not None:
            progress_cb(message)

    def _stopped() -> bool:
        return bool(should_stop and should_stop())

    try:
        from deconvolve_ci import (
            ci_generate_psf,
            ci_rl_deconvolve,
            ci_sparse_hessian_deconvolve,
        )
    except OSError as exc:
        raise RuntimeError(
            f"Failed to load deconvolve_ci (torch DLL error).\n\n"
            f"Your PyTorch installation appears broken. Try:\n"
            f"  conda install pytorch torchvision torchaudio "
            f"pytorch-cuda=12.1 -c pytorch -c nvidia\n\n"
            f"Original error: {exc}"
        ) from exc

    results: list[np.ndarray] = []
    n_channels = len(channels_zyx)
    for ci, channel_zyx in enumerate(channels_zyx):
        if _stopped():
            raise RuntimeError("Stopped by user")

        ch_data = _channel_stack_to_solver_input(channel_zyx)
        em_list = params["emission_wavelengths"]
        em_wl = em_list[ci] if ci < len(em_list) else em_list[-1] if em_list else 520.0
        ex_list = params["excitation_wavelengths"]
        ex_wl = ex_list[ci] if ci < len(ex_list) else ex_list[-1] if ex_list else None
        if params["microscope_type"] != "confocal":
            ex_wl = None

        if ch_data.ndim == 3:
            nz_img, _, _ = ch_data.shape
            n_z_psf = max(2 * nz_img - 1, 1) | 1
        else:
            n_z_psf = 1

        airy_radius_nm = 0.61 * em_wl / params["na"]
        airy_radius_px = airy_radius_nm / params["pixel_size_xy_nm"]
        n_xy_psf = int(max(64, 2 * int(4 * airy_radius_px) + 1))
        if n_xy_psf % 2 == 0:
            n_xy_psf += 1

        _progress(
            f"T {t_index + 1} — generating PSF for channel {ci + 1}/{n_channels} (λ={em_wl:.0f} nm)…"
        )
        psf = ci_generate_psf(
            na=params["na"],
            wavelength_nm=em_wl,
            pixel_size_xy_nm=params["pixel_size_xy_nm"],
            pixel_size_z_nm=params["pixel_size_z_nm"],
            n_xy=n_xy_psf,
            n_z=n_z_psf,
            ri_immersion=params["ri_immersion"],
            ri_sample=params["ri_sample"],
            ri_coverslip=params["ri_immersion"],
            ri_coverslip_design=params["ri_immersion"],
            ri_immersion_design=params["ri_immersion"],
            t_g=params["t_g"],
            t_g0=params["t_g0"],
            t_i0=params["t_i0"],
            z_p=params["z_p"],
            microscope_type=params["microscope_type"],
            excitation_nm=ex_wl,
            integrate_pixels=params["integrate_pixels"],
            n_subpixels=params["n_subpixels"],
            n_pupil=params["n_pupil"],
            device=params["device"],
        )

        if psf.ndim == 3 and ch_data.ndim == 3 and psf.shape[0] > ch_data.shape[0]:
            start = (psf.shape[0] - ch_data.shape[0]) // 2
            psf = psf[start:start + ch_data.shape[0]]

        gc.collect()
        try:
            import torch as _torch
            if _torch.cuda.is_available():
                _torch.cuda.empty_cache()
        except Exception:
            pass

        niter_list = params["niter_list"]
        niter = niter_list[ci] if ci < len(niter_list) else niter_list[-1]
        if _stopped():
            raise RuntimeError("Stopped by user")

        _progress(f"T {t_index + 1} — deconvolving channel {ci + 1}/{n_channels} ({niter} iter)…")
        common = dict(
            niter=niter,
            offset=params["offset"],
            prefilter_sigma=params["prefilter_sigma"],
            start=params["start"],
            background=params["background"],
            convergence=params["convergence"],
            rel_threshold=params["rel_threshold"],
            check_every=params["check_every"],
            pixel_size_xy=params["pixel_size_xy_nm"],
            pixel_size_z=params["pixel_size_z_nm"],
            device=params["device"],
            tiling=params["tiling"],
            max_tile_xy=params["max_tile_xy"],
            max_tile_z=params["max_tile_z"],
        )
        if params["method"] == "ci_sparse_hessian":
            out = ci_sparse_hessian_deconvolve(
                ch_data,
                psf,
                sparse_hessian_weight=params["sparse_hessian_weight"],
                sparse_hessian_reg=params["sparse_hessian_reg"],
                **common,
            )
        else:
            out = ci_rl_deconvolve(
                ch_data,
                psf,
                tv_lambda=params["tv_lambda"],
                damping=params["damping"],
                **common,
            )
        results.append(_solver_output_to_zyx(out["result"].copy()))

        del psf
        del out
        gc.collect()
        try:
            import torch as _torch
            if _torch.cuda.is_available():
                _torch.cuda.empty_cache()
        except Exception:
            pass

    return results


def _write_ome_tiff(data: np.ndarray, path: str, metadata: dict) -> None:
    from bioio.writers import OmeTiffWriter

    physical_pixel_sizes = None
    px_x = metadata.get("pixel_size_x")
    px_z = metadata.get("pixel_size_z")
    if px_x or px_z:
        from bioio_base.types import PhysicalPixelSizes
        physical_pixel_sizes = PhysicalPixelSizes(
            Z=px_z or 1.0, Y=px_x or 1.0, X=px_x or 1.0
        )

    channel_names = _current_channel_names(metadata, data.shape[1])
    OmeTiffWriter.save(
        data.astype(np.float32),
        path,
        dim_order="TCZYX",
        physical_pixel_sizes=physical_pixel_sizes,
        channel_names=channel_names,
    )


class _DeconvolveWorker(QThread):
    """Preview deconvolution for a single timepoint."""

    finished = pyqtSignal(object)
    progress = pyqtSignal(str)

    def __init__(
        self,
        channels: list[np.ndarray],
        metadata: dict,
        params: dict,
        t_index: int,
        parent=None,
    ):
        super().__init__(parent)
        self.channels = channels
        self.metadata = metadata
        self.params = params
        self.t_index = int(t_index)

    def run(self):
        try:
            results = _deconvolve_channel_stacks(
                self.channels,
                self.metadata,
                self.params,
                self.t_index,
                progress_cb=self.progress.emit,
                should_stop=self.isInterruptionRequested,
            )
            self.finished.emit({"timepoint": self.t_index, "channels": results})
        except Exception as exc:
            traceback.print_exc()
            self.finished.emit(exc)


class _SaveTSeriesWorker(QThread):
    """Export a full deconvolved T-series to OME-TIFF."""

    finished = pyqtSignal(object)
    progress = pyqtSignal(str)

    def __init__(
        self,
        source: _BaseTimepointSource,
        metadata: dict,
        params: dict,
        output_path: str,
        parent=None,
    ):
        super().__init__(parent)
        self.source = source
        self.metadata = metadata
        self.params = params
        self.output_path = output_path

    def run(self):
        stage_path = None
        try:
            size_t = max(int(self.metadata.get("size_t", 1)), 1)
            size_c = max(int(self.metadata.get("size_c", 1)), 1)
            size_z = max(int(self.metadata.get("size_z", 1)), 1)
            size_y = max(int(self.metadata.get("size_y", 1)), 1)
            size_x = max(int(self.metadata.get("size_x", 1)), 1)

            fd, stage_path = tempfile.mkstemp(prefix="cideconvolve_", suffix=".dat")
            os.close(fd)
            staged = np.memmap(
                stage_path,
                dtype=np.float32,
                mode="w+",
                shape=(size_t, size_c, size_z, size_y, size_x),
            )

            for t_index in range(size_t):
                if self.isInterruptionRequested():
                    raise RuntimeError("Stopped by user")
                self.progress.emit(f"Saving T-series — loading timepoint {t_index + 1}/{size_t}…")
                channels = self.source.load_timepoint(
                    t_index,
                    progress_cb=lambda done, total, text: self.progress.emit(
                        f"{text} ({done}/{total})"
                    ),
                )
                self.progress.emit(f"Saving T-series — processing timepoint {t_index + 1}/{size_t}…")
                results = _deconvolve_channel_stacks(
                    channels,
                    self.metadata,
                    self.params,
                    t_index,
                    progress_cb=self.progress.emit,
                    should_stop=self.isInterruptionRequested,
                )
                for c_index, result in enumerate(results):
                    staged[t_index, c_index, :, :, :] = result
                staged.flush()

            self.progress.emit("Writing OME-TIFF…")
            _write_ome_tiff(staged, self.output_path, self.metadata)
            self.finished.emit({"path": self.output_path})
        except Exception as exc:
            traceback.print_exc()
            self.finished.emit(exc)
        finally:
            if stage_path and os.path.exists(stage_path):
                try:
                    os.remove(stage_path)
                except OSError:
                    pass


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

def _detect_gpu_info() -> str:
    """Return a short string with torch + GPU version info for the title bar."""
    try:
        import torch
        parts = [f"torch {torch.__version__}"]
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            cuda_ver = torch.version.cuda or "?"
            parts.append(f"{name}  CUDA {cuda_ver}")
        else:
            parts.append("CPU only")
        return "  |  ".join(parts)
    except Exception:
        return "torch not available"


class DeconvolveCIWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        gpu_info = _detect_gpu_info()
        self.setWindowTitle(f"CI Deconvolve — {gpu_info}")
        if ICON_PATH.exists():
            self.setWindowIcon(QIcon(str(ICON_PATH)))
        self.setMinimumSize(1430, 700)

        # State
        self._input_channels: list[np.ndarray] = []
        self._input_source: Optional[_BaseTimepointSource] = None
        self._input_source_factory: Optional[Callable[[], _BaseTimepointSource]] = None
        self._loaded_timepoint: Optional[int] = None
        self._preview_outputs_by_t: dict[int, list[np.ndarray]] = {}
        self._metadata: dict = {}
        self._worker: Optional[_DeconvolveWorker] = None
        self._save_worker: Optional[_SaveTSeriesWorker] = None
        self._monitor: Optional[_ResourceMonitor] = None
        self._input_path: Optional[Path] = None
        self._last_open_dir: str = _default_settings_dir()
        self._last_zarr_dir: str = _default_settings_dir()
        self._last_save_dir: str = _default_settings_dir()
        self._last_settings_dir: str = _default_settings_dir()
        self._omero_gw = None  # OmeroGateway instance (lazy)
        self._omero_session_deadline: float = 0.0

        self._build_ui()

        # Start the resource monitor immediately and keep it running
        self._monitor = _ResourceMonitor(parent=self)
        self._monitor.metrics_updated.connect(self._monitor_bar.update_metrics)
        self._monitor.start()

    # -----------------------------------------------------------------------
    # UI construction
    # -----------------------------------------------------------------------

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # ---- Left: controls (scrollable) ----
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(440)
        scroll.setMaximumWidth(500)
        ctrl_widget = QWidget()
        ctrl_layout = QVBoxLayout(ctrl_widget)
        ctrl_layout.setSpacing(6)
        scroll.setWidget(ctrl_widget)
        splitter.addWidget(scroll)

        # Title
        title = QLabel("CI Deconvolve")
        title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        ctrl_layout.addWidget(title)

        # --- Method ---
        method_group = QGroupBox("Method")
        ml = QFormLayout()
        method_group.setLayout(ml)

        self._method_combo = NoWheelComboBox()
        self._method_combo.addItems(["ci_rl", "ci_rl_tv", "ci_sparse_hessian"])
        self._method_combo.currentTextChanged.connect(self._on_method_changed)
        ml.addRow("Method:", self._method_combo)

        self._le_niter = QLineEdit("150")
        self._le_niter.setToolTip(
            "Iterations per channel, comma-separated.\n"
            "E.g. '50' for all channels, or '50, 80' for ch1=50, ch2=80."
        )
        ml.addRow("Iterations:", self._le_niter)

        self._conv_combo = NoWheelComboBox()
        self._conv_combo.addItems(["auto", "fixed"])
        ml.addRow("Convergence:", self._conv_combo)

        self._sp_rel_thresh = NoWheelDoubleSpinBox()
        self._sp_rel_thresh.setRange(1e-8, 1.0)
        self._sp_rel_thresh.setDecimals(6)
        self._sp_rel_thresh.setSingleStep(0.0001)
        self._sp_rel_thresh.setValue(0.001)
        ml.addRow("Rel. threshold:", self._sp_rel_thresh)

        ctrl_layout.addWidget(method_group)

        advanced_section = CollapsibleSection("Advanced Parameters", expanded=False)
        advanced_layout = advanced_section.content_layout()

        # --- Tiling ---
        tiling_group = QGroupBox("Tiling")
        tl = QFormLayout()
        tiling_group.setLayout(tl)

        self._tiling_combo = NoWheelComboBox()
        self._tiling_combo.addItems(["custom", "none"])
        self._tiling_combo.currentTextChanged.connect(self._on_tiling_changed)
        tl.addRow("Tiling:", self._tiling_combo)

        self._sp_max_tile_xy = NoWheelSpinBox()
        self._sp_max_tile_xy.setRange(64, 4096)
        self._sp_max_tile_xy.setSingleStep(64)
        self._sp_max_tile_xy.setValue(512)
        tl.addRow("Max tile XY:", self._sp_max_tile_xy)

        self._sp_max_tile_z = NoWheelSpinBox()
        self._sp_max_tile_z.setRange(8, 512)
        self._sp_max_tile_z.setSingleStep(8)
        self._sp_max_tile_z.setValue(64)
        tl.addRow("Max tile Z:", self._sp_max_tile_z)

        advanced_layout.addWidget(tiling_group)

        # --- Method tuning ---
        method_adv_group = QGroupBox("Method Tuning")
        aml = QFormLayout()
        method_adv_group.setLayout(aml)

        self._sp_tv_lambda = NoWheelDoubleSpinBox()
        self._sp_tv_lambda.setRange(0.0, 1.0)
        self._sp_tv_lambda.setDecimals(6)
        self._sp_tv_lambda.setSingleStep(0.0001)
        self._sp_tv_lambda.setValue(0.0001)
        aml.addRow("TV lambda:", self._sp_tv_lambda)
        self._tv_lambda_label = aml.labelForField(self._sp_tv_lambda)  # type: ignore

        self._damping_combo = NoWheelComboBox()
        self._damping_combo.addItems(["none", "auto", "manual"])
        self._damping_combo.currentTextChanged.connect(self._on_damping_changed)
        aml.addRow("Damping:", self._damping_combo)
        self._damping_label = aml.labelForField(self._damping_combo)  # type: ignore

        self._sp_damping = NoWheelDoubleSpinBox()
        self._sp_damping.setRange(0.0, 100.0)
        self._sp_damping.setDecimals(2)
        self._sp_damping.setSingleStep(0.1)
        self._sp_damping.setValue(3.0)
        self._sp_damping.setEnabled(False)
        aml.addRow("Damping value:", self._sp_damping)
        self._damping_value_label = aml.labelForField(self._sp_damping)  # type: ignore

        self._sp_sparse_weight = NoWheelDoubleSpinBox()
        self._sp_sparse_weight.setRange(0.0, 1.0)
        self._sp_sparse_weight.setDecimals(4)
        self._sp_sparse_weight.setSingleStep(0.01)
        self._sp_sparse_weight.setValue(0.6)
        aml.addRow("Sparse weight:", self._sp_sparse_weight)
        self._sparse_weight_label = aml.labelForField(self._sp_sparse_weight)  # type: ignore

        self._sp_sparse_reg = NoWheelDoubleSpinBox()
        self._sp_sparse_reg.setRange(0.0, 1.0)
        self._sp_sparse_reg.setDecimals(4)
        self._sp_sparse_reg.setSingleStep(0.001)
        self._sp_sparse_reg.setValue(0.98)
        aml.addRow("Sparse reg:", self._sp_sparse_reg)
        self._sparse_reg_label = aml.labelForField(self._sp_sparse_reg)  # type: ignore

        self._bg_combo = NoWheelComboBox()
        self._bg_combo.addItems(["auto", "manual"])
        self._bg_combo.currentTextChanged.connect(self._on_bg_changed)
        aml.addRow("Background:", self._bg_combo)

        self._sp_bg_value = NoWheelDoubleSpinBox()
        self._sp_bg_value.setRange(0.0, 1e9)
        self._sp_bg_value.setDecimals(2)
        self._sp_bg_value.setValue(0.0)
        self._sp_bg_value.setEnabled(False)
        aml.addRow("BG value:", self._sp_bg_value)

        self._offset_combo = NoWheelComboBox()
        self._offset_combo.addItems(["auto", "none", "manual"])
        self._offset_combo.currentTextChanged.connect(self._on_offset_changed)
        aml.addRow("Offset:", self._offset_combo)

        self._sp_offset = NoWheelDoubleSpinBox()
        self._sp_offset.setRange(0.0, 1000.0)
        self._sp_offset.setDecimals(1)
        self._sp_offset.setSingleStep(1.0)
        self._sp_offset.setValue(5.0)
        self._sp_offset.setEnabled(False)
        aml.addRow("Offset value:", self._sp_offset)

        self._sp_prefilter = NoWheelDoubleSpinBox()
        self._sp_prefilter.setRange(0.0, 5.0)
        self._sp_prefilter.setDecimals(2)
        self._sp_prefilter.setSingleStep(0.1)
        self._sp_prefilter.setValue(0.0)
        aml.addRow("Prefilter sigma:", self._sp_prefilter)

        self._start_combo = NoWheelComboBox()
        self._start_combo.addItems(["flat", "observed", "lowpass"])
        aml.addRow("Start:", self._start_combo)

        self._sp_check_every = NoWheelSpinBox()
        self._sp_check_every.setRange(1, 1000)
        self._sp_check_every.setValue(5)
        self._sp_check_every.setEnabled(False)
        aml.addRow("Check every:", self._sp_check_every)
        self._conv_combo.currentTextChanged.connect(self._on_conv_changed)

        self._device_combo = NoWheelComboBox()
        self._device_combo.addItems(["auto", "cuda", "cpu"])
        aml.addRow("Device:", self._device_combo)

        advanced_layout.addWidget(method_adv_group)

        # --- Optics / PSF ---
        optics_group = QGroupBox("Optics / PSF")
        ol = QFormLayout()
        optics_group.setLayout(ol)

        self._sp_na = NoWheelDoubleSpinBox()
        self._sp_na.setRange(0.1, 2.0)
        self._sp_na.setDecimals(3)
        self._sp_na.setSingleStep(0.05)
        self._sp_na.setValue(1.4)
        ol.addRow("NA:", self._sp_na)

        self._le_emission = QLineEdit("520")
        self._le_emission.setToolTip(
            "Emission wavelength(s) in nm, comma-separated per channel."
        )
        ol.addRow("Emission (nm):", self._le_emission)

        self._sp_px_xy = NoWheelDoubleSpinBox()
        self._sp_px_xy.setRange(1.0, 10000.0)
        self._sp_px_xy.setDecimals(3)
        self._sp_px_xy.setSingleStep(1.0)
        self._sp_px_xy.setValue(65.0)
        ol.addRow("Pixel XY (nm):", self._sp_px_xy)

        self._sp_px_z = NoWheelDoubleSpinBox()
        self._sp_px_z.setRange(1.0, 50000.0)
        self._sp_px_z.setDecimals(3)
        self._sp_px_z.setSingleStep(10.0)
        self._sp_px_z.setValue(200.0)
        ol.addRow("Pixel Z (nm):", self._sp_px_z)

        self._micro_combo = NoWheelComboBox()
        self._micro_combo.addItems(["widefield", "confocal"])
        self._micro_combo.currentTextChanged.connect(self._on_micro_changed)
        ol.addRow("Microscope:", self._micro_combo)

        self._le_excitation = QLineEdit("488")
        self._le_excitation.setToolTip(
            "Excitation wavelength(s) in nm, comma-separated per channel.\n"
            "Used only for confocal PSF generation."
        )
        self._le_excitation.setEnabled(False)
        ol.addRow("Excitation (nm):", self._le_excitation)

        ctrl_layout.addWidget(optics_group)

        # --- Refractive Indices ---
        ri_group = QGroupBox("Refractive Indices")
        rl = QFormLayout()
        ri_group.setLayout(rl)

        self._sp_ri_imm = NoWheelDoubleSpinBox()
        self._sp_ri_imm.setRange(1.0, 2.0)
        self._sp_ri_imm.setDecimals(4)
        self._sp_ri_imm.setSingleStep(0.001)
        self._sp_ri_imm.setValue(1.515)
        rl.addRow("RI immersion:", self._sp_ri_imm)

        # Embedding / mounting medium combo
        _MEDIUM_RI = {
            "water (1.333)": 1.333,
            "PBS (1.334)": 1.334,
            "culture medium (1.337)": 1.337,
            "vectashield (1.45)": 1.45,
            "prolong gold (1.47)": 1.47,
            "glycerol (1.474)": 1.474,
            "oil (1.515)": 1.515,
            "prolong glass (1.52)": 1.52,
        }
        self._medium_ri_map = _MEDIUM_RI
        self._medium_combo = NoWheelComboBox()
        self._medium_combo.addItems(list(_MEDIUM_RI.keys()))
        self._medium_combo.setCurrentText("prolong gold (1.47)")
        self._medium_combo.currentTextChanged.connect(self._on_medium_changed)
        rl.addRow("Emb. medium:", self._medium_combo)

        self._sp_ri_sample = NoWheelDoubleSpinBox()
        self._sp_ri_sample.setRange(1.0, 2.0)
        self._sp_ri_sample.setDecimals(4)
        self._sp_ri_sample.setSingleStep(0.001)
        self._sp_ri_sample.setValue(1.47)
        rl.addRow("RI sample:", self._sp_ri_sample)



        ctrl_layout.addWidget(ri_group)

        # --- Coverslip / depths ---
        cov_group = QGroupBox("Coverslip / Depth")
        cl = QFormLayout()
        cov_group.setLayout(cl)

        self._sp_tg = NoWheelDoubleSpinBox()
        self._sp_tg.setRange(0, 1e7)
        self._sp_tg.setDecimals(0)
        self._sp_tg.setSingleStep(1000)
        self._sp_tg.setValue(170000)
        self._sp_tg.setSuffix(" nm")
        cl.addRow("Coverslip thickness:", self._sp_tg)

        self._sp_tg0 = NoWheelDoubleSpinBox()
        self._sp_tg0.setRange(0, 1e7)
        self._sp_tg0.setDecimals(0)
        self._sp_tg0.setSingleStep(1000)
        self._sp_tg0.setValue(170000)
        self._sp_tg0.setSuffix(" nm")
        cl.addRow("Coverslip thickness (design):", self._sp_tg0)

        self._sp_ti0 = NoWheelDoubleSpinBox()
        self._sp_ti0.setRange(0, 1e7)
        self._sp_ti0.setDecimals(0)
        self._sp_ti0.setSingleStep(1000)
        self._sp_ti0.setValue(100000)
        self._sp_ti0.setSuffix(" nm")
        cl.addRow("Immersion thickness (design):", self._sp_ti0)

        self._sp_zp = NoWheelDoubleSpinBox()
        self._sp_zp.setRange(0, 1e7)
        self._sp_zp.setDecimals(0)
        self._sp_zp.setSingleStep(100)
        self._sp_zp.setValue(0)
        self._sp_zp.setSuffix(" nm")
        cl.addRow("Particle depth (z_p):", self._sp_zp)

        advanced_layout.addWidget(cov_group)

        # --- PSF advanced ---
        psf_group = QGroupBox("PSF Advanced")
        pl = QFormLayout()
        psf_group.setLayout(pl)

        self._cb_integrate = QCheckBox()
        self._cb_integrate.setChecked(True)
        pl.addRow("Pixel integration:", self._cb_integrate)

        self._sp_subpixels = NoWheelSpinBox()
        self._sp_subpixels.setRange(1, 9)
        self._sp_subpixels.setValue(3)
        pl.addRow("Sub-pixels:", self._sp_subpixels)

        self._sp_n_pupil = NoWheelSpinBox()
        self._sp_n_pupil.setRange(33, 513)
        self._sp_n_pupil.setSingleStep(2)
        self._sp_n_pupil.setValue(129)
        pl.addRow("Pupil samples:", self._sp_n_pupil)

        advanced_layout.addWidget(psf_group)
        advanced_layout.addStretch()
        ctrl_layout.addWidget(advanced_section)

        ctrl_layout.addStretch()

        # ---- Right: image viewer ----
        viewer = QWidget()
        vl = QVBoxLayout(viewer)
        vl.setContentsMargins(0, 0, 0, 0)
        self._viewer = DualViewerWidget()
        self._viewer.timepointChanged.connect(self._on_viewer_time_changed)
        vl.addWidget(self._viewer, stretch=1)
        splitter.addWidget(viewer)

        # --- Top toolbar: Open / Run / Save ---
        bottom = QHBoxLayout()
        bottom.setContentsMargins(0, 0, 0, 4)

        btn_open = QPushButton("Open\u2026")
        btn_open.clicked.connect(self._on_open)
        bottom.addWidget(btn_open)

        btn_open_zarr = QPushButton("Open Zarr\u2026")
        btn_open_zarr.clicked.connect(self._on_open_zarr)
        bottom.addWidget(btn_open_zarr)

        btn_open_omero = QPushButton("Open OMERO\u2026")
        btn_open_omero.clicked.connect(self._on_open_omero)
        bottom.addWidget(btn_open_omero)

        self._file_label = QLabel("No file loaded")
        self._file_label.setWordWrap(False)
        bottom.addWidget(self._file_label, stretch=1)

        self._btn_run = QPushButton("Run Deconvolution")
        self._btn_run.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; "
            "font-weight: bold; padding: 8px; }"
        )
        self._btn_run.setEnabled(False)
        self._btn_run.clicked.connect(self._on_run)
        bottom.addWidget(self._btn_run)

        self._btn_save = QPushButton("Save\u2026")
        self._btn_save.setEnabled(False)
        self._btn_save.clicked.connect(self._on_save)
        bottom.addWidget(self._btn_save)

        self._btn_save_series = QPushButton("Save T-Series\u2026")
        self._btn_save_series.setEnabled(False)
        self._btn_save_series.setVisible(False)
        self._btn_save_series.clicked.connect(self._on_save_t_series)
        bottom.addWidget(self._btn_save_series)

        # --- Settings buttons ---
        self._btn_restore = QPushButton("Restore")
        self._btn_restore.setToolTip("Restore parameter values from the previous run")
        self._btn_restore.setEnabled(LAST_SETTINGS_PATH.exists())
        self._btn_restore.clicked.connect(self._on_restore_settings)
        bottom.addWidget(self._btn_restore)

        btn_save_settings = QPushButton("Save Settings\u2026")
        btn_save_settings.setToolTip("Save current parameter values to a JSON file")
        btn_save_settings.clicked.connect(self._on_save_settings)
        bottom.addWidget(btn_save_settings)

        btn_load_settings = QPushButton("Load Settings\u2026")
        btn_load_settings.setToolTip("Load parameter values from a JSON file")
        btn_load_settings.clicked.connect(self._on_load_settings)
        bottom.addWidget(btn_load_settings)

        self._progress = QProgressBar()
        self._progress.setRange(0, 0)  # indeterminate
        self._progress.setVisible(False)
        self._progress.setMaximumWidth(120)
        bottom.addWidget(self._progress)

        root.insertLayout(0, bottom)
        root.addWidget(splitter, stretch=1)

        # Status bar
        self._status = QStatusBar()
        self.setStatusBar(self._status)

        # Resource monitor bar (permanent widget on the right side)
        self._monitor_bar = ResourceMonitorBar()
        self._status.addPermanentWidget(self._monitor_bar)

        # Splitter proportions
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        # Initial method state
        self._on_method_changed(self._method_combo.currentText())

    # -----------------------------------------------------------------------
    # Slots — control panel
    # -----------------------------------------------------------------------

    def _on_method_changed(self, text: str):
        is_rl_family = text in ("ci_rl", "ci_rl_tv")
        is_tv = text == "ci_rl_tv"
        is_sparse = text == "ci_sparse_hessian"
        self._sp_tv_lambda.setEnabled(is_tv)
        if not is_tv:
            self._sp_tv_lambda.setValue(0.0)
        else:
            if self._sp_tv_lambda.value() == 0.0:
                self._sp_tv_lambda.setValue(0.0001)
        for label, widget in (
            (self._tv_lambda_label, self._sp_tv_lambda),
            (self._damping_label, self._damping_combo),
            (self._damping_value_label, self._sp_damping),
            (self._sparse_weight_label, self._sp_sparse_weight),
            (self._sparse_reg_label, self._sp_sparse_reg),
        ):
            if label is not None:
                label.setVisible(False)
            widget.setVisible(False)

        if self._tv_lambda_label is not None:
            self._tv_lambda_label.setVisible(is_tv)
        self._sp_tv_lambda.setVisible(is_tv)

        if self._damping_label is not None:
            self._damping_label.setVisible(is_rl_family)
        self._damping_combo.setVisible(is_rl_family)
        if self._damping_value_label is not None:
            self._damping_value_label.setVisible(is_rl_family)
        self._sp_damping.setVisible(is_rl_family)

        if self._sparse_weight_label is not None:
            self._sparse_weight_label.setVisible(is_sparse)
        self._sp_sparse_weight.setVisible(is_sparse)
        if self._sparse_reg_label is not None:
            self._sparse_reg_label.setVisible(is_sparse)
        self._sp_sparse_reg.setVisible(is_sparse)

        self._on_damping_changed(self._damping_combo.currentText())

    def _on_bg_changed(self, text: str):
        self._sp_bg_value.setEnabled(text == "manual")

    def _on_offset_changed(self, text: str):
        self._sp_offset.setEnabled(text == "manual")

    def _on_damping_changed(self, text: str):
        rl_family = self._method_combo.currentText() in ("ci_rl", "ci_rl_tv")
        self._sp_damping.setEnabled(rl_family and text == "manual")

    def _on_conv_changed(self, text: str):
        auto = text == "auto"
        self._sp_rel_thresh.setEnabled(auto)
        self._sp_check_every.setEnabled(auto)

    def _on_micro_changed(self, text: str):
        self._le_excitation.setEnabled(text == "confocal")
        # Auto-set iterations: 50 for confocal, 150 for widefield
        if text == "confocal":
            self._le_niter.setText("50")
        else:
            self._le_niter.setText("150")

    def _on_medium_changed(self, text: str):
        """Set RI sample spinbox from embedding medium combo selection."""
        ri = self._medium_ri_map.get(text)
        if ri is not None:
            self._sp_ri_sample.setValue(ri)

    def _on_tiling_changed(self, text: str):
        enabled = text == "custom"
        self._sp_max_tile_xy.setEnabled(enabled)
        self._sp_max_tile_z.setEnabled(enabled)

    def _on_proj_changed(self, text: str):
        del text

    def _begin_progress(self, total: int, text: str) -> None:
        self._progress.setRange(0, max(int(total), 1))
        self._progress.setValue(0)
        self._progress.setFormat("%v / %m")
        self._progress.setVisible(True)
        self._status.showMessage(text)
        QApplication.processEvents()

    def _begin_busy_progress(self, text: str) -> None:
        self._progress.setRange(0, 0)
        self._progress.setVisible(True)
        self._status.showMessage(text)
        QApplication.processEvents()

    def _advance_progress(self, value: int, text: Optional[str] = None) -> None:
        if self._progress.maximum() > 0:
            self._progress.setValue(max(0, min(int(value), self._progress.maximum())))
        if text is not None:
            self._status.showMessage(text)
        QApplication.processEvents()

    def _end_progress(self) -> None:
        self._end_progress()

    # -----------------------------------------------------------------------
    # File open
    # -----------------------------------------------------------------------

    def _on_open(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image",
            self._last_open_dir,
            "Images (*.ome.tiff *.ome.tif *.tiff *.tif *.nd2 *.czi);;All Files (*)",
        )
        if path:
            self._last_open_dir = str(Path(path).parent)
            self._do_load(path)

    def _on_open_zarr(self):
        path = QFileDialog.getExistingDirectory(
            self, "Open OME-Zarr Folder", self._last_zarr_dir,
        )
        if path:
            self._last_zarr_dir = str(Path(path).parent)
            self._do_load(path)

    def _do_load(self, path: str):
        try:
            self._begin_busy_progress(f"Opening {Path(path).name} …")
            source = _build_file_source(path)
            self._apply_image_source(
                source,
                Path(path).name,
                source_path=Path(path),
                source_factory=lambda _path=str(path): _build_file_source(_path),
            )
        except Exception as exc:
            QMessageBox.critical(self, "Load Error", str(exc))
            self._status.showMessage("Load failed", 5000)
        finally:
            self._end_progress()

    def _apply_image_source(
        self,
        source: _BaseTimepointSource,
        display_name: str,
        source_path: Optional[Path] = None,
        source_factory: Optional[Callable[[], _BaseTimepointSource]] = None,
    ):
        """Apply a lazy timepoint source to the UI (shared by file and OMERO open)."""
        self._input_source = source
        self._input_source_factory = source_factory
        self._input_channels = []
        self._loaded_timepoint = None
        self._metadata = dict(source.metadata)
        self._preview_outputs_by_t.clear()
        self._input_path = source_path

        # Populate UI from metadata
        meta = self._metadata
        from_file = meta.get("_from_file", set())

        def _bg(found: bool) -> str:
            """Stylesheet snippet: green if from metadata, orange if default."""
            if found:
                return "background-color: #c8e6c9; color: black;"   # soft green
            return "background-color: #ffe0b2; color: black;"       # soft orange

        if meta.get("na"):
            self._sp_na.setValue(float(meta["na"]))
        self._sp_na.setStyleSheet(_bg("na" in from_file))
        px_x = meta.get("pixel_size_x")
        if px_x:
            self._sp_px_xy.setValue(float(px_x) * 1000.0)  # µm → nm
        self._sp_px_xy.setStyleSheet(_bg("pixel_size_x" in from_file))
        px_z = meta.get("pixel_size_z")
        if px_z:
            self._sp_px_z.setValue(float(px_z) * 1000.0)
        self._sp_px_z.setStyleSheet(_bg("pixel_size_z" in from_file))
        ri = meta.get("refractive_index")
        if ri:
            self._sp_ri_imm.setValue(float(ri))
        self._sp_ri_imm.setStyleSheet(_bg("refractive_index" in from_file))
        micro = meta.get("microscope_type")
        if micro:
            idx = self._micro_combo.findText(micro)
            if idx >= 0:
                self._micro_combo.setCurrentIndex(idx)
            # Auto-set iterations based on microscope type
            if micro == "confocal":
                self._le_niter.setText("50")
            else:
                self._le_niter.setText("150")
        self._micro_combo.setStyleSheet(_bg("microscope_type" in from_file))

        # Per-channel wavelengths
        ch_info = meta.get("channels", [])
        if ch_info:
            self._le_emission.setText(
                _format_channel_values(ch_info, "emission_wavelength", 520.0)
            )
            self._le_excitation.setText(
                _format_channel_values(ch_info, "excitation_wavelength", 488.0)
            )
        self._le_emission.setStyleSheet(
            _bg("emission_wavelength" in from_file))
        self._le_excitation.setStyleSheet(
            _bg("excitation_wavelength" in from_file))

        # RI sample is never in metadata — red (needs user input)
        self._sp_ri_sample.setStyleSheet(
            "background-color: #ffe0e0; color: black;")  # pastel red

        self._viewer.set_input_data([], self._metadata)
        self._load_timepoint_into_viewer(int(self._metadata.get("default_t", 0)), force=True)

        size_t = self._metadata.get("size_t", 1)
        size_z = self._metadata.get("size_z", 1)
        size_y = self._metadata.get("size_y", "?")
        size_x = self._metadata.get("size_x", "?")
        n_ch = int(self._metadata.get("size_c", 0))
        self._file_label.setText(
            f"{display_name}\n{n_ch} ch, T={size_t}, Z={size_z}, YX={size_y}×{size_x}"
        )
        self._btn_run.setEnabled(True)
        self._btn_save_series.setEnabled(self._viewer.has_time_axis())
        self._btn_save_series.setVisible(self._viewer.has_time_axis())
        self._btn_save.setEnabled(False)
        self._sync_preview_buttons()
        self._viewer.refresh_view()
        self._status.showMessage(f"Loaded {display_name}", 5000)

    # -----------------------------------------------------------------------
    # Open from OMERO
    # -----------------------------------------------------------------------

    def _on_open_omero(self):
        try:
            from omero_browser_qt import (
                LoginDialog,
                OmeroBrowserDialog,
                OmeroGateway,
            )
        except ImportError:
            QMessageBox.warning(
                self,
                "OMERO not available",
                "omero-browser-qt is not installed.\n\n"
                "Install with:\n  pip install \"omero-browser-qt[viewer]==0.2.2\"",
            )
            return

        if self._omero_gw is None:
            try:
                self._omero_gw = OmeroGateway()
            except RuntimeError:
                # PyQt6 singleton init order: QObject.__init__() must be
                # called before any attribute access.  Work around by
                # bootstrapping the QObject base on the raw singleton.
                from PyQt6.QtCore import QObject as _QObj
                inst = OmeroGateway._instance
                _QObj.__init__(inst)
                OmeroGateway.__init__(inst)
                self._omero_gw = inst
        gw = self._omero_gw

        if gw.is_connected() and not self._omero_session_is_reusable():
            gw.disconnect()
            self._omero_session_deadline = 0.0

        if not self._omero_session_is_reusable():
            dlg = LoginDialog(self, gateway=gw)
            self._configure_omero_login_dialog(dlg)
            if dlg.exec() != LoginDialog.DialogCode.Accepted:
                return
            self._refresh_omero_session_deadline()
        else:
            self._refresh_omero_session_deadline()

        browser = OmeroBrowserDialog(self, gateway=gw)
        if browser.exec() != OmeroBrowserDialog.DialogCode.Accepted:
            return

        self._refresh_omero_session_deadline()

        images = browser.get_selected_images()
        if not images:
            return

        image = images[0]
        name = image.getName()
        try:
            self._begin_busy_progress(f"Opening {name} from OMERO …")
            source = _build_omero_source(image)
            self._apply_image_source(
                source,
                f"OMERO: {name}",
                source_factory=lambda _image=image: _build_omero_source(_image),
            )
        except Exception as exc:
            QMessageBox.critical(self, "OMERO Error", str(exc))
            self._status.showMessage("OMERO load failed", 5000)
        finally:
            self._end_progress()

    # -----------------------------------------------------------------------
    # Run deconvolution
    # -----------------------------------------------------------------------

    def _collect_params(self) -> dict:
        device_text = self._device_combo.currentText()
        device = None if device_text == "auto" else device_text

        bg_text = self._bg_combo.currentText()
        background: str | float = "auto"
        if bg_text == "manual":
            background = self._sp_bg_value.value()

        offset_text = self._offset_combo.currentText()
        offset: str | float = "auto"
        if offset_text == "none":
            offset = 0.0
        elif offset_text == "manual":
            offset = self._sp_offset.value()

        em_list = _parse_float_list(self._le_emission.text())
        ex_list = _parse_float_list(self._le_excitation.text())

        niter_list = []
        for s in self._le_niter.text().split(","):
            s = s.strip()
            if s:
                try:
                    niter_list.append(max(1, int(s)))
                except ValueError:
                    pass
        if not niter_list:
            niter_list = [50]

        damping_text = self._damping_combo.currentText()
        damping: str | float = "auto"
        if damping_text == "none":
            damping = 0.0
        elif damping_text == "manual":
            damping = self._sp_damping.value()

        return {
            "method": self._method_combo.currentText(),
            "niter_list": niter_list,
            "tv_lambda": self._sp_tv_lambda.value(),
            "damping": damping,
            "offset": offset,
            "prefilter_sigma": self._sp_prefilter.value(),
            "start": self._start_combo.currentText(),
            "sparse_hessian_weight": self._sp_sparse_weight.value(),
            "sparse_hessian_reg": self._sp_sparse_reg.value(),
            "background": background,
            "convergence": self._conv_combo.currentText(),
            "rel_threshold": self._sp_rel_thresh.value(),
            "check_every": self._sp_check_every.value(),
            "device": device,
            "na": self._sp_na.value(),
            "emission_wavelengths": em_list,
            "excitation_wavelengths": ex_list,
            "pixel_size_xy_nm": self._sp_px_xy.value(),
            "pixel_size_z_nm": self._sp_px_z.value(),
            "ri_immersion": self._sp_ri_imm.value(),
            "ri_sample": self._sp_ri_sample.value(),
            "t_g": self._sp_tg.value(),
            "t_g0": self._sp_tg0.value(),
            "t_i0": self._sp_ti0.value(),
            "z_p": self._sp_zp.value(),
            "microscope_type": self._micro_combo.currentText(),
            "integrate_pixels": self._cb_integrate.isChecked(),
            "n_subpixels": self._sp_subpixels.value(),
            "n_pupil": self._sp_n_pupil.value(),
            "tiling": self._tiling_combo.currentText(),
            "max_tile_xy": self._sp_max_tile_xy.value(),
            "max_tile_z": self._sp_max_tile_z.value(),
        }

    def _on_run(self):
        # --- Stop mode: cancel running worker ---
        if self._worker is not None and self._worker.isRunning():
            self._worker.requestInterruption()
            self._btn_run.setEnabled(False)
            self._status.showMessage("Stopping …")
            return

        if self._save_worker is not None and self._save_worker.isRunning():
            QMessageBox.information(
                self,
                "Save in progress",
                "A full T-series export is currently running. Please wait for it to finish first.",
            )
            return

        if not self._input_channels:
            return

        self._btn_run.setText("Stop")
        self._btn_run.setStyleSheet(
            "QPushButton { background-color: #e53935; color: white; "
            "font-weight: bold; padding: 8px; }"
        )
        self._btn_save.setEnabled(False)
        self._btn_save_series.setEnabled(False)
        self._begin_busy_progress("Running deconvolution …")

        # Signal that deconvolution is active (dot indicator)
        self._monitor_bar.set_active(True)

        # Auto-save settings before each run
        self._save_last_settings()

        current_t = self._viewer.current_timepoint()
        params = self._collect_params()
        self._worker = _DeconvolveWorker(
            self._input_channels, self._metadata, params, current_t, parent=self
        )
        self._worker.progress.connect(lambda msg: self._status.showMessage(msg))
        self._worker.finished.connect(self._on_deconv_done)
        self._worker.finished.connect(self._worker.deleteLater)
        self._worker.start()

    def _on_deconv_done(self, result):
        self._monitor_bar.set_active(False)

        self._end_progress()
        self._btn_run.setText("Run Deconvolution")
        self._btn_run.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; "
            "font-weight: bold; padding: 8px; }"
        )
        self._btn_run.setEnabled(True)
        self._btn_save_series.setEnabled(bool(self._input_channels) and self._viewer.has_time_axis())

        if isinstance(result, Exception):
            self._worker = None
            msg = str(result)
            if "Stopped by user" in msg:
                self._status.showMessage("Deconvolution stopped", 5000)
            else:
                QMessageBox.critical(self, "Deconvolution Error", msg)
                self._status.showMessage("Deconvolution failed", 5000)
            return

        try:
            timepoint = int(result["timepoint"])
            self._preview_outputs_by_t[timepoint] = result["channels"]
            self._viewer.set_preview_result(timepoint, result["channels"])
            self._sync_preview_buttons()
            self._status.showMessage(f"Deconvolution complete for T={timepoint + 1}", 5000)
        except Exception as exc:
            traceback.print_exc()
            QMessageBox.critical(self, "Viewer Error", str(exc))
        finally:
            self._worker = None

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------

    def _on_save(self):
        current_t = self._viewer.current_timepoint()
        preview_channels = self._preview_outputs_by_t.get(current_t)
        if not preview_channels:
            return

        stem = self._input_path.stem if self._input_path else "deconvolved"
        method = self._method_combo.currentText()
        niter_text = self._le_niter.text().strip().replace(", ", "-").replace(",", "-")
        suggested = f"{stem}_{method}_{niter_text}i_T{current_t:03d}.ome.tiff"

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Deconvolved Preview",
            str(Path(self._last_save_dir) / suggested),
            "OME-TIFF (*.ome.tiff);;TIFF (*.tiff *.tif)",
        )
        if not path:
            return
        self._last_save_dir = str(Path(path).parent)

        try:
            stack = np.stack(preview_channels, axis=0)
            data = stack[np.newaxis, ...].astype(np.float32)
            _write_ome_tiff(data, path, self._metadata)
            self._status.showMessage(f"Saved → {Path(path).name}", 5000)
        except Exception as exc:
            QMessageBox.critical(self, "Save Error", str(exc))

    def _on_save_t_series(self):
        if not self._input_channels:
            return
        if self._worker is not None and self._worker.isRunning():
            QMessageBox.information(
                self,
                "Preview running",
                "Stop the current preview deconvolution before starting a full T-series export.",
            )
            return
        if self._save_worker is not None and self._save_worker.isRunning():
            return

        stem = self._input_path.stem if self._input_path else "deconvolved"
        method = self._method_combo.currentText()
        niter_text = self._le_niter.text().strip().replace(", ", "-").replace(",", "-")
        suggested = f"{stem}_{method}_{niter_text}i_TSERIES.ome.tiff"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Deconvolved T-Series",
            str(Path(self._last_save_dir) / suggested),
            "OME-TIFF (*.ome.tiff);;TIFF (*.tiff *.tif)",
        )
        if not path:
            return
        self._last_save_dir = str(Path(path).parent)

        self._save_last_settings()
        self._begin_busy_progress("Saving full T-series …")
        self._btn_run.setEnabled(False)
        self._btn_save.setEnabled(False)
        self._btn_save_series.setEnabled(False)
        self._monitor_bar.set_active(True)

        if self._input_source_factory is None:
            QMessageBox.warning(self, "Save Error", "No image source is available for T-series export.")
            self._monitor_bar.set_active(False)
            self._end_progress()
            self._btn_run.setEnabled(bool(self._input_channels))
            self._sync_preview_buttons()
            self._btn_save_series.setEnabled(bool(self._input_channels) and self._viewer.has_time_axis())
            return

        params = self._collect_params()
        try:
            save_source = self._input_source_factory()
        except Exception as exc:
            self._monitor_bar.set_active(False)
            self._end_progress()
            self._btn_run.setEnabled(bool(self._input_channels))
            self._sync_preview_buttons()
            self._btn_save_series.setEnabled(bool(self._input_channels) and self._viewer.has_time_axis())
            QMessageBox.critical(self, "Save Error", str(exc))
            return

        self._save_worker = _SaveTSeriesWorker(
            save_source,
            self._metadata,
            params,
            path,
            parent=self,
        )
        self._save_worker.progress.connect(lambda msg: self._status.showMessage(msg))
        self._save_worker.finished.connect(self._on_save_t_series_done)
        self._save_worker.finished.connect(self._save_worker.deleteLater)
        self._save_worker.start()

    def _on_save_t_series_done(self, result):
        self._monitor_bar.set_active(False)
        self._end_progress()
        self._btn_run.setEnabled(bool(self._input_channels))
        self._sync_preview_buttons()
        self._btn_save_series.setEnabled(bool(self._input_channels) and self._viewer.has_time_axis())
        self._save_worker = None

        if isinstance(result, Exception):
            msg = str(result)
            if "Stopped by user" in msg:
                self._status.showMessage("T-series export stopped", 5000)
            else:
                QMessageBox.critical(self, "Save Error", msg)
                self._status.showMessage("T-series export failed", 5000)
            return

        self._status.showMessage(f"Saved → {Path(result['path']).name}", 5000)

    # -----------------------------------------------------------------------
    # Settings Save / Load / Restore
    # -----------------------------------------------------------------------

    def _settings_to_dict(self) -> dict:
        """Collect all UI parameter values into a JSON-serializable dict."""
        return {
            "method": self._method_combo.currentText(),
            "iterations": self._le_niter.text(),
            "tv_lambda": self._sp_tv_lambda.value(),
            "damping": self._damping_combo.currentText(),
            "damping_value": self._sp_damping.value(),
            "background": self._bg_combo.currentText(),
            "background_value": self._sp_bg_value.value(),
            "offset": self._offset_combo.currentText(),
            "offset_value": self._sp_offset.value(),
            "prefilter_sigma": self._sp_prefilter.value(),
            "start": self._start_combo.currentText(),
            "sparse_hessian_weight": self._sp_sparse_weight.value(),
            "sparse_hessian_reg": self._sp_sparse_reg.value(),
            "convergence": self._conv_combo.currentText(),
            "rel_threshold": self._sp_rel_thresh.value(),
            "check_every": self._sp_check_every.value(),
            "device": self._device_combo.currentText(),
            "tiling": self._tiling_combo.currentText(),
            "max_tile_xy": self._sp_max_tile_xy.value(),
            "max_tile_z": self._sp_max_tile_z.value(),
            "na": self._sp_na.value(),
            "emission_wavelengths": self._le_emission.text(),
            "excitation_wavelengths": self._le_excitation.text(),
            "pixel_size_xy_nm": self._sp_px_xy.value(),
            "pixel_size_z_nm": self._sp_px_z.value(),
            "ri_immersion": self._sp_ri_imm.value(),
            "ri_sample": self._sp_ri_sample.value(),
            "embedding_medium": self._medium_combo.currentText(),
            "t_g": self._sp_tg.value(),
            "t_g0": self._sp_tg0.value(),
            "t_i0": self._sp_ti0.value(),
            "z_p": self._sp_zp.value(),
            "microscope_type": self._micro_combo.currentText(),
            "integrate_pixels": self._cb_integrate.isChecked(),
            "n_subpixels": self._sp_subpixels.value(),
            "n_pupil": self._sp_n_pupil.value(),
            "pct_lo": self._viewer.lo_percentile(),
            "pct_hi": self._viewer.hi_percentile(),
        }

    def _apply_settings(self, data: dict):
        """Populate UI widgets from a settings dict."""
        def _combo(combo: QComboBox, key: str):
            val = data.get(key)
            if val is not None:
                idx = combo.findText(str(val))
                if idx >= 0:
                    combo.setCurrentIndex(idx)

        def _spin(spin, key: str):
            val = data.get(key)
            if val is not None:
                spin.setValue(type(spin.value())(val))

        def _line(le: QLineEdit, key: str):
            val = data.get(key)
            if val is not None:
                le.setText(str(val))

        _combo(self._method_combo, "method")
        _line(self._le_niter, "iterations")
        _spin(self._sp_tv_lambda, "tv_lambda")
        _combo(self._damping_combo, "damping")
        _spin(self._sp_damping, "damping_value")
        _combo(self._bg_combo, "background")
        _spin(self._sp_bg_value, "background_value")
        _combo(self._offset_combo, "offset")
        _spin(self._sp_offset, "offset_value")
        _spin(self._sp_prefilter, "prefilter_sigma")
        _combo(self._start_combo, "start")
        _spin(self._sp_sparse_weight, "sparse_hessian_weight")
        _spin(self._sp_sparse_reg, "sparse_hessian_reg")
        _combo(self._conv_combo, "convergence")
        _spin(self._sp_rel_thresh, "rel_threshold")
        _spin(self._sp_check_every, "check_every")
        _combo(self._device_combo, "device")
        _combo(self._tiling_combo, "tiling")
        _spin(self._sp_max_tile_xy, "max_tile_xy")
        _spin(self._sp_max_tile_z, "max_tile_z")
        _spin(self._sp_na, "na")
        _line(self._le_emission, "emission_wavelengths")
        _line(self._le_excitation, "excitation_wavelengths")
        _spin(self._sp_px_xy, "pixel_size_xy_nm")
        _spin(self._sp_px_z, "pixel_size_z_nm")
        _spin(self._sp_ri_imm, "ri_immersion")
        _spin(self._sp_ri_sample, "ri_sample")
        # Try to match combo to the loaded RI value; value is leading
        ri_val = data.get("ri_sample")
        if ri_val is not None:
            matched = False
            for name, ri in self._medium_ri_map.items():
                if abs(ri - float(ri_val)) < 1e-4:
                    self._medium_combo.blockSignals(True)
                    self._medium_combo.setCurrentText(name)
                    self._medium_combo.blockSignals(False)
                    matched = True
                    break
            if not matched:
                # No exact match — just leave combo as-is
                emb = data.get("embedding_medium")
                if emb is not None:
                    idx = self._medium_combo.findText(str(emb))
                    if idx >= 0:
                        self._medium_combo.blockSignals(True)
                        self._medium_combo.setCurrentIndex(idx)
                        self._medium_combo.blockSignals(False)
        _spin(self._sp_tg, "t_g")
        _spin(self._sp_tg0, "t_g0")
        _spin(self._sp_ti0, "t_i0")
        _spin(self._sp_zp, "z_p")
        _combo(self._micro_combo, "microscope_type")
        _spin(self._sp_subpixels, "n_subpixels")
        _spin(self._sp_n_pupil, "n_pupil")
        if data.get("pct_lo") is not None:
            self._viewer.set_lo_percentile(float(data["pct_lo"]))
        if data.get("pct_hi") is not None:
            self._viewer.set_hi_percentile(float(data["pct_hi"]))

        # integrate_pixels checkbox
        val = data.get("integrate_pixels")
        if val is not None:
            self._cb_integrate.setChecked(bool(val))

    def _save_last_settings(self):
        """Auto-save settings to .last_settings.json."""
        try:
            with open(LAST_SETTINGS_PATH, "w", encoding="utf-8") as f:
                json.dump(self._settings_to_dict(), f, indent=2)
            self._btn_restore.setEnabled(True)
        except OSError:
            pass

    def _on_restore_settings(self):
        """Restore settings from .last_settings.json."""
        try:
            with open(LAST_SETTINGS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            return
        self._apply_settings(data)
        self._viewer.refresh_view()
        self._status.showMessage("Settings restored", 3000)

    def _on_save_settings(self):
        """Save current settings to a user-chosen JSON file."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Settings", str(Path(self._last_settings_dir) / "settings.json"),
            "JSON files (*.json);;All files (*)",
        )
        if not path:
            return
        self._last_settings_dir = str(Path(path).parent)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._settings_to_dict(), f, indent=2)
        self._status.showMessage(f"Settings saved → {Path(path).name}", 3000)

    def _on_load_settings(self):
        """Load settings from a user-chosen JSON file."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Settings", self._last_settings_dir,
            "JSON files (*.json);;All files (*)",
        )
        if not path:
            return
        self._last_settings_dir = str(Path(path).parent)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            QMessageBox.warning(self, "Load Error", str(exc))
            return
        self._apply_settings(data)
        self._viewer.refresh_view()
        self._status.showMessage(f"Settings loaded from {Path(path).name}", 3000)

    def _configure_omero_login_dialog(self, dlg):
        """Hide any remember-me checkbox and force 10-minute reuse."""
        for cb in dlg.findChildren(QCheckBox):
            text = cb.text().strip().lower()
            if "remember" in text:
                cb.setChecked(True)
                cb.hide()

    def _refresh_omero_session_deadline(self):
        """Extend OMERO session reuse window from the current moment."""
        self._omero_session_deadline = time.monotonic() + OMERO_SESSION_REUSE_S

    def _omero_session_is_reusable(self) -> bool:
        """Return True when the current OMERO connection is still reusable."""
        if self._omero_gw is None or not self._omero_gw.is_connected():
            return False
        return time.monotonic() < self._omero_session_deadline

    # -----------------------------------------------------------------------
    # Viewer
    # -----------------------------------------------------------------------
    def _load_timepoint_into_viewer(self, timepoint: int, *, force: bool = False) -> None:
        if self._input_source is None:
            return
        target_t = max(0, min(int(timepoint), max(int(self._metadata.get("size_t", 1)) - 1, 0)))
        if not force and self._loaded_timepoint == target_t and self._input_channels:
            return

        size_z = max(int(self._metadata.get("size_z", 1)), 1)
        size_c = max(int(self._metadata.get("size_c", 1)), 1)
        total = size_c * size_z if isinstance(self._input_source, _OmeroTimepointSource) and size_z > 1 else size_c
        if total > 1:
            self._begin_progress(total, f"Loading T={target_t + 1}…")
        else:
            self._begin_busy_progress(f"Loading T={target_t + 1}…")

        try:
            channels = self._input_source.load_timepoint(
                target_t,
                progress_cb=lambda done, total_steps, text: self._advance_progress(done, text),
            )
            self._input_channels = channels
            self._loaded_timepoint = target_t
            self._viewer.set_input_timepoint_data(target_t, channels)
        finally:
            self._end_progress()

    def _sync_preview_buttons(self) -> None:
        current_t = self._viewer.current_timepoint()
        self._btn_save.setEnabled(current_t in self._preview_outputs_by_t)

    def _on_viewer_time_changed(self, timepoint: int) -> None:
        previous_t = self._loaded_timepoint
        try:
            self._load_timepoint_into_viewer(int(timepoint))
        except Exception as exc:
            if previous_t is not None and previous_t != int(timepoint):
                self._viewer.set_timepoint(previous_t)
            QMessageBox.critical(self, "Load Error", str(exc))
            self._status.showMessage("Timepoint load failed", 5000)
        self._sync_preview_buttons()

    def _update_viewer(self):
        self._viewer.refresh_view()

    def closeEvent(self, event):
        """Ensure background threads are stopped before the window closes."""
        if self._worker is not None and self._worker.isRunning():
            self._worker.requestInterruption()
            self._worker.wait(2000)
            self._worker = None
        if self._save_worker is not None and self._save_worker.isRunning():
            self._save_worker.requestInterruption()
            self._save_worker.wait(2000)
            self._save_worker = None
        if self._monitor is not None:
            self._monitor.request_stop()
            self._monitor = None
        if self._omero_gw is not None:
            try:
                self._omero_gw.disconnect()
            except Exception:
                pass
            self._omero_gw = None
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _excepthook(exc_type, exc_value, exc_tb):
    """Show uncaught exceptions in a dialog instead of silently crashing."""
    msg = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    log.critical("Uncaught exception:\n%s", msg)
    print(msg, file=sys.stderr, flush=True)
    QMessageBox.critical(None, "Fatal Error", msg)


def main():
    sys.excepthook = _excepthook

    app = QApplication(sys.argv)
    app.setApplicationName("CI Deconvolve")
    if ICON_PATH.exists():
        app.setWindowIcon(QIcon(str(ICON_PATH)))

    window = DeconvolveCIWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
