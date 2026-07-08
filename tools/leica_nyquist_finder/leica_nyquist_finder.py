from __future__ import annotations

import json
import math
import os
import re
import sys
import time
import traceback
from dataclasses import dataclass, field
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PyQt6.QtCore import QSize, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QPlainTextEdit,
    QDoubleSpinBox,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


STATUS_ORDER = {"pass": 0, "xy only": 1, "near": 2, "too coarse": 3, "unknown": 4}
DEFAULT_PINHOLE_AIRY = 1.0
LEICA_LIF_EXTENSIONS = {".lif"}
LEICA_PROJECT_EXTENSIONS = {".lof", ".xlef", ".xlif"}
LEICA_EXTENSIONS = LEICA_LIF_EXTENSIONS | LEICA_PROJECT_EXTENSIONS
SKIP_FOLDER_NAME_PARTS = ("metadata", "pyramid")
CACHE_DIR = Path(os.environ.get("LOCALAPPDATA") or (Path.home() / "AppData" / "Local")) / "CIDeconvolve" / "leica_nyquist_finder"
CACHE_PATH = CACHE_DIR / "scan_cache.json"
CACHE_VERSION = 2
CACHE_SAVE_INTERVAL_SECONDS = 10.0
CACHE_EDIT_SAVE_DELAY_MS = 1500
SCAN_COOPERATE_INTERVAL_SECONDS = 0.05
SCAN_COOPERATE_OPS = 50
SCAN_COOPERATE_SLEEP_MS = 2


@dataclass
class NyquistResult:
    status: str
    limit_xy_um: float | None = None
    limit_z_um: float | None = None
    ratio_xy: float | None = None
    ratio_z: float | None = None
    xy_ok: bool | None = None
    z_ok: bool | None = None
    pinhole_au: float | None = None
    cxy: float | None = None
    cz: float | None = None
    wavelength_nm: float | None = None
    wavelength_source: str = ""
    notes: list[str] = field(default_factory=list)


@dataclass
class LeicaRecord:
    context: Any
    file_path: Path
    internal_path: str
    image_name: str
    size_x: int | None
    size_y: int | None
    size_z: int | None
    size_c: int | None
    size_t: int | None
    pixel_x_um: float | None
    pixel_y_um: float | None
    pixel_z_um: float | None
    na: float | None
    refractive_index: float | None
    excitation_nm: list[float]
    emission_nm: list[float]
    microscope_type: str
    channel_names: list[str]
    metadata: dict[str, Any]
    nyquist: NyquistResult
    thumbnail_path: Path | None = None
    thumbnail_note: str = ""
    checked: bool = False

    @property
    def is_3d(self) -> bool:
        return int(self.size_z or 1) > 1

    @property
    def is_confocal(self) -> bool:
        return _looks_confocal(self.microscope_type, self.metadata)


def nyquist_to_dict(result: NyquistResult) -> dict[str, Any]:
    return {
        "status": result.status,
        "limit_xy_um": result.limit_xy_um,
        "limit_z_um": result.limit_z_um,
        "ratio_xy": result.ratio_xy,
        "ratio_z": result.ratio_z,
        "xy_ok": result.xy_ok,
        "z_ok": result.z_ok,
        "pinhole_au": result.pinhole_au,
        "cxy": result.cxy,
        "cz": result.cz,
        "wavelength_nm": result.wavelength_nm,
        "wavelength_source": result.wavelength_source,
        "notes": list(result.notes),
    }


def nyquist_from_dict(data: dict[str, Any]) -> NyquistResult:
    return NyquistResult(
        status=str(data.get("status") or "unknown"),
        limit_xy_um=_first_float(data.get("limit_xy_um")),
        limit_z_um=_first_float(data.get("limit_z_um")),
        ratio_xy=_first_float(data.get("ratio_xy")),
        ratio_z=_first_float(data.get("ratio_z")),
        xy_ok=data.get("xy_ok") if isinstance(data.get("xy_ok"), bool) else None,
        z_ok=data.get("z_ok") if isinstance(data.get("z_ok"), bool) else None,
        pinhole_au=_first_float(data.get("pinhole_au")),
        cxy=_first_float(data.get("cxy")),
        cz=_first_float(data.get("cz")),
        wavelength_nm=_first_float(data.get("wavelength_nm")),
        wavelength_source=str(data.get("wavelength_source") or ""),
        notes=[str(item) for item in data.get("notes", []) if item is not None],
    )


def context_to_dict(context: Any) -> dict[str, Any]:
    if context is None:
        return {}
    if hasattr(context, "to_dict"):
        try:
            data = context.to_dict()
            if isinstance(data, dict):
                return _json_safe(data)
        except Exception:
            pass
    return {
        "name": str(getattr(context, "name", "")),
        "container_path": str(getattr(context, "container_path", "")),
        "internal_path": str(getattr(context, "internal_path", "")),
        "image_id": getattr(context, "image_id", None),
        "kind": str(getattr(context, "kind", "lif-image")),
        "size_x": getattr(context, "size_x", None),
        "size_y": getattr(context, "size_y", None),
        "size_z": getattr(context, "size_z", None),
        "size_c": getattr(context, "size_c", None),
        "size_t": getattr(context, "size_t", None),
        "size_s": getattr(context, "size_s", None),
        "pixel_size_x_um": getattr(context, "pixel_size_x_um", None),
        "pixel_size_y_um": getattr(context, "pixel_size_y_um", None),
        "pixel_size_z_um": getattr(context, "pixel_size_z_um", None),
        "selected_s": getattr(context, "selected_s", None),
        "channel_names": list(getattr(context, "channel_names", []) or []),
        "metadata": _json_safe(getattr(context, "metadata", {}) or {}),
    }


def context_from_dict(data: dict[str, Any]) -> Any:
    try:
        from leica_browser_qt.models import LeicaImageContext
    except Exception:
        return None
    if not isinstance(data, dict) or not data:
        return None
    return LeicaImageContext(
        name=str(data.get("name") or ""),
        container_path=Path(str(data.get("container_path") or "")),
        internal_path=str(data.get("internal_path") or ""),
        image_id=data.get("image_id"),
        kind=str(data.get("kind") or "lif-image"),
        size_x=_positive_int(data.get("size_x"), None),
        size_y=_positive_int(data.get("size_y"), None),
        size_z=_positive_int(data.get("size_z"), None),
        size_c=_positive_int(data.get("size_c"), None),
        size_t=_positive_int(data.get("size_t"), None),
        size_s=_positive_int(data.get("size_s"), None),
        pixel_size_x_um=_first_float(data.get("pixel_size_x_um")),
        pixel_size_y_um=_first_float(data.get("pixel_size_y_um")),
        pixel_size_z_um=_first_float(data.get("pixel_size_z_um")),
        selected_s=_nonnegative_int(data.get("selected_s")),
        channel_names=[str(item) for item in data.get("channel_names", [])],
        metadata=dict(data.get("metadata") or {}),
    )


def record_to_dict(record: LeicaRecord) -> dict[str, Any]:
    context_data = context_to_dict(record.context)
    if isinstance(context_data, dict):
        context_data = dict(context_data)
        context_data.pop("metadata", None)
    metadata = _json_safe(record.metadata)
    return {
        "context": context_data,
        "file_path": str(record.file_path),
        "internal_path": record.internal_path,
        "image_name": record.image_name,
        "size_x": record.size_x,
        "size_y": record.size_y,
        "size_z": record.size_z,
        "size_c": record.size_c,
        "size_t": record.size_t,
        "pixel_x_um": record.pixel_x_um,
        "pixel_y_um": record.pixel_y_um,
        "pixel_z_um": record.pixel_z_um,
        "na": record.na,
        "refractive_index": record.refractive_index,
        "excitation_nm": list(record.excitation_nm),
        "emission_nm": list(record.emission_nm),
        "microscope_type": record.microscope_type,
        "channel_names": list(record.channel_names),
        "metadata": metadata,
        "nyquist": nyquist_to_dict(record.nyquist),
        "thumbnail_path": str(record.thumbnail_path) if record.thumbnail_path else "",
        "thumbnail_note": record.thumbnail_note,
        "checked": record.checked,
    }


def record_from_dict(data: dict[str, Any]) -> LeicaRecord:
    context_data = dict(data.get("context") or {})
    if "metadata" not in context_data and data.get("metadata") is not None:
        context_data["metadata"] = data.get("metadata")
    context = context_from_dict(context_data)
    context_metadata = getattr(context, "metadata", {}) if context is not None else {}
    metadata = dict(data.get("metadata") or context_metadata or {})
    return LeicaRecord(
        context=context,
        file_path=Path(str(data.get("file_path") or "")),
        internal_path=str(data.get("internal_path") or ""),
        image_name=str(data.get("image_name") or ""),
        size_x=_positive_int(data.get("size_x"), None),
        size_y=_positive_int(data.get("size_y"), None),
        size_z=_positive_int(data.get("size_z"), None),
        size_c=_positive_int(data.get("size_c"), None),
        size_t=_positive_int(data.get("size_t"), None),
        pixel_x_um=_first_float(data.get("pixel_x_um")),
        pixel_y_um=_first_float(data.get("pixel_y_um")),
        pixel_z_um=_first_float(data.get("pixel_z_um")),
        na=_first_float(data.get("na")),
        refractive_index=_first_float(data.get("refractive_index")),
        excitation_nm=_positive_numbers(data.get("excitation_nm")),
        emission_nm=_positive_numbers(data.get("emission_nm")),
        microscope_type=str(data.get("microscope_type") or ""),
        channel_names=[str(item) for item in data.get("channel_names", [])],
        metadata=metadata,
        nyquist=nyquist_from_dict(data.get("nyquist") or {}),
        thumbnail_path=Path(str(data.get("thumbnail_path"))) if data.get("thumbnail_path") else None,
        thumbnail_note=str(data.get("thumbnail_note") or ""),
        checked=bool(data.get("checked")),
    )


def cache_payload(
    *,
    settings: dict[str, Any],
    records: list[LeicaRecord],
    completed_containers: set[str],
    status: str,
) -> dict[str, Any]:
    return {
        "version": CACHE_VERSION,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": status,
        "settings": settings,
        "completed_containers": sorted(completed_containers),
        "records": [record_to_dict(record) for record in records],
    }


def save_cache(payload: dict[str, Any]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = CACHE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    tmp.replace(CACHE_PATH)


def load_cache() -> dict[str, Any] | None:
    if not CACHE_PATH.is_file():
        return None
    try:
        data = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict) or int(data.get("version", 0) or 0) != CACHE_VERSION:
        return None
    return data


class ScanWorker(QThread):
    record_found = pyqtSignal(object)
    container_completed = pyqtSignal(str)
    log_message = pyqtSignal(str)
    progress_message = pyqtSignal(str)
    finished_scan = pyqtSignal(int)

    def __init__(
        self,
        root: Path,
        tolerance: float,
        *,
        scan_collections: bool = False,
        create_thumbnails: bool = True,
        include_project_files: bool = False,
        max_xy_pixels: int = 2048,
        thumbnail_max_xy_pixels: int = 2048,
        folder_filter: str = "",
        completed_containers: Iterable[str] | None = None,
        parent: Any = None,
    ) -> None:
        super().__init__(parent)
        self.root = root
        self.tolerance = tolerance
        self.scan_collections = bool(scan_collections)
        self.create_thumbnails = bool(create_thumbnails)
        self.include_project_files = bool(include_project_files)
        self.max_xy_pixels = max(0, int(max_xy_pixels))
        self.thumbnail_max_xy_pixels = max(0, int(thumbnail_max_xy_pixels))
        self.folder_filter = str(folder_filter or "").strip()
        self.completed_containers = {str(Path(item)) for item in (completed_containers or [])}
        self._cancelled = False
        self._started_at = 0.0
        self._containers_seen = 0
        self._containers_done = 0
        self._folders_seen = 0
        self._records_found = 0
        self._last_cooperate_at = time.monotonic()
        self._ops_since_cooperate = 0

    def cancel(self) -> None:
        self._cancelled = True

    def _cooperate(self, *, force: bool = False) -> None:
        """Let the GUI thread breathe during very large filesystem/metadata scans."""
        self._ops_since_cooperate += 1
        now = time.monotonic()
        if (
            force
            or self._ops_since_cooperate >= SCAN_COOPERATE_OPS
            or (now - self._last_cooperate_at) >= SCAN_COOPERATE_INTERVAL_SECONDS
        ):
            QThread.msleep(SCAN_COOPERATE_SLEEP_MS)
            self._last_cooperate_at = time.monotonic()
            self._ops_since_cooperate = 0

    def run(self) -> None:
        count = 0
        self._started_at = time.monotonic()
        self._progress("starting")
        try:
            from leica_browser_qt import LeicaGateway

            gateway = LeicaGateway()
            self.log_message.emit(f"Scanning Leica root: {self.root}")
            for container in self._iter_leica_containers(self.root):
                self._cooperate(force=True)
                if self._cancelled:
                    self.log_message.emit("Scan cancelled.")
                    break
                container_key = str(container)
                if container_key in self.completed_containers:
                    self._containers_done += 1
                    self.log_message.emit(f"Skipping cached completed Leica container: {container}")
                    self._progress(f"cached {container.name}")
                    continue
                self.log_message.emit(f"Opening Leica container: {container}")
                self._progress(f"opening {container.name}")
                t_container = time.monotonic()
                try:
                    node = gateway.container_node(container)
                except Exception as exc:
                    self.log_message.emit(f"Could not open Leica container {container}: {exc}")
                    self._containers_done += 1
                    self.container_completed.emit(container_key)
                    self._progress(f"failed {container.name}")
                    continue
                self.log_message.emit(
                    f"Opened Leica container in {format_duration(time.monotonic() - t_container)}: {container}"
                )
                if self.scan_collections:
                    self.log_message.emit(f"Reading all Leica entries: {container}")
                    count += self._walk_node(gateway, node)
                else:
                    self.log_message.emit(f"Reading first root Leica image only: {container}")
                    count += self._walk_container_root_only(gateway, node)
                self._containers_done += 1
                self.container_completed.emit(container_key)
                self._progress(f"done {container.name}")
                self.log_message.emit(
                    f"Finished Leica container in {format_duration(time.monotonic() - t_container)}: {container}"
                )
                self._cooperate(force=True)
        except Exception as exc:
            self.log_message.emit(f"Scan failed: {exc}\n{traceback.format_exc()}")
        self._progress("finished")
        self.finished_scan.emit(count)

    def _iter_leica_containers(self, root: Path) -> Iterable[Path]:
        root = Path(root)
        if root.is_file():
            if self._is_allowed_leica_container(root):
                self.log_message.emit(f"Found Leica file: {root}")
                self._containers_seen += 1
                self._progress(f"found {root.name}")
                yield root
            elif root.suffix.lower() in LEICA_PROJECT_EXTENSIONS and not self.include_project_files:
                self.log_message.emit(f"Skipping Leica LOF/XLEF/XLIF file because option is off: {root}")
            else:
                self.log_message.emit(f"Not a Leica file: {root}")
            return

        if not root.is_dir():
            self.log_message.emit(f"Path is not a directory: {root}")
            return

        folder_count = 0
        file_count = 0
        scan_roots = self._scan_roots_for_filter(root)
        if self.folder_filter:
            self.log_message.emit(
                f"Folder filter '{self.folder_filter}' selected {len(scan_roots)} root subfolder(s)."
            )
            for scan_root in scan_roots:
                self.log_message.emit(f"Folder filter root: {scan_root}")

        for scan_root in scan_roots:
            for dirpath, dirnames, filenames in os.walk(scan_root):
                self._cooperate()
                if self._cancelled:
                    return
                folder = Path(dirpath)
                folder_count += 1
                self._folders_seen = folder_count
                filenames.sort(key=str.lower)
                skipped_dirs = self._prune_subdirs(folder, dirnames, filenames)
                dirnames.sort(key=str.lower)
                self.log_message.emit(
                    f"Scanning filesystem folder {folder_count}: {folder} "
                    f"({len(filenames)} files, {len(dirnames)} subfolders)"
                )
                self._progress(f"folder {folder_count}: {folder.name}")
                for skipped_dir, reason in skipped_dirs:
                    self.log_message.emit(f"Skipping folder: {skipped_dir} ({reason})")
                for filename in filenames:
                    self._cooperate()
                    if self._cancelled:
                        return
                    path = folder / filename
                    if path.suffix.lower() in LEICA_PROJECT_EXTENSIONS and not self.include_project_files:
                        self.log_message.emit(f"Skipping Leica LOF/XLEF/XLIF file because option is off: {path}")
                        continue
                    if not self._is_allowed_leica_container(path):
                        continue
                    file_count += 1
                    self._containers_seen += 1
                    self.log_message.emit(f"Found Leica file {file_count}: {path}")
                    self._progress(f"found {path.name}")
                    yield path
        self.log_message.emit(f"Filesystem scan finished: {folder_count} folders, {file_count} Leica files.")

    def _scan_roots_for_filter(self, root: Path) -> list[Path]:
        if not self.folder_filter:
            return [root]
        try:
            children = sorted(root.iterdir(), key=lambda p: p.name.lower())
        except OSError as exc:
            self.log_message.emit(f"Could not list root folder for filter: {root}: {exc}")
            return []
        patterns = self._folder_filter_patterns(self.folder_filter)
        if not patterns:
            return [root]
        selected = []
        for child in children:
            if self._cancelled:
                break
            if not child.is_dir():
                continue
            name = child.name.lower()
            if any(fnmatchcase(name, pattern) for pattern in patterns):
                selected.append(child)
        return selected

    def _folder_filter_patterns(self, text: str) -> list[str]:
        patterns: list[str] = []
        for token in re.split(r"[;,]", str(text or "")):
            token = token.strip().lower()
            if not token:
                continue
            if ":" in token:
                expanded = self._expand_pattern_range(token)
                if expanded:
                    patterns.extend(expanded)
                    continue
            patterns.append(token)
        deduped: list[str] = []
        seen: set[str] = set()
        for pattern in patterns:
            if pattern not in seen:
                deduped.append(pattern)
                seen.add(pattern)
        return deduped

    def _expand_pattern_range(self, token: str) -> list[str]:
        start, end, *rest = token.split(":")
        if rest:
            return []
        start = start.strip().lower()
        end = end.strip().lower()
        match_start = re.fullmatch(r"([a-z])(\*)", start)
        match_end = re.fullmatch(r"([a-z])(\*)", end)
        if not match_start or not match_end:
            return []
        a = ord(match_start.group(1))
        b = ord(match_end.group(1))
        if a > b:
            a, b = b, a
        return [f"{chr(code)}*" for code in range(a, b + 1)]

    def _prune_subdirs(self, folder: Path, dirnames: list[str], filenames: list[str]) -> list[tuple[Path, str]]:
        skipped: list[tuple[Path, str]] = []
        keep: list[str] = []

        contains_xlif_project = any(Path(name).suffix.lower() in {".xlif", ".xlef"} for name in filenames)
        if contains_xlif_project:
            for dirname in dirnames:
                skipped.append((folder / dirname, "parent folder contains an XLIF/XLEF project file"))
            dirnames[:] = []
            return skipped

        for dirname in dirnames:
            name = dirname.lower()
            if name.endswith(".zarr"):
                skipped.append((folder / dirname, "Zarr folder"))
            elif any(part in name for part in SKIP_FOLDER_NAME_PARTS):
                skipped.append((folder / dirname, "metadata/pyramid folder"))
            else:
                keep.append(dirname)
        dirnames[:] = keep
        return skipped

    def _is_allowed_leica_container(self, path: Path) -> bool:
        suffix = path.suffix.lower()
        if suffix in LEICA_LIF_EXTENSIONS:
            return True
        if suffix in LEICA_PROJECT_EXTENSIONS:
            return self.include_project_files
        return False

    def _walk_container_root_only(self, gateway: Any, container_node: Any) -> int:
        """Scan direct root images from a Leica container.

        Leica stores some root-level series as images and others as shallow
        folders that contain the real same-named image. Collections-off means
        scan root images and same-named root wrappers, while skipping real
        collection folders such as "Test".
        """
        children = list(getattr(container_node, "children", []) or [])
        count = 0
        skipped = 0
        for child in children:
            if self._cancelled:
                break
            if getattr(child, "context", None) is not None:
                found = self._walk_root_image_node(gateway, child)
            else:
                found = self._walk_root_wrapper_if_same_named(gateway, child)
            if found <= 0:
                skipped += 1
                self.log_message.emit(
                    f"Skipping Leica collection/root entry because collections are off or no usable root image was found: "
                    f"{getattr(child, 'internal_path', getattr(child, 'name', ''))}"
                )
                continue
            count += found
        if skipped:
            self.log_message.emit(f"Skipped {skipped} Leica collection/root entries because collections are off.")
        if count <= 0:
            self.log_message.emit(
                f"No direct root image found in {getattr(container_node, 'path', getattr(container_node, 'name', ''))}; "
                "turn on 'Scan LIF images in Collections' to inspect collections/folders."
            )
        return count

    def _walk_root_image_node(self, gateway: Any, node: Any) -> int:
        if self._cancelled:
            return 0
        if getattr(node, "context", None) is not None:
            if self._looks_like_auxiliary_image(node) or self._looks_like_snapshot(node):
                return 0
            return self._walk_node(gateway, node)
        return 0

    def _walk_root_wrapper_if_same_named(self, gateway: Any, node: Any) -> int:
        if self._cancelled:
            return 0
        wrapper_name = self._normalized_leica_name(getattr(node, "name", "") or "")
        children = list(getattr(node, "children", []) or [])
        if not children:
            try:
                path = getattr(node, "path", None)
                image_id = getattr(node, "image_id", None)
                if path and image_id:
                    self.log_message.emit(
                        f"Expanding root Leica entry to check whether it wraps a root image: "
                        f"{getattr(node, 'internal_path', getattr(node, 'name', ''))}"
                    )
                    children = gateway.children_for_folder(
                        path,
                        image_id,
                        getattr(node, "internal_path", getattr(node, "name", "")),
                    )
            except Exception as exc:
                self.log_message.emit(
                    f"Could not expand Leica root entry {getattr(node, 'name', '')}: {exc}"
                )
                return 0

        for child in children:
            if getattr(child, "context", None) is None:
                continue
            if self._looks_like_auxiliary_image(child) or self._looks_like_snapshot(child):
                continue
            child_name = self._normalized_leica_name(getattr(child, "name", "") or "")
            if child_name == wrapper_name:
                return self._walk_root_image_node(gateway, child)
        names = ", ".join(str(getattr(child, "name", "")) for child in children[:5])
        self.log_message.emit(
            f"Root folder is treated as a collection, not a same-named root wrapper: "
            f"{getattr(node, 'internal_path', getattr(node, 'name', ''))}"
            f"{f' (children: {names})' if names else ''}"
        )
        return 0

    def _normalized_leica_name(self, value: str) -> str:
        return re.sub(r"\s+", " ", str(value or "").strip().lower())

    def _looks_like_auxiliary_image(self, node: Any) -> bool:
        context = getattr(node, "context", None)
        name = str(getattr(node, "name", "") or getattr(node, "internal_path", "")).lower()
        if "climatedatagraph" in name or "climate data graph" in name:
            return True
        if context is None:
            return False
        size_x = int(getattr(context, "size_x", 0) or 0)
        size_y = int(getattr(context, "size_y", 0) or 0)
        size_t = int(getattr(context, "size_t", 0) or 0)
        return size_x <= 1 and size_y <= 1 and size_t > 1

    def _looks_like_snapshot(self, node: Any) -> bool:
        name = str(getattr(node, "name", "") or getattr(node, "internal_path", "")).lower()
        return "snapshot" in name

    def _walk_node(self, gateway: Any, node: Any) -> int:
        self._cooperate()
        if self._cancelled:
            return 0
        kind = str(getattr(node, "kind", "") or "")
        path = getattr(node, "path", None)
        internal_path = str(getattr(node, "internal_path", "") or getattr(node, "name", ""))
        if kind in {"folder", "container"}:
            if path:
                self.log_message.emit(f"Scanning {kind}: {path} :: {internal_path}")
            else:
                self.log_message.emit(f"Scanning {kind}: {internal_path}")
        if getattr(node, "warning", None):
            self.log_message.emit(f"Warning for {getattr(node, 'path', '')}: {node.warning}")

        count = 0
        context = getattr(node, "context", None)
        if context is not None:
            if self._looks_like_snapshot(node):
                self.log_message.emit(
                    f"Skipping snapshot image: {getattr(context, 'container_path', '')} :: "
                    f"{getattr(context, 'internal_path', getattr(context, 'name', ''))}"
                )
                return 0
            if self._looks_like_auxiliary_image(node):
                self.log_message.emit(
                    f"Skipping auxiliary Leica image: {getattr(context, 'container_path', '')} :: "
                    f"{getattr(context, 'internal_path', getattr(context, 'name', ''))}"
                )
                return 0
            try:
                self.log_message.emit(
                    f"Reading metadata: {getattr(context, 'container_path', '')} :: "
                    f"{getattr(context, 'internal_path', getattr(context, 'name', ''))}"
                )
                t_meta = time.monotonic()
                hydrated = gateway.hydrate_image_node(node) or context
                record = make_record(hydrated, self.tolerance)
                meta_elapsed = time.monotonic() - t_meta
                if meta_elapsed >= 2.0:
                    self.log_message.emit(
                        f"Metadata read took {format_duration(meta_elapsed)}: "
                        f"{getattr(context, 'container_path', '')} :: "
                        f"{getattr(context, 'internal_path', getattr(context, 'name', ''))}"
                    )
                if self._skip_by_position_dimension(record):
                    return 0
                if self._skip_by_xy_size(record):
                    return 0
                if self.create_thumbnails:
                    _eligible, record.thumbnail_note = thumbnail_candidate_note(record, self.thumbnail_max_xy_pixels)
                else:
                    record.thumbnail_note = "thumbnail creation disabled"
                thumb = f"thumbnail: {record.thumbnail_path}" if record.thumbnail_path else f"no thumbnail ({record.thumbnail_note or 'not created'})"
                self.log_message.emit(
                    f"Found {record.nyquist.status}: {record.file_path.name} :: "
                    f"{record.internal_path} ({record.size_x}x{record.size_y}x{record.size_z}, {thumb})"
                )
                self._records_found += 1
                self._progress(f"image {self._records_found}: {record.image_name or record.internal_path}")
                self.record_found.emit(record)
                return 1
            except Exception as exc:
                self.log_message.emit(
                    f"Could not read metadata for {getattr(node, 'internal_path', node.name)}: {exc}"
                )
                return 0

        children = list(getattr(node, "children", []) or [])
        is_leica_internal_folder = (
            str(getattr(node, "kind", "")).lower() == "folder"
            and getattr(node, "path", None) is not None
            and Path(str(getattr(node, "path"))).suffix.lower() in {".lif", ".xlef", ".lof"}
            and getattr(node, "image_id", None)
        )
        if is_leica_internal_folder and not self.scan_collections:
            self.log_message.emit(f"Skipping Leica collection/subfolder: {node.path} :: {internal_path}")
            return 0
        if (
            not children
            and is_leica_internal_folder
        ):
            try:
                self.log_message.emit(f"Expanding Leica folder: {node.path} :: {internal_path}")
                children = gateway.children_for_folder(
                    node.path,
                    node.image_id,
                    getattr(node, "internal_path", node.name),
                )
            except Exception as exc:
                self.log_message.emit(f"Could not expand Leica folder {node.name}: {exc}")

        for child in children:
            count += self._walk_node(gateway, child)
            if self._cancelled:
                break
            self._cooperate()
        return count

    def _skip_by_xy_size(self, record: LeicaRecord) -> bool:
        if self.max_xy_pixels <= 0:
            return False
        size_x = int(record.size_x or 0)
        size_y = int(record.size_y or 0)
        if size_x <= 0 or size_y <= 0:
            return False
        if max(size_x, size_y) <= self.max_xy_pixels:
            return False
        self.log_message.emit(
            f"Skipping by XY size filter > {self.max_xy_pixels}px: "
            f"{record.file_path.name} :: {record.internal_path} ({size_x}x{size_y})"
        )
        return True

    def _skip_by_position_dimension(self, record: LeicaRecord) -> bool:
        count = position_dimension_count(record)
        if count <= 1:
            return False
        self.log_message.emit(
            f"Skipping position/tile image with S={count}: "
            f"{record.file_path.name} :: {record.internal_path}"
        )
        return True

    def _progress(self, phase: str) -> None:
        if self._started_at <= 0:
            elapsed = 0.0
        else:
            elapsed = max(0.0, time.monotonic() - self._started_at)
        eta = None
        estimated_total = None
        if self._containers_seen > 0 and self._containers_done > 0:
            average = elapsed / max(self._containers_done, 1)
            estimated_total = average * max(self._containers_seen, self._containers_done)
            eta = max(0.0, estimated_total - elapsed)
        elif self._containers_seen > 0:
            estimated_total = None
            eta = None
        parts = [
            f"Elapsed {format_duration(elapsed)}",
            f"Folders {self._folders_seen}",
            f"Files {self._containers_done}/{self._containers_seen}",
            f"Images {self._records_found}",
        ]
        if estimated_total is not None and eta is not None:
            parts.append(f"Est. total {format_duration(estimated_total)}")
            parts.append(f"ETA {format_duration(eta)}")
        else:
            parts.append("Est. total calculating")
            parts.append("ETA calculating")
        parts.append(phase)
        self.progress_message.emit(" | ".join(parts))
        self._cooperate()


class ThumbnailWorker(QThread):
    thumbnail_ready = pyqtSignal(int, str, str)
    log_message = pyqtSignal(str)
    finished_thumbnails = pyqtSignal(int)

    def __init__(
        self,
        records: list[LeicaRecord],
        indices: Iterable[int],
        max_xy_pixels: int,
        parent: Any = None,
    ) -> None:
        super().__init__(parent)
        self.records = records
        self.indices = list(indices)
        self.max_xy_pixels = max(0, int(max_xy_pixels))
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        count = 0
        total = len(self.indices)
        for pos, index in enumerate(self.indices, start=1):
            if self._cancelled:
                break
            if index < 0 or index >= len(self.records):
                continue
            record = self.records[index]
            if record.thumbnail_path and record.thumbnail_path.exists():
                continue
            self.log_message.emit(
                f"Creating thumbnail {pos}/{total}: {record.file_path.name} :: {record.internal_path}"
            )
            path = thumbnail_for_record(record, self.max_xy_pixels)
            self.thumbnail_ready.emit(index, str(path) if path else "", record.thumbnail_note)
            count += 1
        self.finished_thumbnails.emit(count)


class ConvertWorker(QThread):
    log_message = pyqtSignal(str)
    progress = pyqtSignal(int, int)
    finished_convert = pyqtSignal(int, int)

    def __init__(self, records: list[LeicaRecord], output_dir: Path, parent: Any = None) -> None:
        super().__init__(parent)
        self.records = records
        self.output_dir = output_dir

    def run(self) -> None:
        ok = 0
        failed = 0
        write_tczyx_ome_tiff = _resolve_ome_tiff_writer(self.log_message.emit)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        total = len(self.records)
        for idx, record in enumerate(self.records, start=1):
            self.progress.emit(idx - 1, total)
            out_path = self.output_dir / safe_output_name(record)
            try:
                self.log_message.emit(f"Reading pixels: {record.internal_path}")
                arr = np.asarray(record.context.open().read_array(s=getattr(record.context, "selected_s", None)))
                if arr.ndim != 5:
                    raise ValueError(f"Expected TCZYX from Leica reader, got shape {arr.shape}")
                meta = metadata_for_ome_tiff(record)
                self.log_message.emit(f"Writing OME-TIFF: {out_path}")
                write_tczyx_ome_tiff(arr, out_path, metadata=meta, levels=1, compression="lzw")
                write_sidecar_text(record, out_path, arr.shape)
                ok += 1
            except Exception as exc:
                failed += 1
                self.log_message.emit(f"FAILED {record.internal_path}: {exc}\n{traceback.format_exc()}")
            self.progress.emit(idx, total)
        self.finished_convert.emit(ok, failed)


def make_record(context: Any, tolerance: float) -> LeicaRecord:
    metadata = dict(getattr(context, "metadata", {}) or {})
    size_c = _positive_int(getattr(context, "size_c", None), metadata.get("channels"), 1)
    excitation = _positive_numbers(
        metadata.get("excitation"),
        metadata.get("excitation_wavelengths"),
        metadata.get("excitation_wavelength"),
        metadata.get("ExcitationWavelength"),
    )
    emission = _positive_numbers(
        metadata.get("emission"),
        metadata.get("emission_wavelengths"),
        metadata.get("emission_wavelength"),
        metadata.get("EmissionWavelength"),
    )
    channel_names = list(getattr(context, "channel_names", None) or metadata.get("channel_names") or [])
    if len(channel_names) < size_c:
        channel_names.extend(f"Ch{i}" for i in range(len(channel_names), size_c))

    microscope_type = str(
        metadata.get("microscope_type")
        or metadata.get("mic_type2")
        or metadata.get("mic_type")
        or metadata.get("MicroscopeType")
        or ""
    )
    na = _first_float(
        metadata.get("na"),
        metadata.get("NA"),
        metadata.get("numerical_aperture"),
        metadata.get("NumericalAperture"),
        metadata.get("objective_na"),
    )
    refractive_index = _first_float(
        metadata.get("refractiveindex"),
        metadata.get("refractive_index"),
        metadata.get("RefractiveIndex"),
        metadata.get("objective_refractive_index"),
        metadata.get("immersion_refractive_index"),
    )
    if refractive_index is None:
        refractive_index = infer_refractive_index(metadata)

    px = _first_float(
        getattr(context, "pixel_size_x_um", None),
        metadata.get("xres2"),
        metadata.get("pixel_size_x_um"),
        metadata.get("PhysicalSizeX"),
    )
    py = _first_float(
        getattr(context, "pixel_size_y_um", None),
        metadata.get("yres2"),
        metadata.get("pixel_size_y_um"),
        metadata.get("PhysicalSizeY"),
    )
    pz = _first_float(
        getattr(context, "pixel_size_z_um", None),
        metadata.get("zres2"),
        metadata.get("pixel_size_z_um"),
        metadata.get("PhysicalSizeZ"),
    )

    size_z = _positive_int(getattr(context, "size_z", None), metadata.get("zs"), 1)
    is_confocal = _looks_confocal(microscope_type, metadata)
    nyquist = classify_nyquist(
        pixel_x_um=px,
        pixel_y_um=py,
        pixel_z_um=pz,
        size_z=size_z,
        na=na,
        refractive_index=refractive_index,
        excitation_nm=excitation,
        emission_nm=emission,
        pinhole_au=_pinhole_airy_for_metadata(metadata),
        is_confocal=is_confocal,
        tolerance=tolerance,
    )
    return LeicaRecord(
        context=context,
        file_path=Path(getattr(context, "container_path", "")),
        internal_path=str(getattr(context, "internal_path", "") or getattr(context, "name", "")),
        image_name=str(getattr(context, "name", "") or metadata.get("name") or metadata.get("ElementName") or ""),
        size_x=_positive_int(getattr(context, "size_x", None), metadata.get("xs"), None),
        size_y=_positive_int(getattr(context, "size_y", None), metadata.get("ys"), None),
        size_z=size_z,
        size_c=size_c,
        size_t=_positive_int(getattr(context, "size_t", None), metadata.get("ts"), 1),
        pixel_x_um=px,
        pixel_y_um=py,
        pixel_z_um=pz,
        na=na,
        refractive_index=refractive_index,
        excitation_nm=excitation,
        emission_nm=emission,
        microscope_type=microscope_type or ("confocal" if is_confocal else ""),
        channel_names=channel_names[:size_c],
        metadata=metadata,
        nyquist=nyquist,
    )


def thumbnail_for_record(record: LeicaRecord, max_xy_pixels: int = 2048) -> Path | None:
    eligible, note = thumbnail_candidate_note(record, max_xy_pixels)
    if not eligible:
        record.thumbnail_note = note
        return None
    try:
        from leica_browser_qt.preview import preview_png_from_metadata

        path = preview_png_from_metadata(
            record.metadata,
            selected_s=getattr(record.context, "selected_s", None),
            preview_height=96,
            use_memmap=True,
        )
        if not path.exists():
            record.thumbnail_note = f"preview path does not exist: {path}"
            return None
        record.thumbnail_note = "created"
        return path
    except Exception as exc:
        record.thumbnail_note = str(exc)
        record.nyquist.notes.append(f"thumbnail unavailable: {exc}")
        return None


def thumbnail_candidate_note(record: LeicaRecord, max_xy_pixels: int = 2048) -> tuple[bool, str]:
    size_x = int(record.size_x or 0)
    size_y = int(record.size_y or 0)
    if size_x <= 0 or size_y <= 0:
        return False, "missing XY size"
    if max_xy_pixels > 0 and (size_x > max_xy_pixels or size_y > max_xy_pixels):
        return False, f"XY size {size_x}x{size_y} exceeds thumbnail limit {max_xy_pixels}px"
    return True, "thumbnail queued"


def position_dimension_count(record: LeicaRecord) -> int:
    metadata = record.metadata or {}
    dimensions = metadata.get("dimensions") if isinstance(metadata.get("dimensions"), dict) else {}
    values = [
        getattr(record.context, "size_s", None),
        metadata.get("size_s"),
        metadata.get("tiles"),
        dimensions.get("s") if isinstance(dimensions, dict) else None,
    ]
    counts = [_positive_int(value, None) for value in values]
    return max([count for count in counts if count is not None] or [1])


def classify_nyquist(
    *,
    pixel_x_um: float | None,
    pixel_y_um: float | None,
    pixel_z_um: float | None,
    size_z: int | None,
    na: float | None,
    refractive_index: float | None,
    excitation_nm: list[float],
    emission_nm: list[float],
    pinhole_au: float | None,
    is_confocal: bool,
    tolerance: float,
) -> NyquistResult:
    notes: list[str] = []
    if not is_confocal:
        notes.append("not marked confocal")

    wavelength = min(excitation_nm) if excitation_nm else None
    source = "excitation"
    if wavelength is None and emission_nm:
        wavelength = min(emission_nm)
        source = "emission fallback"
        notes.append("used emission wavelength fallback")

    missing = []
    if wavelength is None:
        missing.append("wavelength")
    if na is None or na <= 0:
        missing.append("NA")
    if refractive_index is None or refractive_index <= 0:
        missing.append("refractive index")
    if pixel_x_um is None or pixel_y_um is None:
        missing.append("XY pixel size")
    if missing:
        return NyquistResult("unknown", wavelength_nm=wavelength, wavelength_source=source, notes=notes + missing)

    assert wavelength is not None and na is not None and refractive_index is not None
    if na >= refractive_index:
        return NyquistResult(
            "unknown",
            wavelength_nm=wavelength,
            wavelength_source=source,
            pinhole_au=pinhole_au,
            notes=notes + [f"NA >= refractive index ({na:g} >= {refractive_index:g})"],
        )

    alpha = math.asin(na / refractive_index)
    sin_alpha = math.sin(alpha)
    cos_alpha = math.cos(alpha)
    if sin_alpha <= 0 or (1.0 - cos_alpha) <= 0:
        return NyquistResult(
            "unknown",
            wavelength_nm=wavelength,
            wavelength_source=source,
            pinhole_au=pinhole_au,
            notes=notes + ["invalid optics"],
        )

    if is_confocal:
        if pinhole_au is None:
            pinhole_au = DEFAULT_PINHOLE_AIRY
            notes.append("used 1.0 AU pinhole fallback")
        cxy, cz = _nyquist_confocal_pinhole_factors(pinhole_au)
        limit_xy = (cxy * wavelength / (8.0 * na)) / 1000.0
        limit_z = (cz * wavelength / (4.0 * refractive_index * (1.0 - cos_alpha))) / 1000.0
    else:
        cxy, cz = 1.0, 1.0
        limit_xy = (wavelength / (4.0 * na)) / 1000.0
        limit_z = (wavelength / (2.0 * refractive_index * (1.0 - cos_alpha))) / 1000.0
    actual_xy = max(pixel_x_um, pixel_y_um)
    ratio_xy = actual_xy / limit_xy if limit_xy > 0 else None
    ratio_z = None
    xy_ok = ratio_xy is not None and ratio_xy <= 1.0
    z_ok: bool | None = None
    if int(size_z or 1) > 1:
        if pixel_z_um is None:
            notes.append("missing Z pixel size")
        else:
            ratio_z = pixel_z_um / limit_z if limit_z > 0 else None
            z_ok = ratio_z is not None and ratio_z <= 1.0
    else:
        z_ok = True

    if ratio_xy is None:
        status = "unknown"
    elif xy_ok and (z_ok is True):
        status = "pass"
    elif xy_ok:
        status = "xy only"
    else:
        ratios = [r for r in (ratio_xy, ratio_z) if r is not None]
        worst = max(ratios) if ratios else float("inf")
        if worst <= 1.0:
            status = "pass"
        elif worst <= tolerance:
            status = "near"
        else:
            status = "too coarse"
    return NyquistResult(status, limit_xy, limit_z, ratio_xy, ratio_z, xy_ok, z_ok, pinhole_au, cxy, cz, wavelength, source, notes)


def _interpolate_pinhole_factor(pinhole_au: float, points: tuple[tuple[float, float], ...]) -> float:
    pinhole_au = max(float(pinhole_au), 0.0)
    if pinhole_au <= points[0][0]:
        x0, y0 = 0.0, 1.0
        x1, y1 = points[0]
        if x1 <= x0:
            return y1
        return y0 + (y1 - y0) * ((pinhole_au - x0) / (x1 - x0))
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if pinhole_au <= x1:
            if math.isclose(x0, x1):
                return y1
            return y0 + (y1 - y0) * ((pinhole_au - x0) / (x1 - x0))
    return points[-1][1]


def _nyquist_confocal_pinhole_factors(pinhole_au: float | None) -> tuple[float, float]:
    if pinhole_au is None or float(pinhole_au) <= 0:
        return 1.0, 1.0
    cxy = _interpolate_pinhole_factor(float(pinhole_au), ((0.5, 1.2), (0.68, 1.4), (1.0, 1.7)))
    cz = _interpolate_pinhole_factor(float(pinhole_au), ((0.5, 1.0), (0.68, 1.1), (1.0, 1.3)))
    return cxy, cz


def _pinhole_airy_for_metadata(metadata: dict[str, Any]) -> float | None:
    candidates = [
        metadata.get("pinhole_airy_units"),
        metadata.get("pinhole_airy"),
        metadata.get("pinholeairyunits"),
        metadata.get("PinholeAiryUnits"),
    ]
    channels = metadata.get("channels")
    if isinstance(channels, dict):
        channels = [channels]
    if isinstance(channels, (list, tuple)):
        for ch in channels:
            if isinstance(ch, dict):
                candidates.extend(
                    [
                        ch.get("pinhole_airy_units"),
                        ch.get("pinhole_airy"),
                        ch.get("pinhole_airy_units_from_metadata"),
                    ]
                )
    values = _positive_numbers(*candidates)
    return values[0] if values else None


def _looks_confocal(microscope_type: str, metadata: dict[str, Any]) -> bool:
    text = microscope_type.lower()
    if "conf" in text or "scanner" in text:
        return True
    for key in ("microscope_type", "mic_type2", "mic_type", "MicroscopeType", "acquisition_mode", "detector_type"):
        value = str(metadata.get(key, "")).lower()
        if "conf" in value or "scanner" in value:
            return True
    if _pinhole_airy_for_metadata(metadata) is not None:
        return True
    if _positive_numbers(metadata.get("pinhole_size"), metadata.get("PinholeSize")):
        return True
    channels = metadata.get("channels")
    if isinstance(channels, dict):
        channels = [channels]
    if isinstance(channels, (list, tuple)):
        for channel in channels:
            if not isinstance(channel, dict):
                continue
            mode = str(channel.get("acquisition_mode") or channel.get("mode") or "").lower()
            if "conf" in mode or "scanner" in mode:
                return True
            if _positive_numbers(
                channel.get("pinhole_size"),
                channel.get("PinholeSize"),
                channel.get("pinhole_airy_units"),
                channel.get("pinhole_airy"),
                channel.get("pinhole_airy_units_from_metadata"),
            ):
                return True
    return False


def _positive_int(*values: Any) -> int | None:
    for value in values:
        try:
            if value is None or value == "":
                continue
            number = int(float(value))
        except (TypeError, ValueError):
            continue
        if number > 0:
            return number
    return None


def _nonnegative_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        number = int(float(value))
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None


def _numbers_from_value(value: Any) -> list[float]:
    if value is None:
        return []
    if isinstance(value, dict):
        out: list[float] = []
        for sub in value.values():
            out.extend(_numbers_from_value(sub))
        return out
    if isinstance(value, (list, tuple, set)):
        out = []
        for sub in value:
            out.extend(_numbers_from_value(sub))
        return out
    if isinstance(value, (int, float)):
        return [float(value)] if float(value) > 0 else []
    text = str(value)
    return [float(match) for match in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text) if float(match) > 0]


def _positive_numbers(*values: Any) -> list[float]:
    seen = set()
    out = []
    for value in values:
        for number in _numbers_from_value(value):
            rounded = round(number, 6)
            if rounded not in seen:
                seen.add(rounded)
                out.append(float(number))
    return out


def _first_float(*values: Any) -> float | None:
    for value in values:
        numbers = _numbers_from_value(value)
        if numbers:
            return numbers[0]
    return None


def infer_refractive_index(metadata: dict[str, Any]) -> float | None:
    text = " ".join(str(metadata.get(k, "")) for k in ("immersion", "objective_name", "ObjectiveName", "medium")).lower()
    if "oil" in text:
        return 1.515
    if "water" in text:
        return 1.33
    if "glycer" in text:
        return 1.47
    return None


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}".rstrip("0").rstrip(".")
    return str(value)


def fmt_list(values: Iterable[float]) -> str:
    return ", ".join(fmt(v, 1) for v in values)


def format_duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(float(seconds)):
        return "calculating"
    total = max(0, int(round(float(seconds))))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:d}:{secs:02d}"


def safe_output_name(record: LeicaRecord) -> str:
    stem = record.file_path.stem or "leica"
    label = f"{stem}_{record.internal_path or record.image_name}"
    label = re.sub(r"[^\w.-]+", "_", label, flags=re.UNICODE).strip("_")
    if not label:
        label = "leica_image"
    return f"{label[:180]}.ome.tiff"


def metadata_for_ome_tiff(record: LeicaRecord) -> dict[str, Any]:
    channels = []
    for idx, name in enumerate(record.channel_names or []):
        ch: dict[str, Any] = {"name": name}
        if idx < len(record.emission_nm):
            ch["emission_wavelength"] = record.emission_nm[idx]
        elif record.emission_nm:
            ch["emission_wavelength"] = record.emission_nm[0]
        if idx < len(record.excitation_nm):
            ch["excitation_wavelength"] = record.excitation_nm[idx]
        elif record.excitation_nm:
            ch["excitation_wavelength"] = record.excitation_nm[0]
        if record.nyquist.pinhole_au is not None:
            ch["pinhole_airy_units"] = record.nyquist.pinhole_au
        channels.append(ch)

    return {
        "name": record.image_name or record.file_path.stem,
        "source_file": str(record.file_path),
        "source_internal_path": record.internal_path,
        "pixel_size_x": record.pixel_x_um,
        "pixel_size_y": record.pixel_y_um,
        "pixel_size_z": record.pixel_z_um,
        "channel_names": record.channel_names,
        "channels": channels,
        "na": record.na,
        "refractive_index": record.refractive_index,
        "microscope_type": record.microscope_type,
        "leica_metadata": _json_safe(record.metadata),
        "nyquist": {
            "status": record.nyquist.status,
            "limit_xy_um": record.nyquist.limit_xy_um,
            "limit_z_um": record.nyquist.limit_z_um,
            "ratio_xy": record.nyquist.ratio_xy,
            "ratio_z": record.nyquist.ratio_z,
            "xy_ok": record.nyquist.xy_ok,
            "z_ok": record.nyquist.z_ok,
            "pinhole_au": record.nyquist.pinhole_au,
            "cxy": record.nyquist.cxy,
            "cz": record.nyquist.cz,
            "wavelength_nm": record.nyquist.wavelength_nm,
            "wavelength_source": record.nyquist.wavelength_source,
            "notes": record.nyquist.notes,
        },
    }


def _resolve_ome_tiff_writer(log_func: Any):
    try:
        from cideconvolve_io.ome_tiff_io import write_tczyx_ome_tiff

        return write_tczyx_ome_tiff
    except Exception as exc:
        log_func(f"Shared OME-TIFF writer unavailable, using local tifffile fallback: {exc}")
        return write_tczyx_ome_tiff_fallback


def write_tczyx_ome_tiff_fallback(
    data: np.ndarray,
    path: str | Path,
    metadata: dict[str, Any] | None = None,
    *,
    levels: int = 1,
    compression: str | None = "lzw",
) -> Path:
    try:
        import tifffile
    except Exception as exc:
        raise RuntimeError("Writing OME-TIFF requires tifffile in this environment") from exc

    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim != 5:
        raise ValueError(f"Expected TCZYX data for OME-TIFF export, got {arr.shape}")

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(metadata or {})
    ome_metadata = _fallback_ome_metadata(meta, arr.shape)
    description = _json_safe(meta)
    write_kwargs: dict[str, Any] = {
        "bigtiff": True,
        "ome": True,
        "metadata": ome_metadata,
    }
    if compression:
        write_kwargs["compression"] = compression
        write_kwargs["predictor"] = True
    try:
        tifffile.imwrite(out_path, arr, **write_kwargs, extratags=[(65000, "s", 0, json.dumps(description), True)])
    except Exception:
        if not compression:
            raise
        write_kwargs.pop("compression", None)
        write_kwargs.pop("predictor", None)
        tifffile.imwrite(out_path, arr, **write_kwargs, extratags=[(65000, "s", 0, json.dumps(description), True)])
    return out_path


def _fallback_ome_metadata(metadata: dict[str, Any], shape: tuple[int, ...]) -> dict[str, Any]:
    size_c = int(shape[1])
    channels = metadata.get("channels") or []
    if isinstance(channels, dict):
        channels = [channels]
    if not isinstance(channels, (list, tuple)):
        channels = []
    names = metadata.get("channel_names") or []
    if not isinstance(names, (list, tuple)):
        names = []
    channel_names = []
    emission = []
    excitation = []
    for idx in range(size_c):
        ch = channels[idx] if idx < len(channels) and isinstance(channels[idx], dict) else {}
        channel_names.append(str(ch.get("name") or (names[idx] if idx < len(names) else f"Ch{idx}")))
        emission.append(ch.get("emission_wavelength"))
        excitation.append(ch.get("excitation_wavelength"))

    channel_meta: dict[str, Any] = {"Name": channel_names}
    if emission and all(value is not None for value in emission):
        channel_meta["EmissionWavelength"] = [float(value) for value in emission]
        channel_meta["EmissionWavelengthUnit"] = ["nm"] * size_c
    if excitation and all(value is not None for value in excitation):
        channel_meta["ExcitationWavelength"] = [float(value) for value in excitation]
        channel_meta["ExcitationWavelengthUnit"] = ["nm"] * size_c

    return {
        "axes": "TCZYX",
        "Name": str(metadata.get("name") or "Leica export"),
        "PhysicalSizeX": _first_float(metadata.get("pixel_size_x")) or 1.0,
        "PhysicalSizeXUnit": "µm",
        "PhysicalSizeY": _first_float(metadata.get("pixel_size_y")) or (_first_float(metadata.get("pixel_size_x")) or 1.0),
        "PhysicalSizeYUnit": "µm",
        "PhysicalSizeZ": _first_float(metadata.get("pixel_size_z")) or 1.0,
        "PhysicalSizeZUnit": "µm",
        "Channel": channel_meta,
    }


def write_sidecar_text(record: LeicaRecord, out_path: Path, shape: tuple[int, ...]) -> Path:
    sidecar = out_path.with_name(out_path.name + ".txt")
    lines = [
        "Leica Nyquist Finder export",
        "",
        f"OME-TIFF: {out_path}",
        f"Original Leica file: {record.file_path}",
        f"Original Leica image: {record.internal_path}",
        f"Image name: {record.image_name}",
        f"Exported TCZYX shape: {shape}",
        "",
        "Sampling",
        f"Pixel X um: {fmt(record.pixel_x_um)}",
        f"Pixel Y um: {fmt(record.pixel_y_um)}",
        f"Pixel Z um: {fmt(record.pixel_z_um)}",
        f"Size X: {fmt(record.size_x)}",
        f"Size Y: {fmt(record.size_y)}",
        f"Size Z: {fmt(record.size_z)}",
        f"Channels: {fmt(record.size_c)}",
        f"Timepoints: {fmt(record.size_t)}",
        "",
        "Optics",
        f"Microscope type: {record.microscope_type}",
        f"NA: {fmt(record.na)}",
        f"Refractive index: {fmt(record.refractive_index)}",
        f"Excitation nm: {fmt_list(record.excitation_nm)}",
        f"Emission nm: {fmt_list(record.emission_nm)}",
        f"Pinhole AU: {fmt(record.nyquist.pinhole_au)}",
        "",
        "Nyquist",
        f"Status: {record.nyquist.status}",
        f"XY OK: {record.nyquist.xy_ok}",
        f"Z OK: {record.nyquist.z_ok}",
        f"XY limit um: {fmt(record.nyquist.limit_xy_um)}",
        f"Z limit um: {fmt(record.nyquist.limit_z_um)}",
        f"XY ratio: {fmt(record.nyquist.ratio_xy)}",
        f"Z ratio: {fmt(record.nyquist.ratio_z)}",
        f"Wavelength source: {record.nyquist.wavelength_source}",
        f"Notes: {'; '.join(record.nyquist.notes)}",
    ]
    sidecar.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return sidecar


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


class MainWindow(QMainWindow):
    COLUMNS = [
        "Use",
        "Thumb",
        "Status",
        "XY OK",
        "Z OK",
        "3D",
        "Confocal",
        "File",
        "Image",
        "Internal path",
        "Size X",
        "Size Y",
        "Size Z",
        "C",
        "T",
        "Pix X um",
        "Pix Y um",
        "Pix Z um",
        "NA",
        "RI",
        "Ex nm",
        "Em nm",
        "Pinhole AU",
        "Nyq XY um",
        "Nyq Z um",
        "Ratio XY",
        "Ratio Z",
        "Notes",
    ]

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Leica LIF Nyquist Finder")
        self.resize(1700, 900)
        self.records: list[LeicaRecord] = []
        self.completed_containers: set[str] = set()
        self.scan_worker: ScanWorker | None = None
        self.thumbnail_worker: ThumbnailWorker | None = None
        self.convert_worker: ConvertWorker | None = None
        self._convert_started_at = 0.0
        self._populating = False
        self._last_cache_save_at = 0.0
        self._cache_dirty = False
        self._deferred_cache_status = "in_progress"
        self._cache_save_timer = QTimer(self)
        self._cache_save_timer.setSingleShot(True)
        self._cache_save_timer.timeout.connect(self._flush_deferred_cache_save)
        self._thumbnail_pending: list[int] = []
        self._thumbnail_pending_set: set[int] = set()
        self._pending_log_lines: list[str] = []
        self._log_flush_timer = QTimer(self)
        self._log_flush_timer.setInterval(100)
        self._log_flush_timer.timeout.connect(self._flush_log)
        self._pending_records: list[LeicaRecord] = []
        self._record_flush_timer = QTimer(self)
        self._record_flush_timer.setInterval(50)
        self._record_flush_timer.timeout.connect(self._flush_pending_records)
        self._build_ui()
        self._restore_settings_from_cache(load_records=False)

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        root_row = QHBoxLayout()
        self.root_edit = QLineEdit()
        self.root_edit.setPlaceholderText("Folder containing Leica .lif files")
        root_browse = QPushButton("Browse...")
        root_browse.clicked.connect(self._browse_root)
        root_row.addWidget(QLabel("Leica root:"))
        root_row.addWidget(self.root_edit, 1)
        root_row.addWidget(root_browse)
        layout.addLayout(root_row)

        out_row = QHBoxLayout()
        self.output_edit = QLineEdit(str(Path.home() / "Downloads" / "leica_nyquist_exports"))
        out_browse = QPushButton("Browse...")
        out_browse.clicked.connect(self._browse_output)
        out_row.addWidget(QLabel("OME-TIFF output:"))
        out_row.addWidget(self.output_edit, 1)
        out_row.addWidget(out_browse)
        layout.addLayout(out_row)

        controls = QHBoxLayout()
        self.tolerance_spin = QDoubleSpinBox()
        self.tolerance_spin.setRange(1.0, 3.0)
        self.tolerance_spin.setSingleStep(0.05)
        self.tolerance_spin.setDecimals(2)
        self.tolerance_spin.setValue(1.25)
        self.tolerance_spin.setToolTip("Near Nyquist tolerance; 1.25 means up to 25% coarser than strict Nyquist.")
        self.max_xy_spin = QSpinBox()
        self.max_xy_spin.setRange(0, 200000)
        self.max_xy_spin.setSingleStep(512)
        self.max_xy_spin.setValue(2048)
        self.max_xy_spin.setToolTip("Maximum allowed X or Y pixel dimension. Use 0 for no XY size filter.")
        self.thumbnail_max_xy_spin = QSpinBox()
        self.thumbnail_max_xy_spin.setRange(0, 200000)
        self.thumbnail_max_xy_spin.setSingleStep(512)
        self.thumbnail_max_xy_spin.setValue(2048)
        self.thumbnail_max_xy_spin.setToolTip("Maximum X or Y pixel dimension for thumbnail creation. Use 0 for no thumbnail size limit.")
        self.folder_filter_edit = QLineEdit()
        self.folder_filter_edit.setMaximumWidth(160)
        self.folder_filter_edit.setPlaceholderText("e.g. a*")
        self.folder_filter_edit.setToolTip(
            "Optional wildcard filter for direct subfolders of the Leica root. "
            "Example: a* scans only matching root subfolders and everything below them. "
            "Use a*:d* for a range (a*, b*, c*, d*). "
            "Use a*;d* or a*,d* for specific patterns only."
        )
        self.scan_button = QPushButton("Scan")
        self.scan_button.clicked.connect(self._start_scan)
        self.cancel_button = QPushButton("Cancel scan")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self._cancel_scan)
        self.check_matching_button = QPushButton("Check pass/near")
        self.check_matching_button.clicked.connect(self._check_matching)
        self.convert_button = QPushButton("Convert checked")
        self.convert_button.clicked.connect(self._start_convert)
        self.restore_cache_button = QPushButton("Restore cache")
        self.restore_cache_button.setToolTip(f"Restore cached scan list from {CACHE_PATH}")
        self.restore_cache_button.clicked.connect(self._restore_cache_clicked)
        self.clear_cache_button = QPushButton("Clear cache")
        self.clear_cache_button.setToolTip(f"Delete cached scan list at {CACHE_PATH}")
        self.clear_cache_button.clicked.connect(self._clear_cache_clicked)
        self.scan_collections_box = QCheckBox("Scan LIF images in Collections")
        self.scan_collections_box.setChecked(False)
        self.scan_collections_box.setToolTip(
            "When off, scan only the first direct root image per LIF. "
            "When on, also scan images inside Leica Collections, positions, and folders."
        )
        self.thumbnails_box = QCheckBox("create thumbnails")
        self.thumbnails_box.setChecked(True)
        self.thumbnails_box.setToolTip("Queue thumbnails for visible entries after metadata scanning. Uses the thumbnail XY limit.")
        self.thumbnails_box.stateChanged.connect(lambda _state: self._populate_table())
        self.include_project_files_box = QCheckBox("include LOF/XLEF/XLIF")
        self.include_project_files_box.setChecked(False)
        self.include_project_files_box.setToolTip(
            "When off, only .lif files are scanned. Turn on to include Leica LOF/XLEF/XLIF project files."
        )
        controls.addWidget(QLabel("Near tolerance:"))
        controls.addWidget(self.tolerance_spin)
        controls.addWidget(QLabel("Max XY px:"))
        controls.addWidget(self.max_xy_spin)
        controls.addWidget(QLabel("Thumb XY px:"))
        controls.addWidget(self.thumbnail_max_xy_spin)
        controls.addWidget(QLabel("Folder filter:"))
        controls.addWidget(self.folder_filter_edit)
        controls.addSpacing(20)
        controls.addWidget(self.scan_button)
        controls.addWidget(self.cancel_button)
        controls.addWidget(self.check_matching_button)
        controls.addWidget(self.convert_button)
        controls.addWidget(self.restore_cache_button)
        controls.addWidget(self.clear_cache_button)
        controls.addSpacing(20)
        controls.addWidget(self.scan_collections_box)
        controls.addWidget(self.thumbnails_box)
        controls.addWidget(self.include_project_files_box)
        controls.addStretch(1)
        layout.addLayout(controls)

        filters = QHBoxLayout()
        self.show_pass = self._filter_box("pass", True)
        self.show_near = self._filter_box("xy only / near", True)
        self.show_too_coarse = self._filter_box("too coarse", False)
        self.show_unknown = self._filter_box("unknown", False)
        self.require_3d = self._filter_box("require 3D", True)
        self.require_confocal = self._filter_box("require confocal", True)
        for box in (
            self.show_pass,
            self.show_near,
            self.show_too_coarse,
            self.show_unknown,
            self.require_3d,
            self.require_confocal,
        ):
            filters.addWidget(box)
        filters.addStretch(1)
        layout.addLayout(filters)

        self.table = QTableWidget(0, len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(self.COLUMNS)
        self.table.setSortingEnabled(True)
        self.table.setIconSize(QSize(80, 80))
        self.table.verticalHeader().setDefaultSectionSize(84)
        self.table.itemChanged.connect(self._item_changed)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(7, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(9, QHeaderView.ResizeMode.Stretch)
        self.table.setColumnWidth(1, 90)
        layout.addWidget(self.table, 1)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        layout.addWidget(self.progress)

        self.progress_label = QLabel("Elapsed 0:00 | Est. total calculating | ETA calculating")
        self.progress_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.progress_label)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(2000)
        layout.addWidget(self.log, 0)

    def _filter_box(self, text: str, checked: bool) -> QCheckBox:
        box = QCheckBox(text)
        box.setChecked(checked)
        box.stateChanged.connect(lambda _state: self._populate_table())
        return box

    def _current_settings(self) -> dict[str, Any]:
        return {
            "root": self.root_edit.text().strip(),
            "output": self.output_edit.text().strip(),
            "near_tolerance": self.tolerance_spin.value(),
            "max_xy_px": self.max_xy_spin.value(),
            "thumbnail_max_xy_px": self.thumbnail_max_xy_spin.value(),
            "folder_filter": self.folder_filter_edit.text().strip(),
            "scan_collections": self.scan_collections_box.isChecked(),
            "create_thumbnails": self.thumbnails_box.isChecked(),
            "include_project_files": self.include_project_files_box.isChecked(),
            "show_pass": self.show_pass.isChecked(),
            "show_near": self.show_near.isChecked(),
            "show_too_coarse": self.show_too_coarse.isChecked(),
            "show_unknown": self.show_unknown.isChecked(),
            "require_3d": self.require_3d.isChecked(),
            "require_confocal": self.require_confocal.isChecked(),
        }

    def _scan_signature(self, settings: dict[str, Any] | None = None) -> dict[str, Any]:
        data = dict(settings or self._current_settings())
        data.pop("output", None)
        for key in (
            "show_pass",
            "show_near",
            "show_too_coarse",
            "show_unknown",
            "require_3d",
            "require_confocal",
            "create_thumbnails",
            "thumbnail_max_xy_px",
        ):
            data.pop(key, None)
        return data

    def _cache_matches_current_scan(self, payload: dict[str, Any]) -> bool:
        return self._scan_signature(payload.get("settings") or {}) == self._scan_signature()

    def _save_current_cache(self, status: str = "in_progress", *, force: bool = False) -> None:
        now = time.monotonic()
        is_checkpoint = status != "in_progress"
        if (
            not force
            and not is_checkpoint
            and self._last_cache_save_at > 0
            and now - self._last_cache_save_at < CACHE_SAVE_INTERVAL_SECONDS
        ):
            self._cache_dirty = True
            return
        try:
            save_cache(
                cache_payload(
                    settings=self._current_settings(),
                    records=self.records,
                    completed_containers=self.completed_containers,
                    status=status,
                )
            )
            self._last_cache_save_at = now
            self._cache_dirty = False
        except Exception as exc:
            self._log(f"Could not save scan cache: {exc}")

    def _schedule_cache_save(self, status: str = "in_progress", *, delay_ms: int = CACHE_EDIT_SAVE_DELAY_MS) -> None:
        self._cache_dirty = True
        self._deferred_cache_status = status
        if self._cache_save_timer.isActive():
            return
        self._cache_save_timer.start(max(0, int(delay_ms)))

    def _flush_deferred_cache_save(self) -> None:
        if not self._cache_dirty:
            return
        now = time.monotonic()
        wait_s = CACHE_SAVE_INTERVAL_SECONDS - (now - self._last_cache_save_at)
        if self._last_cache_save_at > 0 and wait_s > 0:
            self._cache_save_timer.start(max(250, int(wait_s * 1000)))
            return
        self._save_current_cache(status=self._deferred_cache_status, force=True)

    def _restore_settings_from_cache(self, *, load_records: bool) -> bool:
        payload = load_cache()
        if not payload:
            return False
        settings = payload.get("settings") or {}
        self.root_edit.setText(str(settings.get("root") or self.root_edit.text()))
        self.output_edit.setText(str(settings.get("output") or self.output_edit.text()))
        self.tolerance_spin.setValue(float(settings.get("near_tolerance", self.tolerance_spin.value())))
        self.max_xy_spin.setValue(int(settings.get("max_xy_px", self.max_xy_spin.value())))
        self.thumbnail_max_xy_spin.setValue(int(settings.get("thumbnail_max_xy_px", self.thumbnail_max_xy_spin.value())))
        self.folder_filter_edit.setText(str(settings.get("folder_filter") or ""))
        self.scan_collections_box.setChecked(bool(settings.get("scan_collections", self.scan_collections_box.isChecked())))
        self.thumbnails_box.setChecked(bool(settings.get("create_thumbnails", self.thumbnails_box.isChecked())))
        self.include_project_files_box.setChecked(bool(settings.get("include_project_files", self.include_project_files_box.isChecked())))
        self.show_pass.setChecked(bool(settings.get("show_pass", self.show_pass.isChecked())))
        self.show_near.setChecked(bool(settings.get("show_near", self.show_near.isChecked())))
        self.show_too_coarse.setChecked(bool(settings.get("show_too_coarse", self.show_too_coarse.isChecked())))
        self.show_unknown.setChecked(bool(settings.get("show_unknown", self.show_unknown.isChecked())))
        self.require_3d.setChecked(bool(settings.get("require_3d", self.require_3d.isChecked())))
        self.require_confocal.setChecked(bool(settings.get("require_confocal", self.require_confocal.isChecked())))
        if load_records:
            self.records = []
            skipped_positions = 0
            for item in payload.get("records", []) or []:
                try:
                    if isinstance(item, dict):
                        record = record_from_dict(item)
                        if position_dimension_count(record) > 1:
                            skipped_positions += 1
                            continue
                        self.records.append(record)
                except Exception as exc:
                    self._log(f"Could not restore cached record: {exc}")
            self.completed_containers = {str(Path(item)) for item in payload.get("completed_containers", []) or []}
            self._populate_table()
            self._log(
                f"Restored cached scan from {payload.get('updated_at', 'unknown time')}: "
                f"{len(self.records)} records, {len(self.completed_containers)} completed Leica containers."
            )
            if skipped_positions:
                self._log(f"Skipped {skipped_positions} cached position/tile image entries.")
            self.progress_label.setText(
                f"Restored cache | Records {len(self.records)} | Completed files {len(self.completed_containers)}"
            )
        return True

    def _restore_cache_clicked(self) -> None:
        self._cancel_thumbnail_worker()
        if not self._restore_settings_from_cache(load_records=True):
            QMessageBox.information(self, "No cache", f"No readable scan cache was found at:\n{CACHE_PATH}")

    def _clear_cache_clicked(self) -> None:
        self._cancel_thumbnail_worker()
        self.completed_containers.clear()
        self._thumbnail_pending.clear()
        self._thumbnail_pending_set.clear()
        try:
            CACHE_PATH.unlink(missing_ok=True)
        except Exception as exc:
            QMessageBox.warning(self, "Cache", f"Could not delete cache:\n{exc}")
            return
        self._log(f"Deleted scan cache: {CACHE_PATH}")
        self.progress_label.setText("Cache cleared")

    def _browse_root(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Choose Leica folder", self.root_edit.text() or str(Path.home()))
        if folder:
            self.root_edit.setText(folder)

    def _browse_output(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Choose OME-TIFF output folder", self.output_edit.text())
        if folder:
            self.output_edit.setText(folder)

    def _start_scan(self) -> None:
        root = Path(self.root_edit.text().strip())
        if not root.exists():
            QMessageBox.warning(self, "Missing folder", f"Path does not exist:\n{root}")
            return
        self._cancel_thumbnail_worker()
        existing = load_cache()
        if existing and not self._cache_matches_current_scan(existing):
            self.completed_containers.clear()
            self.records.clear()
            self._thumbnail_pending.clear()
            self._thumbnail_pending_set.clear()
            self._log("Scan settings differ from cached scan; starting a new cache.")
        elif not self.records and existing and self._cache_matches_current_scan(existing):
            self._restore_settings_from_cache(load_records=True)
        if not self.completed_containers:
            self.records.clear()
            self._thumbnail_pending.clear()
            self._thumbnail_pending_set.clear()
            self._populate_table()
        self.progress.setRange(0, 0)
        self.scan_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.table.setSortingEnabled(False)
        self._log(
            f"Starting scan with near tolerance {self.tolerance_spin.value():.2f}; "
            f"max XY px={self.max_xy_spin.value() or 'unlimited'}; "
            f"thumbnail max XY px={self.thumbnail_max_xy_spin.value() or 'unlimited'}; "
            f"folder filter={self.folder_filter_edit.text().strip() or '<none>'}; "
            f"scan LIF images in Collections={self.scan_collections_box.isChecked()}; "
            f"lazy thumbnails={self.thumbnails_box.isChecked()}; "
            f"include LOF/XLEF/XLIF={self.include_project_files_box.isChecked()}"
        )
        self._save_current_cache(status="starting", force=True)
        self.scan_worker = ScanWorker(
            root,
            self.tolerance_spin.value(),
            scan_collections=self.scan_collections_box.isChecked(),
            create_thumbnails=self.thumbnails_box.isChecked(),
            include_project_files=self.include_project_files_box.isChecked(),
            max_xy_pixels=self.max_xy_spin.value(),
            thumbnail_max_xy_pixels=self.thumbnail_max_xy_spin.value(),
            folder_filter=self.folder_filter_edit.text(),
            completed_containers=self.completed_containers,
            parent=self,
        )
        self.scan_worker.record_found.connect(self._queue_record)
        self.scan_worker.container_completed.connect(self._container_completed)
        self.scan_worker.log_message.connect(self._log)
        self.scan_worker.progress_message.connect(self._set_progress_message)
        self.scan_worker.finished_scan.connect(self._scan_finished)
        self.scan_worker.start()

    def _cancel_scan(self) -> None:
        if self.scan_worker is not None:
            self.scan_worker.cancel()

    def _scan_finished(self, count: int) -> None:
        self._flush_log()
        self._flush_pending_records(max_records=None)
        self.progress.setRange(0, 1)
        self.progress.setValue(1)
        self.scan_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self._log(f"Scan finished: {count} Leica image entries.")
        self.table.setSortingEnabled(True)
        self._populate_table()
        self._save_current_cache(
            status="complete" if self.scan_worker is not None and not self.scan_worker._cancelled else "cancelled",
            force=True,
        )
        self._start_thumbnail_worker_if_idle(force=True)

    def _set_progress_message(self, message: str) -> None:
        self.progress_label.setText(message)

    def _queue_record(self, record: LeicaRecord) -> None:
        self._pending_records.append(record)
        if not self._record_flush_timer.isActive():
            self._record_flush_timer.start()

    def _flush_pending_records(self, max_records: int | None = 25) -> None:
        if not self._pending_records:
            self._record_flush_timer.stop()
            return
        limit = len(self._pending_records) if max_records is None else min(max_records, len(self._pending_records))
        self.table.setUpdatesEnabled(False)
        try:
            for _ in range(limit):
                self._add_record(self._pending_records.pop(0), save_cache=False)
        finally:
            self.table.setUpdatesEnabled(True)
        self._save_current_cache(status="in_progress")
        if not self._pending_records:
            self._record_flush_timer.stop()

    def _add_record(self, record: LeicaRecord, *, save_cache: bool = True) -> None:
        self.records.append(record)
        if self._record_visible(record):
            self._append_row(record, len(self.records) - 1)
        if save_cache:
            self._save_current_cache(status="in_progress")

    def _container_completed(self, path: str) -> None:
        self._flush_pending_records(max_records=None)
        if path:
            self.completed_containers.add(str(Path(path)))
            self._save_current_cache(status="in_progress")

    def _populate_table(self) -> None:
        if not hasattr(self, "table"):
            return
        self._populating = True
        self.table.setSortingEnabled(False)
        self.table.setRowCount(0)
        for idx, record in enumerate(self.records):
            if self._record_visible(record):
                self._append_row(record, idx)
        self.table.setSortingEnabled(True)
        self._populating = False
        self._start_thumbnail_worker_if_idle()

    def _record_visible(self, record: LeicaRecord) -> bool:
        if position_dimension_count(record) > 1:
            return False
        status = record.nyquist.status
        if status == "pass" and not self.show_pass.isChecked():
            return False
        if status in {"xy only", "near"} and not self.show_near.isChecked():
            return False
        if status == "too coarse" and not self.show_too_coarse.isChecked():
            return False
        if status == "unknown" and not self.show_unknown.isChecked():
            return False
        if self.require_3d.isChecked() and not record.is_3d:
            return False
        if self.require_confocal.isChecked() and not record.is_confocal:
            return False
        return True

    def _append_row(self, record: LeicaRecord, index: int) -> None:
        self._queue_thumbnail(index)
        row = self.table.rowCount()
        self.table.insertRow(row)
        values = [
            "",
            "",
            record.nyquist.status,
            "yes" if record.nyquist.xy_ok else "no" if record.nyquist.xy_ok is False else "",
            "yes" if record.nyquist.z_ok else "no" if record.nyquist.z_ok is False else "",
            "yes" if record.is_3d else "no",
            "yes" if record.is_confocal else "no",
            str(record.file_path),
            record.image_name,
            record.internal_path,
            record.size_x,
            record.size_y,
            record.size_z,
            record.size_c,
            record.size_t,
            record.pixel_x_um,
            record.pixel_y_um,
            record.pixel_z_um,
            record.na,
            record.refractive_index,
            fmt_list(record.excitation_nm),
            fmt_list(record.emission_nm),
            record.nyquist.pinhole_au,
            record.nyquist.limit_xy_um,
            record.nyquist.limit_z_um,
            record.nyquist.ratio_xy,
            record.nyquist.ratio_z,
            "; ".join(record.nyquist.notes + ([record.nyquist.wavelength_source] if record.nyquist.wavelength_source else [])),
        ]
        color = self._status_color(record.nyquist.status)
        for col, value in enumerate(values):
            item = QTableWidgetItem(fmt(value))
            item.setData(Qt.ItemDataRole.UserRole, index)
            if col == 0:
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(Qt.CheckState.Checked if record.checked else Qt.CheckState.Unchecked)
            elif col == 1:
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                item.setToolTip(record.thumbnail_note or "")
            else:
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            if color is not None:
                item.setBackground(color)
            self.table.setItem(row, col, item)
        self._set_thumbnail_widget(row, record)

    def _queue_thumbnail(self, index: int) -> None:
        if index < 0 or index >= len(self.records):
            return
        record = self.records[index]
        if not self.thumbnails_box.isChecked():
            record.thumbnail_note = "thumbnail creation disabled"
            return
        if record.thumbnail_path and record.thumbnail_path.exists():
            return
        eligible, note = thumbnail_candidate_note(record, self.thumbnail_max_xy_spin.value())
        record.thumbnail_note = note
        if not eligible or index in self._thumbnail_pending_set:
            return
        self._thumbnail_pending.append(index)
        self._thumbnail_pending_set.add(index)

    def _cancel_thumbnail_worker(self) -> None:
        if self.thumbnail_worker is not None and self.thumbnail_worker.isRunning():
            self.thumbnail_worker.cancel()
            self.thumbnail_worker.wait(3000)
        if self.thumbnail_worker is None or not self.thumbnail_worker.isRunning():
            self.thumbnail_worker = None

    def _scan_is_active(self) -> bool:
        return self.scan_worker is not None and self.scan_worker.isRunning()

    def _start_thumbnail_worker_if_idle(self, *, force: bool = False) -> None:
        if not self.thumbnails_box.isChecked():
            return
        if not force and self._scan_is_active():
            return
        if self.thumbnail_worker is not None and self.thumbnail_worker.isRunning():
            return
        indices = []
        while self._thumbnail_pending:
            index = self._thumbnail_pending.pop(0)
            self._thumbnail_pending_set.discard(index)
            if 0 <= index < len(self.records):
                indices.append(index)
        if not indices:
            return
        self._log(f"Creating {len(indices)} queued thumbnails in the background.")
        self.thumbnail_worker = ThumbnailWorker(
            self.records,
            indices,
            self.thumbnail_max_xy_spin.value(),
            self,
        )
        self.thumbnail_worker.thumbnail_ready.connect(self._thumbnail_ready)
        self.thumbnail_worker.log_message.connect(self._log)
        self.thumbnail_worker.finished_thumbnails.connect(self._thumbnail_finished)
        self.thumbnail_worker.start()

    def _thumbnail_ready(self, index: int, path: str, note: str) -> None:
        if index < 0 or index >= len(self.records):
            return
        record = self.records[index]
        record.thumbnail_path = Path(path) if path else None
        record.thumbnail_note = note
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            if item is None or item.data(Qt.ItemDataRole.UserRole) != index:
                continue
            thumb_item = self.table.item(row, 1)
            if thumb_item is not None:
                thumb_item.setToolTip(record.thumbnail_note or "")
            self._set_thumbnail_widget(row, record)
            break

    def _thumbnail_finished(self, count: int) -> None:
        self.thumbnail_worker = None
        if count:
            self._log(f"Thumbnail creation finished: {count} attempted.")
            self._schedule_cache_save(status="in_progress")
        self._start_thumbnail_worker_if_idle(force=True)

    def _set_thumbnail_widget(self, row: int, record: LeicaRecord) -> None:
        label = QLabel()
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setToolTip(record.thumbnail_note or "")
        label.setStyleSheet("background: transparent;")
        if record.thumbnail_path and record.thumbnail_path.exists():
            pixmap = QPixmap(str(record.thumbnail_path))
            if not pixmap.isNull():
                label.setPixmap(
                    pixmap.scaled(
                        80,
                        80,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                )
            else:
                label.setText("bad")
                label.setToolTip(f"Could not load thumbnail: {record.thumbnail_path}")
        elif record.thumbnail_note:
            label.setText("...")
        self.table.setCellWidget(row, 1, label)

    def _status_color(self, status: str) -> QColor | None:
        if status == "pass":
            return QColor(42, 92, 56)
        if status == "xy only":
            return QColor(122, 104, 32)
        if status == "near":
            return QColor(102, 89, 37)
        if status == "too coarse":
            return QColor(99, 48, 48)
        if status == "unknown":
            return QColor(65, 65, 65)
        return None

    def _item_changed(self, item: QTableWidgetItem) -> None:
        if self._populating or item.column() != 0:
            return
        idx = item.data(Qt.ItemDataRole.UserRole)
        if isinstance(idx, int) and 0 <= idx < len(self.records):
            self.records[idx].checked = item.checkState() == Qt.CheckState.Checked
            self._schedule_cache_save(status="in_progress")

    def _check_matching(self) -> None:
        for record in self.records:
            record.checked = (
                position_dimension_count(record) <= 1
                and record.nyquist.status in {"pass", "xy only", "near"}
                and record.is_3d
                and record.is_confocal
            )
        self._populate_table()
        self._log("Checked all visible green/yellow/near 3D confocal entries.")
        self._schedule_cache_save(status="in_progress")

    def _start_convert(self) -> None:
        self._sync_checks_from_table()
        selected = [record for record in self.records if record.checked and position_dimension_count(record) <= 1]
        if not selected:
            QMessageBox.information(self, "Nothing checked", "Check one or more Leica image entries first.")
            return
        output = Path(self.output_edit.text().strip())
        self.convert_button.setEnabled(False)
        self.scan_button.setEnabled(False)
        self.progress.setRange(0, len(selected))
        self.progress.setValue(0)
        self._convert_started_at = time.monotonic()
        self.progress_label.setText(f"Converting 0/{len(selected)} | Elapsed 0:00 | ETA calculating")
        self._log(f"Converting {len(selected)} checked entries to {output}")
        self.convert_worker = ConvertWorker(selected, output, self)
        self.convert_worker.log_message.connect(self._log)
        self.convert_worker.progress.connect(self._set_convert_progress)
        self.convert_worker.finished_convert.connect(self._convert_finished)
        self.convert_worker.start()

    def _set_convert_progress(self, done: int, total: int) -> None:
        self.progress.setRange(0, max(total, 1))
        self.progress.setValue(done)
        elapsed = max(0.0, time.monotonic() - self._convert_started_at) if self._convert_started_at else 0.0
        eta = None
        estimated_total = None
        if done > 0 and total > 0:
            estimated_total = elapsed / done * total
            eta = max(0.0, estimated_total - elapsed)
        if estimated_total is None or eta is None:
            self.progress_label.setText(
                f"Converting {done}/{total} | Elapsed {format_duration(elapsed)} | ETA calculating"
            )
        else:
            self.progress_label.setText(
                f"Converting {done}/{total} | Elapsed {format_duration(elapsed)} | "
                f"Est. total {format_duration(estimated_total)} | ETA {format_duration(eta)}"
            )

    def _sync_checks_from_table(self) -> None:
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            if item is None:
                continue
            idx = item.data(Qt.ItemDataRole.UserRole)
            if isinstance(idx, int) and 0 <= idx < len(self.records):
                self.records[idx].checked = item.checkState() == Qt.CheckState.Checked

    def _convert_finished(self, ok: int, failed: int) -> None:
        self.convert_button.setEnabled(True)
        self.scan_button.setEnabled(True)
        self._set_convert_progress(ok + failed, ok + failed)
        self._log(f"Conversion finished: {ok} ok, {failed} failed.")
        QMessageBox.information(self, "Conversion finished", f"{ok} converted\n{failed} failed")

    def _log(self, message: str) -> None:
        self._pending_log_lines.append(str(message))
        if not self._log_flush_timer.isActive():
            self._log_flush_timer.start()

    def _flush_log(self) -> None:
        if not self._pending_log_lines:
            self._log_flush_timer.stop()
            return
        lines = self._pending_log_lines
        self._pending_log_lines = []
        self.log.appendPlainText("\n".join(lines))
        if not self._pending_log_lines:
            self._log_flush_timer.stop()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API
        self._flush_log()
        self._cancel_thumbnail_worker()
        self._cache_save_timer.stop()
        self._save_current_cache(status="closed", force=True)
        super().closeEvent(event)


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
