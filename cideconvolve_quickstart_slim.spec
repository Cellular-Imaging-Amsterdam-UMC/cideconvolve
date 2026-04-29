# cideconvolve_quickstart_slim.spec - PyInstaller slim one-folder build
# Build with:  pyinstaller cideconvolve_quickstart_slim.spec
#              (run from the repository root)
#
# Produces:  dist/cideconvolve_slim/cideconvolve_slim.exe
#
# This is the folder-build companion to cideconvolve_slim.spec.  It uses the
# same narrowed dependency collection but keeps binaries and data in _internal/
# for faster startup and easier inspection.

import os
import pkgutil

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
    copy_metadata,
)

block_cipher = None


def _try_list(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception:
        return []


def _metadata(*dist_names):
    items = []
    for dist_name in dist_names:
        items += _try_list(copy_metadata, dist_name)
    return items


def _datas(package, **kwargs):
    return _try_list(collect_data_files, package, **kwargs)


def _binaries(package):
    return _try_list(collect_dynamic_libs, package)


def _submodules(package):
    return _try_list(collect_submodules, package)


_icon = os.path.abspath("icon.ico")
if not os.path.exists(_icon):
    raise FileNotFoundError("icon.ico is required for the Windows executable icon")


bioio_plugin_hiddenimports = (
    _submodules("bioio_ome_tiff")
    + _submodules("bioio_ome_zarr")
    + _submodules("bioio_czi")
    + _submodules("bioio_nd2")
    + _submodules("bioio.writers")
)
zarr_hiddenimports = _submodules("zarr") + _submodules("numcodecs")
imagecodecs_hiddenimports = _submodules("imagecodecs")

vispy_hiddenimports = [
    "vispy",
    "vispy.scene",
    "vispy.color",
    "vispy.visuals",
    "vispy.visuals.volume",
    "vispy.visuals.transforms",
    "vispy.app.backends._pyqt6",
]

omero_hiddenimports = [
    "omero",
    "omero.gateway",
    "omero.util",
    "omero.util.sessions",
    "omero.rtypes",
    "omero.model",
    "omero.api",
    "omero.sys",
    "omero.clients",
    "omero.cmd",
    "omero.cmd.graphs",
    "Ice",
    "IcePy",
    "Glacier2",
    "omero_browser_qt",
    "omero_browser_qt.browser_dialog",
    "omero_browser_qt.gateway",
    "omero_browser_qt.image_loader",
    "omero_browser_qt.login_dialog",
    "omero_browser_qt.rendering",
    "omero_browser_qt.scale_bar",
    "omero_browser_qt.selection_context",
    "omero_browser_qt.tree_model",
    "omero_browser_qt.widgets",
    "omero_browser_qt.view_backends",
]
omero_ice_toplevel = [
    name
    for _, name, _ in pkgutil.iter_modules()
    if name.startswith(("omero_", "Glacier2", "IcePatch2", "IceBox", "IceGrid", "IceStorm"))
]


runtime_metadata = _metadata(
    "torch",
    "numpy",
    "tifffile",
    "imagecodecs",
    "Pillow",
    "bioio",
    "bioio-base",
    "bioio-ome-tiff",
    "bioio-ome-zarr",
    "bioio-czi",
    "bioio-nd2",
    "ome-types",
    "xsdata",
    "xsdata-pydantic-basemodel",
    "pydantic",
    "pydantic-extra-types",
    "dask",
    "zarr",
    "numcodecs",
    "nd2",
    "omero-browser-qt",
    "omero-py",
    "zeroc-ice",
    "vispy",
    "PyOpenGL",
    "psutil",
    "nvidia-ml-py",
)

runtime_datas = [
    ("icon.svg", "."),
    ("icon.ico", "."),
] + runtime_metadata

runtime_datas += _datas(
    "vispy",
    includes=["**/*.glsl", "**/*.vert", "**/*.frag", "**/*.png", "**/*.json"],
    excludes=["**/__pycache__/**", "**/tests/**", "**/examples/**"],
)
runtime_datas += _datas(
    "omero_browser_qt",
    excludes=["**/__pycache__/**", "**/tests/**", "**/examples/**"],
)

runtime_binaries = (
    _binaries("torch")
    + _binaries("imagecodecs")
    + _binaries("numcodecs")
    + _binaries("zarr")
    + _binaries("nd2")
)

a = Analysis(
    ["gui_deconvolve_ci.py"],
    pathex=["."],
    binaries=runtime_binaries,
    datas=runtime_datas,
    hiddenimports=[
        "ci_dual_viewer",
        "deconvolve_ci",
        "deconvolve",
        "PyQt6.QtCore",
        "PyQt6.QtGui",
        "PyQt6.QtWidgets",
        "PyQt6.QtSvg",
        "numpy",
        "numpy.core",
        "numpy.lib",
        "torch",
        "torch.special",
        "tifffile",
        "PIL",
        "PIL.Image",
        "PIL.ImageDraw",
        "PIL.ImageFont",
        "nd2",
        "bioio",
        "bioio.writers",
        "bioio_base",
        "bioio_base.types",
        "ome_types",
        "xsdata",
        "xsdata_pydantic_basemodel",
        "xsdata_pydantic_basemodel.hooks",
        "xsdata_pydantic_basemodel.hooks.class_type",
        "pydantic",
        "pydantic_extra_types",
        "dask",
        "dask.array",
        "zarr",
        "numcodecs",
        "numcodecs.blosc",
        "psutil",
        "pynvml",
        "OpenGL",
        "OpenGL.GL",
        "OpenGL.platform.win32",
    ]
    + bioio_plugin_hiddenimports
    + zarr_hiddenimports
    + imagecodecs_hiddenimports
    + vispy_hiddenimports
    + omero_hiddenimports
    + omero_ice_toplevel,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "PyQt5",
        "PyQt5.QtCore",
        "PyQt5.QtGui",
        "PyQt5.QtWidgets",
        "PySide2",
        "PySide6",
        "tkinter",
        "_tkinter",
        "wrapper",
        "bioflows_local",
        "matplotlib",
        "IPython",
        "notebook",
        "nbconvert",
        "nbformat",
        "jupyter",
        "jupyterlab",
        "zmq",
        "jedi",
        "parso",
        "pytest",
        "setuptools.tests",
        "sphinx",
        "mkdocs",
        "mkdocstrings",
        "sklearn",
        "scipy",
        "skimage",
        "cv2",
        "napari",
        "tensorflow",
        "torchvision",
        "torchaudio",
        # PyTorch subsystems not used by CIDeconvolve's eager tensor/FFT code.
        # These mostly reduce Python-package bulk; CUDA DLLs remain necessary
        # for GPU support.
        "triton",
        "torch.distributed",
        "torch.distributions",
        "torch.testing",
        "torch.utils.benchmark",
        "torch.utils.tensorboard",
        "torch.profiler",
        "torch.onnx",
        "torch.package",
        "torch.export",
        "torch._dynamo",
        "torch._inductor",
        "torch._export",
        "dask.dataframe",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    name="cideconvolve_slim",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    icon=_icon,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="cideconvolve_slim",
)

# PyInstaller leaves a non-distributable bootloader next to folder builds.
_stale = os.path.join(DISTPATH, "cideconvolve_slim.exe")
if os.path.isfile(_stale):
    os.remove(_stale)
    print(f"Removed stale bootloader: {_stale}")
