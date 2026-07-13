#!/usr/bin/env python
"""Probe what OMERO/BIOMERO imports from OME-TIFF and OME-Zarr outputs.

This tool is intentionally host-side.  It inspects the input file/folder with
local Python packages, then runs small Python snippets inside the local
NL-BIOMERO ``biomero-importer`` container so the OMERO-side behavior uses the
same libraries, environment variables, and importer code as the deployed stack.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_NL_BIOMERO_ROOT = Path(r"E:\NL-BIOMERO")
TOOL_ROOT = Path(__file__).resolve().parent
DEFAULT_REPORT_ROOT = TOOL_ROOT / "reports"
MASKED = "******"


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    try:
        import numpy as np  # type: ignore

        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
    except Exception:
        pass
    return str(value)


def _safe_json(data: Any) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False, default=_json_default)


def _now_stamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _read_env_file(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        env[key] = value
    return env


def _mask_secret(key: str, value: Any) -> Any:
    if value is None:
        return value
    if any(token in key.upper() for token in ("PASSWORD", "SECRET", "TOKEN", "KEY")):
        return MASKED
    return value


def _masked_env(env: dict[str, str]) -> dict[str, str]:
    return {key: str(_mask_secret(key, value)) for key, value in env.items()}


def _resolve_env_value(text: str, env: dict[str, str]) -> str:
    def repl(match: re.Match[str]) -> str:
        return env.get(match.group(1), match.group(0))

    return re.sub(r"\$\{([^}]+)\}", repl, text)


def _parse_target(target: str) -> tuple[str, int]:
    text = str(target or "").strip()
    match = re.fullmatch(r"(Dataset|Screen):(\d+)", text, flags=re.IGNORECASE)
    if not match:
        raise ValueError("--target must look like Dataset:123 or Screen:123")
    target_type = "Dataset" if match.group(1).lower() == "dataset" else "Screen"
    return target_type, int(match.group(2))


def _namespaces(root: ET.Element) -> dict[str, str]:
    if root.tag.startswith("{"):
        return {"ome": root.tag.split("}", 1)[0].strip("{")}
    return {"ome": ""}


def _ome_find(root: ET.Element, path: str, ns: dict[str, str]) -> ET.Element | None:
    if ns.get("ome"):
        return root.find(path, ns)
    return root.find(path.replace("ome:", ""))


def _ome_findall(root: ET.Element, path: str, ns: dict[str, str]) -> list[ET.Element]:
    if ns.get("ome"):
        return list(root.findall(path, ns))
    return list(root.findall(path.replace("ome:", "")))


def _parse_ome_xml_summary(xml_text: str | None) -> dict[str, Any]:
    if not xml_text:
        return {}
    try:
        root = ET.fromstring(xml_text.encode("utf-8") if isinstance(xml_text, str) else xml_text)
    except Exception as exc:
        return {"parse_error": str(exc), "length": len(xml_text)}
    ns = _namespaces(root)
    image = _ome_find(root, ".//ome:Image", ns)
    pixels = _ome_find(root, ".//ome:Image/ome:Pixels", ns)
    channels = _ome_findall(root, ".//ome:Image/ome:Pixels/ome:Channel", ns)
    annotations = _ome_findall(root, ".//ome:StructuredAnnotations/*", ns)
    summary: dict[str, Any] = {
        "image_name": image.get("Name") if image is not None else None,
        "pixels": dict(pixels.attrib) if pixels is not None else {},
        "channels": [dict(ch.attrib) for ch in channels],
        "structured_annotation_count": len(annotations),
    }
    return summary


def _try_json(value: Any) -> Any:
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8", errors="replace")
    if not isinstance(value, str):
        return value
    text = value.strip().rstrip("\x00")
    if not text:
        return text
    try:
        return json.loads(text)
    except Exception:
        return text


def inspect_ome_tiff(path: Path) -> dict[str, Any]:
    try:
        import tifffile  # type: ignore
    except Exception as exc:
        return {"kind": "ome-tiff", "path": str(path), "error": f"tifffile is not installed: {exc}"}

    info: dict[str, Any] = {"kind": "ome-tiff", "path": str(path)}
    try:
        with tifffile.TiffFile(str(path)) as tif:
            info["is_ome"] = bool(tif.is_ome)
            info["is_bigtiff"] = bool(tif.is_bigtiff)
            info["ome_xml_length"] = len(tif.ome_metadata or "")
            info["ome_xml_summary"] = _parse_ome_xml_summary(tif.ome_metadata)
            series = []
            for index, ser in enumerate(tif.series):
                levels = []
                for level_index, level in enumerate(getattr(ser, "levels", []) or []):
                    levels.append(
                        {
                            "level": level_index,
                            "shape": tuple(int(v) for v in getattr(level, "shape", ()) or ()),
                            "dtype": str(getattr(level, "dtype", "")),
                            "axes": getattr(level, "axes", None),
                        }
                    )
                series.append(
                    {
                        "index": index,
                        "shape": tuple(int(v) for v in getattr(ser, "shape", ()) or ()),
                        "dtype": str(getattr(ser, "dtype", "")),
                        "axes": getattr(ser, "axes", None),
                        "levels": levels,
                    }
                )
            info["series"] = series
            first_page = tif.pages[0] if tif.pages else None
            if first_page is not None:
                tags = first_page.tags
                info["first_page"] = {
                    "compression": str(first_page.compression.name),
                    "is_tiled": bool(first_page.is_tiled),
                    "tile_width": int(getattr(first_page, "tilewidth", 0) or 0),
                    "tile_length": int(getattr(first_page, "tilelength", 0) or 0),
                    "subifds": int(tags["SubIFDs"].count) if "SubIFDs" in tags else 0,
                    "predictor": str(tags["Predictor"].value) if "Predictor" in tags else None,
                    "photometric": str(first_page.photometric.name),
                }
                if 65000 in tags:
                    info["cideconvolve_private_tag_65000"] = _try_json(tags[65000].value)
    except Exception as exc:
        info["error"] = str(exc)
    return info


def _zarr_array_summary(array: Any) -> dict[str, Any]:
    try:
        compressor = getattr(array, "compressor", None)
    except Exception:
        compressor = getattr(array, "compressors", None)
    try:
        filters = getattr(array, "filters", None)
    except Exception:
        filters = None
    return {
        "shape": tuple(int(v) for v in getattr(array, "shape", ()) or ()),
        "chunks": tuple(int(v) for v in getattr(array, "chunks", ()) or ()),
        "dtype": str(getattr(array, "dtype", "")),
        "compressor": str(compressor) if compressor is not None else None,
        "filters": [str(f) for f in filters] if filters else [],
    }


def inspect_ome_zarr(path: Path) -> dict[str, Any]:
    try:
        import zarr  # type: ignore
    except Exception as exc:
        return {"kind": "ome-zarr", "path": str(path), "error": f"zarr is not installed: {exc}"}

    info: dict[str, Any] = {"kind": "ome-zarr", "path": str(path)}
    try:
        root = zarr.open(str(path), mode="r")
        info["attrs"] = dict(root.attrs.asdict())
        arrays: dict[str, Any] = {}

        def visit(name: str, obj: Any) -> None:
            if hasattr(obj, "shape") and hasattr(obj, "dtype"):
                arrays[name or "/"] = _zarr_array_summary(obj)

        try:
            root.visititems(visit)
        except Exception:
            for key in root.keys():
                obj = root[key]
                if hasattr(obj, "shape"):
                    arrays[str(key)] = _zarr_array_summary(obj)
        info["arrays"] = arrays
        ome_xml_path = path / "OME" / "METADATA.ome.xml"
        if ome_xml_path.exists():
            ome_xml = ome_xml_path.read_text(encoding="utf-8", errors="replace")
            info["ome_xml_length"] = len(ome_xml)
            info["ome_xml_summary"] = _parse_ome_xml_summary(ome_xml)
        info["is_hcs_plate"] = "plate" in info["attrs"]
        if info["is_hcs_plate"]:
            try:
                plate = info["attrs"].get("plate") or {}
                first_well = (plate.get("wells") or [])[0]
                well_path = first_well["path"]
                well_group = zarr.open(str(path), mode="r", path=well_path)
                well_attrs = dict(well_group.attrs.asdict())
                first_image = (well_attrs.get("well", {}).get("images") or [])[0]
                field_path = f"{well_path}/{first_image['path']}"
                field_group = zarr.open(str(path), mode="r", path=field_path)
                field_attrs = dict(field_group.attrs.asdict())
                info["first_hcs_field"] = {
                    "path": field_path,
                    "attrs": field_attrs,
                    "arrays": {
                        key: _zarr_array_summary(field_group[key])
                        for key in field_group.array_keys()
                    },
                }
            except Exception as exc:
                info["first_hcs_field"] = {"error": str(exc)}
    except Exception as exc:
        info["error"] = str(exc)
    return info


def inspect_input(path: Path) -> dict[str, Any]:
    if path.is_dir():
        return inspect_ome_zarr(path)
    suffixes = "".join(path.suffixes).lower()
    if ".zarr" in suffixes:
        return inspect_ome_zarr(path)
    return inspect_ome_tiff(path)


@dataclass
class StackConfig:
    root: Path
    compose_file: Path
    env_file: Path
    env: dict[str, str]
    importer_container: str | None = None

    @classmethod
    def from_root(cls, root: Path) -> "StackConfig":
        compose_file = root / "docker-compose.yml"
        env_file = root / ".env"
        if not compose_file.exists():
            raise FileNotFoundError(f"docker-compose.yml not found under {root}")
        env = _read_env_file(env_file)
        return cls(root=root, compose_file=compose_file, env_file=env_file, env=env)

    @classmethod
    def from_container(cls, container: str) -> "StackConfig":
        env_proc = subprocess.run(
            ["docker", "inspect", container, "--format", "{{json .Config.Env}}"],
            text=True,
            capture_output=True,
            timeout=30,
        )
        if env_proc.returncode != 0:
            raise RuntimeError(f"Could not inspect container {container}: {env_proc.stderr}")
        env_list = json.loads(env_proc.stdout.strip() or "[]")
        env: dict[str, str] = {}
        for item in env_list:
            if isinstance(item, str) and "=" in item:
                key, value = item.split("=", 1)
                env[key] = value
        return cls(
            root=Path.cwd(),
            compose_file=Path(""),
            env_file=Path(""),
            env=env,
            importer_container=container,
        )

    def compose(self, *args: str, input_text: str | None = None, timeout: int | None = None) -> subprocess.CompletedProcess[str]:
        if self.importer_container:
            if not args:
                raise ValueError("No docker command arguments supplied")
            if args[0] == "exec":
                exec_args = list(args[1:])
                if exec_args and exec_args[0] == "-T":
                    exec_args.pop(0)
                if exec_args and exec_args[0] == "biomero-importer":
                    exec_args.pop(0)
                cmd = ["docker", "exec", "-i", self.importer_container, *exec_args]
            elif args[0] == "cp":
                cp_args = []
                for value in args[1:]:
                    cp_args.append(value.replace("biomero-importer:", f"{self.importer_container}:"))
                cmd = ["docker", "cp", *cp_args]
            elif args[0] == "ps":
                cmd = ["docker", "ps", "--format", "json"]
            else:
                raise ValueError(f"Direct container mode does not support docker compose command: {args[0]}")
            return subprocess.run(
                cmd,
                cwd=str(self.root),
                input=input_text,
                text=True,
                capture_output=True,
                timeout=timeout,
            )
        cmd = [
            "docker",
            "compose",
            "--profile",
            "IMPORTER_ENABLED",
            "--env-file",
            str(self.env_file),
            "-f",
            str(self.compose_file),
            *args,
        ]
        return subprocess.run(
            cmd,
            cwd=str(self.root),
            input=input_text,
            text=True,
            capture_output=True,
            timeout=timeout,
        )


def _container_helper_code() -> str:
    return r'''
import json
import logging
import os
import sys
import uuid
import xml.sax.saxutils as saxutils

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

from omero.gateway import BlitzGateway
from omero.rtypes import rstring
from omero.model import DatasetI, ImageI, ScreenI, PlateI
from omero.sys import Parameters

from biomero_importer.utils.initialize import load_settings
from biomero_importer.utils.importer import DataPackageImporter
from biomero_importer.utils.ingest_tracker import (
    initialize_ingest_tracker,
    get_ingest_tracker,
    log_ingestion_step,
    IngestionTracking,
    STAGE_NEW_ORDER,
    STAGE_INGEST_STARTED,
    STAGE_IMPORTED,
    STAGE_INGEST_FAILED,
)


def rv(value):
    if isinstance(value, dict):
        return {str(k): rv(v) for k, v in value.items()}
    return getattr(value, "val", value)


def rvalue(value):
    if value is None:
        return None
    if isinstance(value, dict):
        return {str(k): rvalue(v) for k, v in value.items()}
    for attr in ("val", "_val"):
        if hasattr(value, attr):
            try:
                return rvalue(getattr(value, attr))
            except Exception:
                pass
    try:
        got = value.getValue()
        if got is not value:
            return rvalue(got)
    except Exception:
        pass
    return value


def len_to_dict(value):
    if value is None:
        return None
    try:
        raw = value.getValue()
    except Exception:
        raw = value
    raw = rvalue(raw)
    if isinstance(raw, dict):
        numeric = raw.get("value")
        unit = raw.get("unit") or raw.get("symbol")
        return {"value": numeric, "unit": unit}
    numeric = raw
    unit = None
    try:
        got_unit = value.getUnit()
        unit = rvalue(got_unit) if got_unit else None
    except Exception:
        try:
            got_unit = value.unit
            unit = rvalue(got_unit) if got_unit else None
        except Exception:
            unit = None
    return {"value": numeric, "unit": unit}


def enum_value(value):
    try:
        return value.getValue().getValue()
    except Exception:
        try:
            return value.getValue()._val
        except Exception:
            return str(value)


def safe_projection(query, hql, params=None, opts=None):
    try:
        return query.projection(hql, params or Parameters(), opts)
    except Exception as exc:
        return {"error": str(exc), "query": hql}


def _rendering_value(value):
    raw = rvalue(value)
    try:
        return raw.getValue()
    except Exception:
        return raw


def summarize_rendering(conn, pixels):
    rendering = {}
    rend = None
    try:
        rend = conn.c.sf.createRenderingEngine()
        rend.lookupPixels(int(pixels.getId()))
        try:
            has_def = bool(rend.lookupRenderingDef(int(pixels.getId())))
        except Exception:
            has_def = False
        if has_def:
            rend.load()
        else:
            try:
                rend.resetDefaultSettings(True)
            except TypeError:
                rend.resetDefaultSettings()
            rend.load()
        rendering = {
            "default_z": _rendering_value(rend.getDefaultZ()),
            "default_t": _rendering_value(rend.getDefaultT()),
            "model": str(_rendering_value(rend.getModel())),
            "channels": [],
        }
        for index in range(pixels.getSizeC()):
            try:
                rendering["channels"].append({
                    "index": index,
                    "active": True,
                    "input_start": _rendering_value(rend.getChannelWindowStart(index)),
                    "input_end": _rendering_value(rend.getChannelWindowEnd(index)),
                })
            except Exception as exc:
                rendering["channels"].append({"index": index, "error": str(exc)})
    except Exception as exc:
        rendering = {"error": str(exc)}
    finally:
        if rend is not None:
            try:
                rend.close()
            except Exception:
                pass
    return rendering


def map_annotation_values(ann):
    try:
        return {str(v.name): str(v.value) for v in ann.getValue()}
    except Exception:
        return {}


def summarize_annotations(obj):
    out = []
    try:
        for ann in obj.listAnnotations():
            entry = {
                "id": ann.getId(),
                "type": ann.OMERO_CLASS,
                "namespace": ann.getNs(),
            }
            if ann.OMERO_CLASS == "MapAnnotation":
                entry["values"] = map_annotation_values(ann)
            elif ann.OMERO_CLASS == "CommentAnnotation":
                entry["text"] = ann.getTextValue()
            elif ann.OMERO_CLASS == "TagAnnotation":
                entry["text"] = ann.getTextValue()
            out.append(entry)
    except Exception as exc:
        out.append({"error": str(exc)})
    return out


def summarize_image(conn, image_id):
    image = conn.getObject("Image", int(image_id))
    if image is None:
        return {"id": int(image_id), "error": "Image not found"}
    pixels = image.getPrimaryPixels()
    px_obj = pixels._obj
    query = conn.getQueryService()
    from omero.rtypes import rlong
    channel_params = Parameters()
    channel_params.map = {"pixels_id": rlong(int(pixels.getId()))}
    channel_rows = safe_projection(
        query,
        "SELECT ch.id, lc.name, lc.emissionWave, lc.excitationWave, lc.pinHoleSize "
        "FROM Channel ch JOIN ch.logicalChannel lc JOIN ch.pixels p "
        "WHERE p.id = :pixels_id ORDER BY ch.id",
        channel_params,
        conn.SERVICE_OPTS,
    )
    channels = []
    if isinstance(channel_rows, dict) and channel_rows.get("error"):
        channels.append(channel_rows)
    else:
        for index, row in enumerate(channel_rows):
            channels.append({
                "index": index,
                "id": rv(row[0]),
                "name": rv(row[1]),
                "emission_wavelength": len_to_dict(row[2]),
                "excitation_wavelength": len_to_dict(row[3]),
                "pinhole_size": len_to_dict(row[4]),
            })
    rendering = summarize_rendering(conn, pixels)

    params = Parameters()
    params.map = {"image_id": rlong(int(image_id))}
    db_rows = {
        "dataset_links": safe_projection(
            query,
            "SELECT d.id, d.name FROM DatasetImageLink l JOIN l.parent d WHERE l.child.id = :image_id",
            params,
            conn.SERVICE_OPTS,
        ),
        "filesets": safe_projection(
            query,
            "SELECT fs.id, fs.templatePrefix FROM Image i JOIN i.fileset fs WHERE i.id = :image_id",
            params,
            conn.SERVICE_OPTS,
        ),
        "original_files": safe_projection(
            query,
            "SELECT ofile.id, ofile.name, ofile.path, ofile.mimetype "
            "FROM Image i JOIN i.fileset fs JOIN fs.usedFiles fe JOIN fe.originalFile ofile "
            "WHERE i.id = :image_id",
            params,
            conn.SERVICE_OPTS,
        ),
    }
    external_rows = safe_projection(
        query,
        "SELECT e.entityId, e.entityType, e.lsid "
        "FROM Image i JOIN i.details.externalInfo e WHERE i.id = :image_id",
        params,
        conn.SERVICE_OPTS,
    )
    if isinstance(external_rows, dict) and external_rows.get("error"):
        external = external_rows
    elif external_rows:
        row = external_rows[0]
        external = {
            "entity_id": rv(row[0]),
            "entity_type": rv(row[1]),
            "lsid": rv(row[2]),
        }
    else:
        external = None
    return {
        "id": image.getId(),
        "name": image.getName(),
        "description": image.getDescription(),
        "pixels": {
            "id": pixels.getId(),
            "size_x": pixels.getSizeX(),
            "size_y": pixels.getSizeY(),
            "size_z": pixels.getSizeZ(),
            "size_c": pixels.getSizeC(),
            "size_t": pixels.getSizeT(),
            "type": enum_value(px_obj.getPixelsType()),
            "physical_size_x": len_to_dict(px_obj.getPhysicalSizeX()),
            "physical_size_y": len_to_dict(px_obj.getPhysicalSizeY()),
            "physical_size_z": len_to_dict(px_obj.getPhysicalSizeZ()),
        },
        "channels": channels,
        "rendering": rendering,
        "external_info": external,
        "annotations": summarize_annotations(image),
        "db_rows": db_rows,
    }


def summarize_plate(conn, plate_id):
    plate = conn.getObject("Plate", int(plate_id))
    if plate is None:
        return {"id": int(plate_id), "error": "Plate not found"}
    image_ids = []
    try:
        for well in plate.listChildren():
            for sample in well.listChildren():
                img = sample.getImage()
                if img:
                    image_ids.append(img.getId())
    except Exception:
        pass
    return {
        "id": plate.getId(),
        "name": plate.getName(),
        "annotations": summarize_annotations(plate),
        "image_ids": image_ids,
        "images": [summarize_image(conn, iid) for iid in image_ids[:25]],
        "image_count_reported": len(image_ids),
    }


def connect_as_user(payload):
    settings = load_settings("/auto-importer/config/settings.yml")
    root_user = os.environ.get("OMERO_USER")
    root_password = os.environ.get("OMERO_PASSWORD")
    host = os.environ.get("OMERO_HOST")
    port = int(os.environ.get("OMERO_PORT", "4064"))
    username = payload.get("user")
    group = payload.get("group")
    import ezomero
    root = BlitzGateway(root_user, root_password, host=host, port=port, secure=True)
    if not root.connect():
        raise RuntimeError("Could not connect to OMERO as importer/root user")
    group_id = ezomero.get_group_id(root, group)
    user_conn = root.suConn(username, ttl=600000)
    if not user_conn:
        raise RuntimeError("Could not suConn to requested user")
    user_conn.setGroupForSession(group_id)
    return settings, root, user_conn


def summarize_objects(payload, image_ids=None, plate_ids=None):
    settings, root, conn = connect_as_user(payload)
    try:
        return {
            "images": [summarize_image(conn, iid) for iid in (image_ids or [])],
            "plates": [summarize_plate(conn, pid) for pid in (plate_ids or [])],
        }
    finally:
        try:
            conn.close()
        finally:
            root.close()


def ome_xml_attr(name, value):
    if value is None:
        return ""
    return f' {name}="{xml_escape_attr(value)}"'


def xml_escape_attr(value):
    return saxutils.escape(str(value), {'"': "&quot;"})


def ome_color_int(rgb):
    if not rgb:
        return None
    r, g, b = rgb[:3]
    value = (int(r) << 24) | (int(g) << 16) | (int(b) << 8) | 255
    if value >= (1 << 31):
        value -= (1 << 32)
    return value


def source_image_to_ome_xml(image_summary, image_name, dtype):
    pixels = image_summary["pixels"]
    channels = image_summary.get("channels", [])
    rendering_channels = image_summary.get("rendering", {}).get("channels", [])
    channel_xml = []
    for idx, channel in enumerate(channels):
        render = rendering_channels[idx] if idx < len(rendering_channels) else {}
        attrs = [
            ome_xml_attr("ID", f"Channel:{idx}"),
            ome_xml_attr("Name", channel.get("name") or f"Channel {idx + 1}"),
        ]
        color = ome_color_int((render.get("red"), render.get("green"), render.get("blue"))) if render else None
        attrs.append(ome_xml_attr("Color", color))
        emission = channel.get("emission_wavelength")
        excitation = channel.get("excitation_wavelength")
        pinhole = channel.get("pinhole_size")
        if isinstance(emission, dict) and emission.get("value") is not None:
            attrs.append(ome_xml_attr("EmissionWavelength", emission.get("value")))
            attrs.append(ome_xml_attr("EmissionWavelengthUnit", emission.get("unit") or "nm"))
        if isinstance(excitation, dict) and excitation.get("value") is not None:
            attrs.append(ome_xml_attr("ExcitationWavelength", excitation.get("value")))
            attrs.append(ome_xml_attr("ExcitationWavelengthUnit", excitation.get("unit") or "nm"))
        if isinstance(pinhole, dict) and pinhole.get("value") is not None:
            attrs.append(ome_xml_attr("PinholeSize", pinhole.get("value")))
            attrs.append(ome_xml_attr("PinholeSizeUnit", pinhole.get("unit") or "µm"))
        channel_xml.append("<Channel" + "".join(attrs) + "/>")

    def size_value(key, default=None):
        obj = pixels.get(key)
        if isinstance(obj, dict):
            return obj.get("value", default)
        return obj if obj is not None else default

    processing_note = saxutils.escape(json.dumps({
        "source": "OMERO image exported by CIDeconvolve OMERO metadata probe",
        "source_image_id": image_summary.get("id"),
    }))
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">'
        f'<Image ID="Image:0" Name="{xml_escape_attr(image_name)}">'
        f"<Description>{processing_note}</Description>"
        f'<Pixels DimensionOrder="XYZCT" ID="Pixels:0"'
        + ome_xml_attr("PhysicalSizeX", size_value("physical_size_x"))
        + ome_xml_attr("PhysicalSizeXUnit", (pixels.get("physical_size_x") or {}).get("unit") if isinstance(pixels.get("physical_size_x"), dict) else "µm")
        + ome_xml_attr("PhysicalSizeY", size_value("physical_size_y"))
        + ome_xml_attr("PhysicalSizeYUnit", (pixels.get("physical_size_y") or {}).get("unit") if isinstance(pixels.get("physical_size_y"), dict) else "µm")
        + ome_xml_attr("PhysicalSizeZ", size_value("physical_size_z"))
        + ome_xml_attr("PhysicalSizeZUnit", (pixels.get("physical_size_z") or {}).get("unit") if isinstance(pixels.get("physical_size_z"), dict) else "µm")
        + ome_xml_attr("SizeC", pixels.get("size_c"))
        + ome_xml_attr("SizeT", pixels.get("size_t"))
        + ome_xml_attr("SizeX", pixels.get("size_x"))
        + ome_xml_attr("SizeY", pixels.get("size_y"))
        + ome_xml_attr("SizeZ", pixels.get("size_z"))
        + ome_xml_attr("Type", dtype)
        + ">"
        + "".join(channel_xml)
        + "</Pixels></Image></OME>"
    )


def export_image_to_zarr(payload):
    import numpy as np
    import zarr

    settings, root, conn = connect_as_user(payload)
    output_dir = payload["output_dir"]
    image_id = int(payload["image_id"])
    image = conn.getObject("Image", image_id)
    if image is None:
        raise RuntimeError(f"Image {image_id} not found")
    source_summary = summarize_image(conn, image_id)
    pixels = image.getPrimaryPixels()
    px = source_summary["pixels"]
    dtype_text = str(px.get("type") or "uint16").lower()
    dtype_map = {
        "uint8": "uint8",
        "int8": "int8",
        "uint16": "uint16",
        "int16": "int16",
        "uint32": "uint32",
        "int32": "int32",
        "float": "float32",
        "float32": "float32",
        "double": "float64",
        "float64": "float64",
    }
    dtype = np.dtype(dtype_map.get(dtype_text, "float32"))
    shape = (
        int(px["size_t"]),
        int(px["size_c"]),
        int(px["size_z"]),
        int(px["size_y"]),
        int(px["size_x"]),
    )
    chunks = (
        1,
        1,
        max(1, min(shape[2], 16)),
        max(1, min(shape[3], 512)),
        max(1, min(shape[4], 512)),
    )
    if os.path.exists(output_dir):
        import shutil
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    root_group = zarr.open_group(output_dir, mode="w")
    arr = root_group.create_dataset("0", shape=shape, chunks=chunks, dtype=dtype, overwrite=True)

    raw_store = None
    try:
        raw_store = conn.c.sf.createRawPixelsStore()
        raw_store.setPixelsId(int(pixels.getId()), False)
        for t in range(shape[0]):
            for c in range(shape[1]):
                for z in range(shape[2]):
                    plane = raw_store.getPlane(z, c, t)
                    if isinstance(plane, (bytes, bytearray, memoryview)):
                        if np.issubdtype(dtype, np.integer) and dtype.itemsize > 1:
                            read_dtype = dtype.newbyteorder(">")
                        else:
                            read_dtype = dtype
                        plane_arr = np.frombuffer(plane, dtype=read_dtype).astype(dtype, copy=False)
                        plane_arr = plane_arr.reshape((shape[3], shape[4]))
                    else:
                        plane_arr = np.asarray(plane, dtype=dtype).reshape((shape[3], shape[4]))
                    arr[t, c, z, :, :] = plane_arr
    finally:
        if raw_store is not None:
            try:
                raw_store.close()
            except Exception:
                pass

    def physical(axis_key, default):
        obj = px.get(axis_key)
        if isinstance(obj, dict) and obj.get("value") is not None:
            return float(obj["value"])
        return default

    px_x = physical("physical_size_x", 1.0)
    px_y = physical("physical_size_y", px_x)
    px_z = physical("physical_size_z", 1.0)
    root_group.attrs["multiscales"] = [{
        "version": "0.4",
        "axes": [
            {"name": "t", "type": "time"},
            {"name": "c", "type": "channel"},
            {"name": "z", "type": "space", "unit": "micrometer"},
            {"name": "y", "type": "space", "unit": "micrometer"},
            {"name": "x", "type": "space", "unit": "micrometer"},
        ],
        "datasets": [{
            "path": "0",
            "coordinateTransformations": [{"type": "scale", "scale": [1, 1, px_z, px_y, px_x]}],
        }],
        "name": image.getName(),
    }]
    rendering = source_summary.get("rendering", {})
    rendering_channels = rendering.get("channels", [])
    zarr_channels = []
    for idx, ch in enumerate(source_summary.get("channels", [])):
        render = rendering_channels[idx] if idx < len(rendering_channels) else {}
        color = "FFFFFF"
        if render:
            color = f"{int(render.get('red') or 0):02X}{int(render.get('green') or 0):02X}{int(render.get('blue') or 0):02X}"
        entry = {
            "label": ch.get("name") or f"Channel {idx + 1}",
            "color": color,
            "active": bool(render.get("active", True)) if render else True,
            "coefficient": float(render.get("coefficient", 1.0)) if render else 1.0,
            "family": "linear",
            "inverted": False,
            "window": {
                "start": float(render.get("input_start", 0.0)) if render else 0.0,
                "end": float(render.get("input_end", 1.0)) if render else 1.0,
                "min": 0.0,
                "max": float(np.iinfo(dtype).max) if np.issubdtype(dtype, np.integer) else 1.0,
            },
        }
        zarr_channels.append(entry)
    root_group.attrs["omero"] = {
        "name": image.getName(),
        "id": image_id,
        "channels": zarr_channels,
        "rdefs": {
            "defaultT": rendering.get("default_t", 0) if isinstance(rendering, dict) else 0,
            "defaultZ": rendering.get("default_z", 0) if isinstance(rendering, dict) else 0,
            "model": "color",
        },
        "description": "Exported from OMERO by CIDeconvolve metadata probe for Slurm-input metadata checking.",
    }
    root_group.attrs["probe_source_omero"] = {
        "image_id": image_id,
        "image_name": image.getName(),
        "pixels_id": pixels.getId(),
        "exporter": "cideconvolve.tools.omero_import_metadata_probe",
    }
    ome_dir = os.path.join(output_dir, "OME")
    os.makedirs(ome_dir, exist_ok=True)
    with open(os.path.join(ome_dir, ".zgroup"), "w", encoding="utf-8") as fh:
        json.dump({"zarr_format": 2}, fh)
    with open(os.path.join(ome_dir, "METADATA.ome.xml"), "w", encoding="utf-8") as fh:
        fh.write(source_image_to_ome_xml(source_summary, image.getName(), str(dtype)))
    return {
        "image_id": image_id,
        "container_zarr_path": output_dir,
        "source_omero": source_summary,
        "shape": shape,
        "chunks": chunks,
        "dtype": str(dtype),
    }


def run_direct(payload):
    target_type = payload["target_type"]
    target_id = int(payload["target_id"])
    file_path = payload["container_input"]
    package = {
        "Group": payload["group"],
        "Username": payload["user"],
        "UUID": payload["uuid"],
        "DestinationID": target_id,
        "DestinationType": target_type,
        "Files": [file_path],
        "FileNames": [os.path.basename(file_path.rstrip("/"))],
    }
    settings = load_settings("/auto-importer/config/settings.yml")
    importer = DataPackageImporter(settings, package)
    image_ids = []
    plate_ids = []
    is_zarr = "zar" in os.path.splitext(file_path)[1].lower()
    if is_zarr and importer.use_register_zarr:
        ids, zarr_is_plate = importer.import_zarr(uri=file_path, target=target_id)
        if zarr_is_plate:
            plate_ids = ids
        else:
            image_ids = ids
    elif target_type == "Dataset":
        if os.path.isfile(file_path):
            image_ids = importer.import_dataset(target=file_path, dataset=target_id, transfer=payload.get("transfer", "upload"))
        else:
            ok = importer.import_to_omero(file_path=file_path, target_id=target_id, target_type="Dataset", uuid=payload["uuid"], depth=10, transfer_type=payload.get("transfer", "upload"))
            image_ids = [target_id] if ok else []
    else:
        ok = importer.import_to_omero(file_path=file_path, target_id=target_id, target_type="Screen", uuid=payload["uuid"], depth=10, transfer_type=payload.get("transfer", "upload"))
        if ok:
            got = importer.get_plate_ids(file_path, target_id)
            plate_ids = got[0] if got else []
    summary = summarize_objects(payload, image_ids=image_ids, plate_ids=plate_ids)
    return {
        "uuid": payload["uuid"],
        "image_ids": image_ids,
        "plate_ids": plate_ids,
        "settings": effective_settings(settings, importer),
        "objects": summary,
    }


def effective_settings(settings, importer=None):
    keys = [
        "parallel_upload_per_worker",
        "parallel_filesets_per_worker",
        "skip_all",
        "skip_checksum",
        "skip_minmax",
        "skip_thumbnails",
        "skip_upgrade",
        "annotation_namespace",
        "use_register_zarr",
    ]
    out = {key: settings.get(key) for key in keys if key in settings}
    out["USE_REGISTER_ZARR_env"] = os.environ.get("USE_REGISTER_ZARR")
    if importer is not None:
        out["resolved_use_register_zarr"] = str(importer.use_register_zarr)
    return out


def fetch_ingest_rows(uuid_value):
    tracker = get_ingest_tracker()
    if tracker is None:
        return []
    with tracker.Session() as session:
        rows = session.query(IngestionTracking).filter(IngestionTracking.uuid == uuid_value).order_by(IngestionTracking.id.asc()).all()
        out = []
        for row in rows:
            out.append({
                "id": row.id,
                "group_name": row.group_name,
                "user_name": row.user_name,
                "destination_id": row.destination_id,
                "destination_type": row.destination_type,
                "stage": row.stage,
                "uuid": row.uuid,
                "timestamp": row.timestamp.isoformat() if row.timestamp else None,
                "files": row.files,
                "file_names": row.file_names,
                "description": row.description,
                "preprocessing_id": row.preprocessing_id,
            })
        return out


def run_biomero(payload):
    target_type = payload["target_type"]
    target_id = int(payload["target_id"])
    file_path = payload["container_input"]
    settings = load_settings("/auto-importer/config/settings.yml")
    initialize_ingest_tracker(settings)
    package = {
        "Group": payload["group"],
        "Username": payload["user"],
        "UUID": payload["uuid"],
        "DestinationID": target_id,
        "DestinationType": target_type,
        "Files": [file_path],
        "FileNames": [os.path.basename(file_path.rstrip("/"))],
    }
    log_ingestion_step(package, STAGE_NEW_ORDER)
    log_ingestion_step(package, STAGE_INGEST_STARTED)
    importer = DataPackageImporter(settings, package)
    successful, failed, import_failed = importer.import_data_package()
    if import_failed or failed or (not successful and not failed):
        package["Description"] = "Probe import failed"
        log_ingestion_step(package, STAGE_INGEST_FAILED)
    else:
        log_ingestion_step(package, STAGE_IMPORTED)
    image_ids = []
    plate_ids = []
    for entry in successful:
        oid = entry[3]
        if isinstance(oid, int):
            if target_type == "Screen":
                plate_ids.append(oid)
            else:
                image_ids.append(oid)
    summary = summarize_objects(payload, image_ids=image_ids, plate_ids=plate_ids)
    return {
        "uuid": payload["uuid"],
        "successful_uploads": successful,
        "failed_uploads": failed,
        "import_failed": import_failed,
        "image_ids": image_ids,
        "plate_ids": plate_ids,
        "settings": effective_settings(settings, importer),
        "ingest_rows": fetch_ingest_rows(payload["uuid"]),
        "objects": summary,
    }


def inspect_existing(payload):
    image_ids = [int(x) for x in payload.get("image_ids", [])]
    plate_ids = [int(x) for x in payload.get("plate_ids", [])]
    return summarize_objects(payload, image_ids=image_ids, plate_ids=plate_ids)


def cleanup(payload):
    settings, root, conn = connect_as_user(payload)
    deleted = []
    try:
        for image_id in payload.get("image_ids", []):
            try:
                conn.deleteObjects("Image", [int(image_id)], deleteAnns=True, deleteChildren=True, wait=True)
                deleted.append({"type": "Image", "id": int(image_id)})
            except Exception as exc:
                deleted.append({"type": "Image", "id": int(image_id), "error": str(exc)})
        for plate_id in payload.get("plate_ids", []):
            try:
                conn.deleteObjects("Plate", [int(plate_id)], deleteAnns=True, deleteChildren=True, wait=True)
                deleted.append({"type": "Plate", "id": int(plate_id)})
            except Exception as exc:
                deleted.append({"type": "Plate", "id": int(plate_id), "error": str(exc)})
        return {"deleted": deleted}
    finally:
        try:
            conn.close()
        finally:
            root.close()


payload = json.load(sys.stdin)
action = payload["action"]
try:
    if action == "direct":
        result = run_direct(payload)
    elif action == "biomero":
        result = run_biomero(payload)
    elif action == "existing":
        result = inspect_existing(payload)
    elif action == "export_zarr":
        result = export_image_to_zarr(payload)
    elif action == "cleanup":
        result = cleanup(payload)
    else:
        raise ValueError("Unknown action: " + action)
    print(json.dumps({"ok": True, "result": result}, default=str))
except Exception as exc:
    import traceback
    print(json.dumps({"ok": False, "error": str(exc), "traceback": traceback.format_exc()}, default=str))
    sys.exit(2)
'''


def _run_container_helper(stack: StackConfig, payload: dict[str, Any], timeout: int = 3600) -> dict[str, Any]:
    proc = stack.compose(
        "exec",
        "-T",
        "biomero-importer",
        "python",
        "-c",
        _container_helper_code(),
        input_text=json.dumps(payload, default=_json_default),
        timeout=timeout,
    )
    raw = (proc.stdout or "").strip()
    if proc.returncode != 0:
        raise RuntimeError(
            f"Container helper failed with exit {proc.returncode}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    last_line = raw.splitlines()[-1] if raw else ""
    try:
        data = json.loads(last_line)
    except Exception as exc:
        raise RuntimeError(f"Could not parse container helper JSON: {exc}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}") from exc
    if not data.get("ok"):
        raise RuntimeError(f"Container helper reported failure: {data.get('error')}\n{data.get('traceback', '')}")
    return data["result"]


def _compose_ps(stack: StackConfig) -> dict[str, Any]:
    proc = stack.compose("ps", "--format", "json", timeout=30)
    result: dict[str, Any] = {
        "returncode": proc.returncode,
        "stderr": proc.stderr,
        "services": [],
    }
    if proc.stdout.strip():
        services = []
        for line in proc.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                services.append(json.loads(line))
            except Exception:
                pass
        result["services"] = services
    return result


def _volume_mapping_from_env(stack: StackConfig) -> list[tuple[Path, str]]:
    if stack.importer_container:
        proc = subprocess.run(
            ["docker", "inspect", stack.importer_container, "--format", "{{json .Mounts}}"],
            text=True,
            capture_output=True,
            timeout=30,
        )
        mappings: list[tuple[Path, str]] = []
        if proc.returncode == 0 and proc.stdout.strip():
            for mount in json.loads(proc.stdout):
                if mount.get("Type") != "bind":
                    continue
                source = str(mount.get("Source") or "")
                destination = str(mount.get("Destination") or "")
                if source.startswith("/run/desktop/mnt/host/") and len(source) > len("/run/desktop/mnt/host/x/"):
                    drive = source[len("/run/desktop/mnt/host/"):]
                    drive_letter, rest = drive.split("/", 1)
                    rest_windows = rest.replace("/", "\\")
                    source = f"{drive_letter.upper()}:\\{rest_windows}"
                if source and destination:
                    mappings.append((Path(source), destination.replace("\\", "/")))
        return mappings
    env = stack.env
    mappings: list[tuple[Path, str]] = []
    candidates = [
        ("INPLACE_STORAGE_HOST_PATH", "IMPORT_MOUNT_PATH", "/data"),
        ("OMERO_DATA_PATH", "OMERO_SERVER_DATA_LOCATION", "/OMERO"),
    ]
    for host_key, container_key, default_container in candidates:
        host = env.get(host_key)
        if not host:
            continue
        host_resolved = Path(_resolve_env_value(host, env))
        if not host_resolved.is_absolute():
            host_resolved = (stack.root / host_resolved).resolve()
        container = _resolve_env_value(env.get(container_key, default_container), env)
        mappings.append((host_resolved, container.replace("\\", "/")))
    return mappings


def _map_host_path_to_container(path: Path, stack: StackConfig) -> str | None:
    resolved = path.resolve()
    for host_root, container_root in _volume_mapping_from_env(stack):
        try:
            rel = resolved.relative_to(host_root.resolve())
        except ValueError:
            continue
        rel_posix = rel.as_posix()
        return container_root.rstrip("/") + ("/" + rel_posix if rel_posix else "")
    return None


def _copy_to_container(stack: StackConfig, input_path: Path, run_uuid: str) -> str:
    container_path = f"/tmp/cideconvolve_omero_probe/{run_uuid}/{input_path.name}"
    parent = str(Path(container_path).parent).replace("\\", "/")
    mkdir = stack.compose("exec", "-T", "biomero-importer", "mkdir", "-p", parent, timeout=60)
    if mkdir.returncode != 0:
        raise RuntimeError(f"Could not create container staging directory: {mkdir.stderr}")
    cp = stack.compose("cp", str(input_path), f"biomero-importer:{container_path}", timeout=1800)
    if cp.returncode != 0:
        raise RuntimeError(f"Could not copy input into biomero-importer container: {cp.stderr}")
    return container_path


def _container_input_path(stack: StackConfig, input_path: Path, run_uuid: str) -> tuple[str, dict[str, Any]]:
    mapped = _map_host_path_to_container(input_path, stack)
    if mapped:
        return mapped, {"method": "mounted_path", "container_path": mapped}
    copied = _copy_to_container(stack, input_path, run_uuid)
    return copied, {
        "method": "docker_cp_to_tmp",
        "container_path": copied,
        "note": "Input was not under a detected BIOMERO mounted host path, so it was staged in the importer container.",
    }


def _collect_logs(stack: StackConfig, report_dir: Path, uuids: list[str]) -> list[dict[str, Any]]:
    copied: list[dict[str, Any]] = []
    logs_dir = report_dir / "import_logs"
    logs_dir.mkdir(exist_ok=True)
    for uid in uuids:
        for pattern in (f"cli.{uid}*", f"cli.{uid}_*"):
            proc = stack.compose("exec", "-T", "biomero-importer", "sh", "-lc", f"ls -1 /auto-importer/logs/{pattern} 2>/dev/null || true", timeout=60)
            for remote in (proc.stdout or "").splitlines():
                remote = remote.strip()
                if not remote:
                    continue
                local = logs_dir / Path(remote).name
                cp = stack.compose("cp", f"biomero-importer:{remote}", str(local), timeout=120)
                copied.append({"remote": remote, "local": str(local), "copied": cp.returncode == 0, "stderr": cp.stderr})
    return copied


def _copy_from_container(stack: StackConfig, remote_path: str, local_path: Path) -> dict[str, Any]:
    if local_path.exists():
        if local_path.is_dir():
            shutil.rmtree(local_path)
        else:
            local_path.unlink()
    local_path.parent.mkdir(parents=True, exist_ok=True)
    cp = stack.compose("cp", f"biomero-importer:{remote_path}", str(local_path), timeout=1800)
    return {
        "remote": remote_path,
        "local": str(local_path),
        "copied": cp.returncode == 0,
        "stderr": cp.stderr,
    }


def _cleanup_container_staging(stack: StackConfig, container_path: str) -> None:
    if not container_path.startswith("/tmp/cideconvolve_omero_probe/"):
        return
    stack.compose("exec", "-T", "biomero-importer", "rm", "-rf", str(Path(container_path).parent).replace("\\", "/"), timeout=120)


def _compare_source_omero_to_zarr(source: dict[str, Any], zarr_info: dict[str, Any]) -> dict[str, Any]:
    pixels = source.get("pixels", {})
    arrays = zarr_info.get("arrays", {})
    first = arrays.get("0") or next(iter(arrays.values()), {})
    shape = first.get("shape") or []
    out: dict[str, Any] = {}
    if len(shape) == 5:
        expected_dims = {
            "SizeT": pixels.get("size_t"),
            "SizeC": pixels.get("size_c"),
            "SizeZ": pixels.get("size_z"),
            "SizeY": pixels.get("size_y"),
            "SizeX": pixels.get("size_x"),
        }
        actual_dims = {
            "SizeT": shape[0],
            "SizeC": shape[1],
            "SizeZ": shape[2],
            "SizeY": shape[3],
            "SizeX": shape[4],
        }
        for key, expected in expected_dims.items():
            actual = actual_dims.get(key)
            out[key] = {
                "source_omero": expected,
                "slurm_input_zarr": actual,
                "status": "matched" if str(expected) == str(actual) else "changed",
            }
    attrs = zarr_info.get("attrs", {})
    omero_attrs = attrs.get("omero", {}) if isinstance(attrs.get("omero"), dict) else {}
    zarr_channels = omero_attrs.get("channels") or []
    source_channels = source.get("channels") or []
    out["channel_count"] = {
        "source_omero": len(source_channels),
        "slurm_input_zarr": len(zarr_channels),
        "status": "matched" if len(source_channels) == len(zarr_channels) else "changed",
    }
    for index, channel in enumerate(source_channels):
        zarr_channel = zarr_channels[index] if index < len(zarr_channels) and isinstance(zarr_channels[index], dict) else {}
        expected = channel.get("name")
        actual = zarr_channel.get("label")
        out[f"Channel{index + 1}Name"] = {
            "source_omero": expected,
            "slurm_input_zarr": actual,
            "status": "matched" if str(expected or "") == str(actual or "") else "changed",
        }
    zarr_ome = zarr_info.get("ome_xml_summary", {}).get("pixels", {})
    for src_key, dst_key in (
        ("physical_size_x", "PhysicalSizeX"),
        ("physical_size_y", "PhysicalSizeY"),
        ("physical_size_z", "PhysicalSizeZ"),
    ):
        obj = pixels.get(src_key)
        expected = obj.get("value") if isinstance(obj, dict) else obj
        actual = zarr_ome.get(dst_key)
        if expected is None and actual is None:
            continue
        try:
            matched = abs(float(expected) - float(actual)) < 1e-6
        except Exception:
            matched = str(expected or "") == str(actual or "")
        out[dst_key] = {
            "source_omero": expected,
            "slurm_input_zarr": actual,
            "status": "matched" if matched else "changed",
        }
    return out


def _compare_input_to_omero(input_info: dict[str, Any], import_results: dict[str, Any]) -> dict[str, Any]:
    def pixel_type_matches(expected: Any, actual: Any) -> bool:
        aliases = {
            "float32": "float",
            "float": "float",
            "float64": "double",
            "double": "double",
        }
        exp = aliases.get(str(expected).lower(), str(expected).lower())
        got = aliases.get(str(actual).lower(), str(actual).lower())
        return exp == got

    expected_pixels: dict[str, Any] = {}
    if input_info.get("kind") == "ome-tiff":
        pixels = input_info.get("ome_xml_summary", {}).get("pixels", {})
        expected_pixels = pixels
    elif input_info.get("kind") == "ome-zarr":
        pixels = input_info.get("ome_xml_summary", {}).get("pixels", {})
        expected_pixels = pixels
        if not expected_pixels:
            arrays = input_info.get("arrays", {})
            first = arrays.get("0") or next(iter(arrays.values()), {})
            shape = first.get("shape") or []
            if len(shape) == 5:
                expected_pixels = {"SizeT": shape[0], "SizeC": shape[1], "SizeZ": shape[2], "SizeY": shape[3], "SizeX": shape[4]}
            first_field = input_info.get("first_hcs_field") or {}
            first_field_arrays = first_field.get("arrays") or {}
            first_field_level0 = first_field_arrays.get("0") or next(iter(first_field_arrays.values()), {})
            field_shape = first_field_level0.get("shape") or []
            if len(field_shape) == 5:
                expected_pixels = {
                    "SizeT": field_shape[0],
                    "SizeC": field_shape[1],
                    "SizeZ": field_shape[2],
                    "SizeY": field_shape[3],
                    "SizeX": field_shape[4],
                    "Type": str(first_field_level0.get("dtype") or ""),
                }
            field_attrs = first_field.get("attrs") or {}
            multiscales = field_attrs.get("multiscales") or []
            try:
                scale = multiscales[0]["datasets"][0]["coordinateTransformations"][0]["scale"]
                if len(scale) >= 5:
                    expected_pixels.update({
                        "PhysicalSizeZ": scale[2],
                        "PhysicalSizeY": scale[3],
                        "PhysicalSizeX": scale[4],
                    })
            except Exception:
                pass

    comparisons: dict[str, Any] = {}
    expected_map = {
        "SizeX": "size_x",
        "SizeY": "size_y",
        "SizeZ": "size_z",
        "SizeC": "size_c",
        "SizeT": "size_t",
        "Type": "type",
    }
    images = []
    for result in import_results.values():
        objects = result.get("objects", {})
        images.extend(objects.get("images", []))
        for plate in objects.get("plates", []):
            images.extend(plate.get("images", []))
    if not images:
        return {"note": "No imported images were available for comparison."}
    first_pixels = images[0].get("pixels", {})
    for src_key, dst_key in expected_map.items():
        expected = expected_pixels.get(src_key)
        actual = first_pixels.get(dst_key)
        if expected is None:
            continue
        if src_key == "Type":
            matched = pixel_type_matches(expected, actual)
        else:
            matched = str(expected).lower() == str(actual).lower()
        comparisons[src_key] = {
            "expected_input": expected,
            "omero": actual,
            "status": "matched" if matched else "changed",
        }
    for src_key, dst_key in (
        ("PhysicalSizeX", "physical_size_x"),
        ("PhysicalSizeY", "physical_size_y"),
        ("PhysicalSizeZ", "physical_size_z"),
    ):
        expected = expected_pixels.get(src_key)
        actual_obj = first_pixels.get(dst_key) or {}
        actual = actual_obj.get("value") if isinstance(actual_obj, dict) else actual_obj
        if expected is None:
            continue
        try:
            ok = abs(float(expected) - float(actual)) < 1e-6
        except Exception:
            ok = str(expected) == str(actual)
        comparisons[src_key] = {
            "expected_input": expected,
            "omero": actual,
            "status": "matched" if ok else "changed",
        }
    expected_channels = []
    if input_info.get("kind") == "ome-zarr":
        attrs = input_info.get("attrs", {})
        root_omero = attrs.get("omero") if isinstance(attrs.get("omero"), dict) else {}
        expected_channels = root_omero.get("channels") or []
        if not expected_channels:
            first_field = input_info.get("first_hcs_field") or {}
            field_attrs = first_field.get("attrs") or {}
            field_omero = field_attrs.get("omero") if isinstance(field_attrs.get("omero"), dict) else {}
            expected_channels = field_omero.get("channels") or []
    actual_channels = images[0].get("channels") or []
    for index, expected_channel in enumerate(expected_channels):
        if not isinstance(expected_channel, dict):
            continue
        actual_channel = actual_channels[index] if index < len(actual_channels) and isinstance(actual_channels[index], dict) else {}
        for key, label in (
            ("emission_wavelength", "EmissionWavelength"),
            ("excitation_wavelength", "ExcitationWavelength"),
        ):
            expected = expected_channel.get(key)
            actual_obj = actual_channel.get(key)
            actual = actual_obj.get("value") if isinstance(actual_obj, dict) else actual_obj
            if expected is None and actual is None:
                continue
            try:
                ok = abs(float(expected) - float(actual)) < 1e-6
            except Exception:
                ok = str(expected or "") == str(actual or "")
            comparisons[f"Channel{index + 1}{label}"] = {
                "expected_input": expected,
                "omero": actual,
                "status": "matched" if ok else "missing" if actual is None else "changed",
            }
    return comparisons


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    def result_objects(result: dict[str, Any]) -> dict[str, Any]:
        if isinstance(result.get("objects"), dict):
            return result["objects"]
        if "images" in result or "plates" in result:
            return {
                "images": result.get("images") or [],
                "plates": result.get("plates") or [],
            }
        return {}

    def fmt_len(value: Any) -> str:
        if not isinstance(value, dict):
            return "" if value is None else str(value)
        if value.get("value") is None:
            return ""
        unit = value.get("unit") or ""
        return f"{value.get('value')} {unit}".strip()

    def fmt_channel(channel: dict[str, Any]) -> str:
        bits = [str(channel.get("name") or f"Ch{channel.get('index')}")]
        em = fmt_len(channel.get("emission_wavelength"))
        ex = fmt_len(channel.get("excitation_wavelength"))
        if em:
            bits.append(f"em {em}")
        if ex:
            bits.append(f"ex {ex}")
        return ", ".join(bits)

    lines = [
        "# OMERO Import Metadata Probe Report",
        "",
        f"- Input: `{report.get('input_path')}`",
        f"- Created: `{report.get('created_at')}`",
        f"- Mode: `{report.get('mode')}`",
        f"- Target: `{report.get('target')}`",
        f"- Container input: `{report.get('container_input', {}).get('container_path')}`",
        "",
        "## Input Metadata",
        "",
        f"- Kind: `{report.get('input_metadata', {}).get('kind')}`",
    ]
    input_meta = report.get("input_metadata", {})
    if input_meta.get("kind") == "ome-tiff":
        first = input_meta.get("first_page", {})
        lines.extend(
            [
                f"- OME: `{input_meta.get('is_ome')}`",
                f"- BigTIFF: `{input_meta.get('is_bigtiff')}`",
                f"- Tiled: `{first.get('is_tiled')}`",
                f"- Compression: `{first.get('compression')}`",
                f"- Predictor: `{first.get('predictor')}`",
                f"- SubIFDs: `{first.get('subifds')}`",
                f"- Private tag 65000: `{'yes' if 'cideconvolve_private_tag_65000' in input_meta else 'no'}`",
            ]
        )
    elif input_meta.get("kind") == "ome-zarr":
        attrs = input_meta.get("attrs", {})
        first_field = input_meta.get("first_hcs_field") or {}
        first_field_attrs = first_field.get("attrs") or {}
        first_field_omero = first_field_attrs.get("omero") or {}
        first_field_channels = first_field_omero.get("channels") or []
        lines.extend(
            [
                f"- HCS plate: `{input_meta.get('is_hcs_plate')}`",
                f"- Multiscales: `{len(attrs.get('multiscales') or [])}`",
                f"- OMERO attr: `{'yes' if 'omero' in attrs else 'no'}`",
                f"- CIDeconvolve attr: `{'yes' if 'cideconvolve' in attrs else 'no'}`",
                f"- OME/METADATA.ome.xml: `{'yes' if input_meta.get('ome_xml_length') else 'no'}`",
            ]
        )
        if first_field:
            lines.extend(
                [
                    f"- First HCS field: `{first_field.get('path')}`",
                    f"- First field multiscales: `{len(first_field_attrs.get('multiscales') or [])}`",
                    f"- First field OMERO attr: `{'yes' if first_field_omero else 'no'}`",
                    f"- First field channels: `{len(first_field_channels)}`",
                ]
            )
            if first_field_channels:
                channel_bits = []
                for idx, channel in enumerate(first_field_channels[:8]):
                    em = channel.get("emission_wavelength")
                    ex = channel.get("excitation_wavelength")
                    label = channel.get("label") or f"Ch{idx}"
                    channel_bits.append(f"{label}: em={em}, ex={ex}")
                lines.append(f"- First field channel wavelengths: `{' | '.join(channel_bits)}`")
    if report.get("slurm_input_export"):
        export = report["slurm_input_export"]
        lines.extend(
            [
                "",
                "## Slurm Input Export",
                "",
                f"- Source OMERO Image: `{export.get('source_image_id')}`",
                f"- Exported Zarr: `{export.get('local_zarr_path')}`",
                f"- Shape: `{export.get('shape')}`",
                f"- Dtype: `{export.get('dtype')}`",
                "",
            ]
        )
        export_comparison = report.get("slurm_input_export_comparison", {})
        if export_comparison:
            lines.append("| Field | Source OMERO | Slurm-input Zarr | Status |")
            lines.append("| --- | --- | --- | --- |")
            for key, row in export_comparison.items():
                if isinstance(row, dict):
                    lines.append(f"| {key} | `{row.get('source_omero')}` | `{row.get('slurm_input_zarr')}` | {row.get('status')} |")
            lines.append("")
    lines.extend(["", "## Import Results", ""])
    for name, result in report.get("imports", {}).items():
        lines.append(f"### {name}")
        lines.append("")
        if result.get("error"):
            lines.append(f"- Error: `{result['error']}`")
        else:
            lines.append(f"- UUID: `{result.get('uuid')}`")
            lines.append(f"- Image IDs: `{result.get('image_ids')}`")
            lines.append(f"- Plate IDs: `{result.get('plate_ids')}`")
            if "ingest_rows" in result:
                stages = [row.get("stage") for row in result.get("ingest_rows", [])]
                lines.append(f"- BIOMERO ingest stages: `{stages}`")
            objects = result_objects(result)
            plates = objects.get("plates") or []
            images = objects.get("images") or []
            if plates:
                lines.append("")
                lines.append("| Plate | Wells/Fields | First field pixels | First field channels |")
                lines.append("| --- | --- | --- | --- |")
                for plate in plates:
                    plate_images = plate.get("images") or []
                    first_image = plate_images[0] if plate_images else {}
                    pixels = first_image.get("pixels") or {}
                    channels = first_image.get("channels") or []
                    dims = ""
                    if pixels:
                        dims = (
                            f"{pixels.get('size_x')} x {pixels.get('size_y')} x {pixels.get('size_z')}; "
                            f"C={pixels.get('size_c')}; T={pixels.get('size_t')}; "
                            f"{pixels.get('type')}; "
                            f"px={fmt_len(pixels.get('physical_size_x'))}, "
                            f"py={fmt_len(pixels.get('physical_size_y'))}, "
                            f"pz={fmt_len(pixels.get('physical_size_z'))}"
                        )
                    channel_text = "<br>".join(fmt_channel(ch) for ch in channels[:8])
                    lines.append(
                        f"| `{plate.get('id')}` {plate.get('name')} | "
                        f"`{plate.get('image_count_reported', len(plate.get('image_ids') or []))}` fields | "
                        f"`{dims}` | {channel_text or ''} |"
                    )
            elif images:
                lines.append("")
                lines.append("| Image | Pixels | Channels |")
                lines.append("| --- | --- | --- |")
                for image in images:
                    pixels = image.get("pixels") or {}
                    dims = (
                        f"{pixels.get('size_x')} x {pixels.get('size_y')} x {pixels.get('size_z')}; "
                        f"C={pixels.get('size_c')}; T={pixels.get('size_t')}; {pixels.get('type')}"
                    )
                    channel_text = "<br>".join(fmt_channel(ch) for ch in (image.get("channels") or [])[:8])
                    lines.append(f"| `{image.get('id')}` {image.get('name')} | `{dims}` | {channel_text or ''} |")
        lines.append("")
    lines.extend(["## Comparison", ""])
    comparison = report.get("comparison", {})
    if comparison:
        lines.append("| Field | Input | OMERO | Status |")
        lines.append("| --- | --- | --- | --- |")
        for key, row in comparison.items():
            if not isinstance(row, dict):
                continue
            lines.append(f"| {key} | `{row.get('expected_input')}` | `{row.get('omero')}` | {row.get('status')} |")
    else:
        lines.append("No comparison was available.")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _print_summary(report: dict[str, Any]) -> None:
    print("\nOMERO Import Metadata Probe")
    print("===========================")
    print(f"Input : {report.get('input_path')}")
    print(f"Target: {report.get('target')}")
    print(f"Report: {report.get('report_dir')}")
    print("")
    for name, result in report.get("imports", {}).items():
        status = "failed" if result.get("error") else "ok"
        print(f"{name:8s}: {status}")
        if result.get("error"):
            print(f"          {result['error']}")
        else:
            print(f"          images={result.get('image_ids')} plates={result.get('plate_ids')}")
    print("")
    comparison = report.get("comparison", {})
    changed = [key for key, row in comparison.items() if isinstance(row, dict) and row.get("status") != "matched"]
    if changed:
        print("Changed/missing fields:", ", ".join(changed))
    elif comparison:
        print("Compared fields matched.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Probe OMERO/BIOMERO metadata import behavior for OME-TIFF and OME-Zarr.")
    parser.add_argument("--input", dest="input_path", help="OME-TIFF file or OME-Zarr folder to import.")
    parser.add_argument("--nl-biomero-root", default=str(DEFAULT_NL_BIOMERO_ROOT), help="Path to the NL-BIOMERO checkout.")
    parser.add_argument("--importer-container", help="Use this already-running biomero-importer container instead of docker compose.")
    parser.add_argument("--mode", choices=["direct", "biomero", "both"], default="both", help="Import path to exercise.")
    parser.add_argument("--target", help="Import target, e.g. Dataset:123 or Screen:456.")
    parser.add_argument("--user", help="OMERO user to impersonate for import.")
    parser.add_argument("--group", help="OMERO group for the import session.")
    parser.add_argument("--cleanup", choices=["always", "success", "never"], default="success", help="Remove imported probe objects after report.")
    parser.add_argument("--out", help="Report output directory.")
    parser.add_argument("--existing-image", action="append", type=int, default=[], help="Read-only report for an existing OMERO Image ID. Can be repeated.")
    parser.add_argument("--existing-plate", action="append", type=int, default=[], help="Read-only report for an existing OMERO Plate ID. Can be repeated.")
    parser.add_argument("--slurm-input-image", type=int, help="Export this existing OMERO Image to OME-Zarr as a Slurm-job input and inspect the metadata loss point.")
    parser.add_argument("--transfer", default="upload", choices=["upload", "ln_s", "ln", "cp", "ln_rm"], help="Transfer mode for direct TIFF import.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    stack = StackConfig.from_container(args.importer_container) if args.importer_container else StackConfig.from_root(Path(args.nl_biomero_root))
    if not args.user or not args.group:
        raise SystemExit("--user and --group are required so the probe can inspect OMERO objects in the correct context.")
    if not args.target and not args.existing_image and not args.existing_plate and not args.slurm_input_image:
        raise SystemExit("--target is required unless --existing-image, --existing-plate, or --slurm-input-image is used.")

    target_type = target_id = None
    if args.target:
        target_type, target_id = _parse_target(args.target)

    if args.input_path and args.slurm_input_image:
        raise SystemExit("Use either --input or --slurm-input-image, not both.")

    input_path = Path(args.input_path).resolve() if args.input_path else None
    if input_path is not None and not input_path.exists():
        raise SystemExit(f"Input path does not exist: {input_path}")

    report_dir = Path(args.out).resolve() if args.out else DEFAULT_REPORT_ROOT / f"probe_{_now_stamp()}_{uuid.uuid4().hex[:8]}"
    report_dir.mkdir(parents=True, exist_ok=True)

    run_id = uuid.uuid4().hex
    container_input: dict[str, Any] = {}
    input_metadata: dict[str, Any] = {}
    slurm_export_container_path = ""
    slurm_export_local_path: Path | None = None
    if input_path is not None:
        input_metadata = inspect_input(input_path)
        container_path, container_input = _container_input_path(stack, input_path, run_id)
    elif args.slurm_input_image:
        export_container_path = f"/tmp/cideconvolve_omero_probe/{run_id}/slurm_input_image_{args.slurm_input_image}.ome.zarr"
        export_payload = {
            "action": "export_zarr",
            "image_id": int(args.slurm_input_image),
            "output_dir": export_container_path,
            "user": args.user,
            "group": args.group,
        }
        export_result = _run_container_helper(stack, export_payload, timeout=7200)
        slurm_export_container_path = export_result["container_zarr_path"]
        slurm_export_local_path = report_dir / Path(slurm_export_container_path).name
        export_copy = _copy_from_container(stack, slurm_export_container_path, slurm_export_local_path)
        if not export_copy.get("copied"):
            raise RuntimeError(f"Could not copy exported Slurm-input Zarr: {export_copy.get('stderr')}")
        input_path = slurm_export_local_path
        input_metadata = inspect_input(slurm_export_local_path)
        container_path = slurm_export_container_path
        container_input = {
            "method": "slurm_input_export",
            "container_path": container_path,
            "local_copy": str(slurm_export_local_path),
            "source_image_id": int(args.slurm_input_image),
        }
    else:
        container_path = ""

    report: dict[str, Any] = {
        "created_at": _dt.datetime.now().isoformat(),
        "input_path": str(input_path) if input_path else None,
        "mode": args.mode,
        "target": args.target,
        "report_dir": str(report_dir),
        "nl_biomero_root": str(stack.root),
        "compose_ps": _compose_ps(stack),
        "stack_env": _masked_env(stack.env),
        "container_input": container_input,
        "input_metadata": input_metadata,
        "imports": {},
    }
    if args.slurm_input_image:
        report["slurm_input_export"] = {
            "source_image_id": int(args.slurm_input_image),
            "container_zarr_path": slurm_export_container_path,
            "local_zarr_path": str(slurm_export_local_path) if slurm_export_local_path else None,
            "source_omero": export_result.get("source_omero", {}),
            "shape": export_result.get("shape"),
            "chunks": export_result.get("chunks"),
            "dtype": export_result.get("dtype"),
        }
        report["slurm_input_export_comparison"] = _compare_source_omero_to_zarr(
            export_result.get("source_omero", {}),
            input_metadata,
        )

    base_payload = {
        "container_input": container_path,
        "target_type": target_type,
        "target_id": target_id,
        "user": args.user,
        "group": args.group,
        "transfer": args.transfer,
    }

    imported_for_cleanup: list[tuple[str, list[int], list[int], bool]] = []
    try:
        if args.existing_image or args.existing_plate:
            existing_payload = {
                **base_payload,
                "action": "existing",
                "image_ids": args.existing_image,
                "plate_ids": args.existing_plate,
            }
            report["imports"]["existing"] = _run_container_helper(stack, existing_payload)

        if input_path is not None and args.target:
            modes = ["direct", "biomero"] if args.mode == "both" else [args.mode]
            for mode in modes:
                import_uuid = f"probe-{mode}-{uuid.uuid4()}"
                payload = {**base_payload, "action": mode, "uuid": import_uuid}
                try:
                    result = _run_container_helper(stack, payload)
                    report["imports"][mode] = result
                    report["imports"][mode]["log_files"] = _collect_logs(stack, report_dir, [import_uuid])
                    imported_for_cleanup.append((mode, result.get("image_ids", []), result.get("plate_ids", []), True))
                except Exception as exc:
                    report["imports"][mode] = {"uuid": import_uuid, "error": str(exc)}
                    report["imports"][mode]["log_files"] = _collect_logs(stack, report_dir, [import_uuid])
                    imported_for_cleanup.append((mode, [], [], False))

        report["comparison"] = _compare_input_to_omero(input_metadata, report["imports"])

        cleanup_records = []
        should_cleanup_success = args.cleanup == "success" and all(ok for _, _, _, ok in imported_for_cleanup)
        should_cleanup = args.cleanup == "always" or should_cleanup_success
        if should_cleanup:
            for mode, image_ids, plate_ids, _ok in imported_for_cleanup:
                if not image_ids and not plate_ids:
                    continue
                payload = {
                    **base_payload,
                    "action": "cleanup",
                    "image_ids": image_ids,
                    "plate_ids": plate_ids,
                }
                try:
                    cleanup_records.append({"mode": mode, **_run_container_helper(stack, payload, timeout=600)})
                except Exception as exc:
                    cleanup_records.append({"mode": mode, "error": str(exc), "image_ids": image_ids, "plate_ids": plate_ids})
        report["cleanup"] = {"policy": args.cleanup, "records": cleanup_records}
    finally:
        if container_input.get("method") in {"docker_cp_to_tmp", "slurm_input_export"}:
            _cleanup_container_staging(stack, container_input.get("container_path", ""))

    (report_dir / "report.json").write_text(_safe_json(report), encoding="utf-8")
    _write_markdown(report, report_dir / "report.md")
    _print_summary(report)
    return 0 if not any(v.get("error") for v in report.get("imports", {}).values() if isinstance(v, dict)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
