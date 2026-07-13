# OMERO Import Metadata Probe

This tool imports an OME-TIFF or OME-Zarr into a local NL-BIOMERO Docker stack
and writes a report showing what was present before import and what OMERO and
BIOMERO stored afterward.

The tool is host-side, but OMERO operations run inside the `biomero-importer`
container so the same OMERO Python client, BIOMERO importer code, settings, and
environment variables are used as in the local stack.

## Example

```cmd
tools\omero_import_metadata_probe\run.cmd ^
  --input C:\Users\p000881\Downloads\3dtime.ome_ci_rl_5i_TSERIES.ome.tiff ^
  --target Dataset:123 ^
  --user root ^
  --group system ^
  --mode both ^
  --cleanup success
```

For an existing OMERO image without importing anything:

```cmd
tools\omero_import_metadata_probe\run.cmd ^
  --existing-image 12345 ^
  --user root ^
  --group system ^
  --out C:\Users\p000881\Downloads\omero_probe_existing
```

To check the metadata-loss point before a Slurm job, export an existing OMERO
Image to the OME-Zarr that would be used as job input:

```cmd
tools\omero_import_metadata_probe\run.cmd ^
  --slurm-input-image 12345 ^
  --user root ^
  --group system ^
  --out C:\Users\p000881\Downloads\omero_probe_slurm_input
```

Add `--target Dataset:123 --mode both` to immediately import that exported
Zarr back through the direct and BIOMERO paths for a full round-trip report.

## Wrapper Round Trip

To rerun the DividingCell end-to-end test against the already-running local
OMERO/BIOMERO containers:

```cmd
tools\omero_import_metadata_probe\run_dividingcell_wrapper_roundtrip.cmd
```

By default this creates a new Dataset, imports
`localdata\DividingCellcrop.ome.tiff`, exports the imported OMERO image to the
Slurm-input OME-Zarr, runs `wrapper.py` twice (`projection=none` and
`projection=mip`, both with `output_format=ome-zarr`), copies the outputs to the
shared `/data` mount, imports both outputs through direct and BIOMERO paths, and
writes a combined summary under
`C:\Users\p000881\Downloads\cideconvolve_omero_roundtrips`.

Useful options:

```cmd
tools\omero_import_metadata_probe\run_dividingcell_wrapper_roundtrip.cmd ^
  --target Dataset:152 ^
  --existing-image 204 ^
  --iterations 5 ^
  --cleanup-imports never
```

To rerun the HCS plate workflow using
`localdata\cellsA1B1.ome.zarr`:

```cmd
tools\omero_import_metadata_probe\run_plate_cellsa1b1_wrapper_roundtrip.cmd
```

This creates or uses a `Screen:ID`, stages the source plate under the shared
`/data` mount, imports it as an OMERO Plate, runs the wrapper on the HCS
OME-Zarr plate, and imports the wrapper output plate through both the direct and
BIOMERO paths. The wrapper plate path currently writes stack HCS OME-Zarr
outputs; Z-projection output is not applied to HCS plate processing.

Useful options:

```cmd
tools\omero_import_metadata_probe\run_plate_cellsa1b1_wrapper_roundtrip.cmd ^
  --target Screen:123 ^
  --source-import-mode both ^
  --iterations 5 ^
  --cleanup-imports never
```

## Output

Each run writes:

- `report.json`: complete machine-readable input/import/OMERO metadata.
- `report.md`: readable summary with matched or changed fields.
- `import_logs\`: OMERO CLI log and error files when the importer created them.

The default report folder is `tools\omero_import_metadata_probe\reports\...`.

## Notes

- `--mode direct` performs a direct import without BIOMERO's post-import
  MapAnnotation.
- `--mode biomero` logs a synthetic BIOMERO import order and runs
  `DataPackageImporter.import_data_package`, including BIOMERO annotations.
- `--mode both` runs both paths against the same target.
- `--slurm-input-image` reads an existing OMERO Image through OMERO's pixel API,
  writes an OME-Zarr into the report folder, and compares source OMERO metadata
  against the generated Slurm-input Zarr before any re-import happens.
- `--cleanup success` removes imported probe Images/Plates only if all requested
  imports succeeded. Use `--cleanup never` when you want to inspect the objects
  in OMERO.web afterward.
- If the input is not under a mounted BIOMERO data path, the tool stages a copy
  inside the importer container under `/tmp/cideconvolve_omero_probe`.
