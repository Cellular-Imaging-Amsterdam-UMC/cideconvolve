# Repository Notes

- For Python commands and tests in this repository, use the `deconvolve` Conda environment:
  `C:\Users\p000881\AppData\Local\miniconda3\envs\deconvolve\python.exe`
- Run pytest through that interpreter, for example:
  `C:\Users\p000881\AppData\Local\miniconda3\envs\deconvolve\python.exe -m pytest`
- When asked to test metadata import/export behavior against local OMERO or BIOMERO, use the probe tool first:
  `tools\omero_import_metadata_probe\run.cmd`
- For OMERO round-trip checks, prefer this probe over ad hoc scripts because it reports source metadata, Slurm-input OME-Zarr metadata, direct import metadata, BIOMERO import metadata, and cleanup status.
