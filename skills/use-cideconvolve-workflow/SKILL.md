---
name: use-cideconvolve-workflow
description: Configure, launch, monitor, and recover the Bilayers CIDeconvolve workflow for microscopy images using CI-RL, CI-RL with total-variation regularisation, or sparse-Hessian deconvolution.
metadata:
  version: "1"
---

# Use CIDeconvolve Workflow

1. Confirm the requested input images or folders and the configured
   CIDeconvolve workflow revision. Do not start execution from inspection
   alone.
2. Inspect the workflow descriptor (`config.yaml`) and available CPU/GPU
   compute resources. Treat that descriptor as the executable parameter
   contract.
3. Validate readable supported inputs, requested output format, parameter
   types and ranges, method-specific settings, metadata-override intent, and
   requested compute device. Image metadata supplies optics defaults unless
   metadata override is explicitly requested.
4. Present the resolved workflow revision, input object, parameters, expected
   image outputs, resource choice, and side effects.
5. Obtain explicit confirmation immediately before submission.
6. Submit exactly once through the available Bilayers workflow execution
   interface. Retain the returned run or job ID.
7. Monitor that ID. Do not resubmit merely because status is delayed or a
   client response times out.
8. Verify completion against the image-output contract below. Consult the
   workflow logs only after validation, execution, or output verification
   fails.
9. Record the workflow key, configured ref, resolved commit, parameters,
   input stores, run ID, timestamps, final status, and discovered outputs as
   provenance.

Require explicit confirmation before submission, cancellation, deletion, or
overwrite. A scheduler completion alone is not proof of successful output
creation.

## Image-output contract

Normal runs produce one deconvolved image for each processed source image:

```text
<source>_decon.ome.zarr
<source>_decon.ome.tiff
```

The chosen `output_format` determines which extension is written. A requested
Z projection adds a projection suffix before the extension. Each streamed
output has a matching sidecar provenance file:

```text
<output>.provenance.json
```

Benchmark mode instead writes `benchmark_metrics_<source>.csv` and PNG
montages. Those are comparison artifacts, not normal deconvolved-image
outputs.

Treat a run as successful only when:

1. the workflow execution interface reports a successful final status for the
   retained run ID;
2. the expected OME-TIFF or OME-Zarr output is discoverable in the output
   location and associated with the intended source input;
3. the output can be enumerated or opened as the requested image format; and
4. the output provenance JSON is present for streamed normal runs, or the
   expected CSV and PNG artifacts are present for benchmark runs.

CIDeconvolve produces images and image provenance, not a measurements
database. Do not look for segmentation labels, object measurements, DuckDB,
or SQLite output unless another workflow explicitly created them.