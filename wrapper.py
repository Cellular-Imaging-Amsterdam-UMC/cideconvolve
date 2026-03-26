"""
wrapper.py — BIAFLOWS-compatible entrypoint for CIDeconvolve.

Parses BIAFLOWS job parameters (--infolder, --outfolder, --gtfolder, etc.)
via bioflows_local, then processes each input image through the
deconvolution pipeline in cidecon.py and writes results to the output
folder.

Usage (inside Docker):
    python wrapper.py --infolder /data/in --outfolder /data/out --gtfolder /data/gt --local

Usage (local):
    python wrapper.py --infolder ./infolder --outfolder ./outfolder --gtfolder ./gtfolder --local --iterations 40 --method sdeconv_rl
"""
import os
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).parent))

from bioflows_local import (
    CLASS_SPTCNT,
    BiaflowsJob,
    get_discipline,
    prepare_data,
)

# Import deconvolve first (handles torch-before-numpy DLL load order)
from deconvolve import (
    MAX_TILE_XY,
    MAX_TILE_Z,
    deconvolve_image,
    load_image,
    save_result,
)


def main(argv):
    with BiaflowsJob.from_cli(argv) as bj:
        parameters = getattr(bj, "parameters", SimpleNamespace())

        # Extract parameters with defaults from descriptor.json
        iterations = int(getattr(parameters, "iterations", 40))
        blocks_raw = getattr(parameters, "blocks", "auto")
        method = getattr(parameters, "method", "sdeconv_rl")
        device_param = getattr(parameters, "device", "auto")
        device = None if device_param in (None, "auto") else device_param
        benchmark_mode = bool(getattr(parameters, "benchmark", False))
        projection = str(getattr(parameters, "projection", "none")).lower()
        save_psf = bool(getattr(parameters, "psf", False))

        # Parse blocks
        if isinstance(blocks_raw, str) and blocks_raw.strip().lower() == "auto":
            n_blocks = "auto"
        else:
            n_blocks = max(int(blocks_raw), 0)

        print("=" * 70)
        print("CIDeconvolve — BIAFLOWS Workflow")
        print("=" * 70)
        print(f"  Input dir    : {bj.input_dir}")
        print(f"  Output dir   : {bj.output_dir}")
        print(f"  Method       : {method}")
        print(f"  Iterations   : {iterations}")
        print(f"  Blocks       : {n_blocks}")
        print(f"  Device       : {device_param}")
        print(f"  Benchmark    : {benchmark_mode}")
        print(f"  Projection   : {projection}")
        print(f"  Save PSF     : {save_psf}")

        # Prepare data directories and collect input images
        in_imgs, _, in_path, _, out_path, tmp_path = prepare_data(
            get_discipline(bj, default=CLASS_SPTCNT), bj, is_2d=False, **bj.flags
        )

        if not in_imgs:
            print("No input images found. Exiting.")
            return

        print(f"\nFound {len(in_imgs)} input image(s).")

        for img_resource in in_imgs:
            img_path = Path(in_path) / img_resource.filename
            print(f"\n{'─' * 60}")
            print(f"Processing: {img_resource.filename}")
            print(f"{'─' * 60}")

            t0 = time.time()

            try:
                # Load image and extract metadata
                data = load_image(img_path)
                meta = data["metadata"]
                images = data["images"]

                print(f"  Channels: {len(images)}")
                for i, img in enumerate(images):
                    print(f"    Ch{i}: shape={img.shape}, dtype={img.dtype}")

                # Deconvolve
                result = deconvolve_image(
                    img_path,
                    method=method,
                    niter=iterations,
                    n_blocks=n_blocks,
                    device=device,
                )

                if result is None:
                    print(f"  WARNING: deconvolve_image returned None for {img_resource.filename}")
                    continue

                # Save result to output folder
                stem = Path(img_resource.filename).stem
                # Strip .ome suffix if present (e.g. "foo.ome" stem from "foo.ome.tiff")
                if stem.lower().endswith(".ome"):
                    stem = stem[:-4]

                # Determine if data is 3D
                is_3d = result["channels"][0].ndim == 3

                if projection in ("mip", "sum") and is_3d and not benchmark_mode:
                    # Projection mode: save 2D Z-projection only
                    import numpy as np
                    out_name = f"{stem}_decon_{projection}.ome.tiff"
                    out_file = Path(out_path) / out_name
                    # Replace channels with Z-projected 2D images
                    proj_result = dict(result)
                    if projection == "mip":
                        proj_result["channels"] = [
                            ch.max(axis=0) for ch in result["channels"]
                        ]
                        if result.get("source_channels"):
                            proj_result["source_channels"] = [
                                ch.max(axis=0) for ch in result["source_channels"]
                            ]
                    else:  # sum
                        proj_result["channels"] = [
                            ch.astype(np.float32).sum(axis=0) for ch in result["channels"]
                        ]
                        if result.get("source_channels"):
                            proj_result["source_channels"] = [
                                ch.astype(np.float32).sum(axis=0) for ch in result["source_channels"]
                            ]
                    save_result(proj_result, str(out_file))
                    print(f"  Saved {projection.upper()}: {out_name}")
                    # Clean up any extra MIP files from save_result (2D has none,
                    # but be safe)
                    import glob
                    for pattern in ("mip_*.ome.tiff", "mip_*.png"):
                        for f in glob.glob(str(Path(out_path) / pattern)):
                            try:
                                os.remove(f)
                            except OSError:
                                pass
                else:
                    out_name = f"{stem}_decon.ome.tiff"
                    out_file = Path(out_path) / out_name
                    save_result(result, str(out_file))
                    print(f"  Saved: {out_name}")

                    if not benchmark_mode:
                        # Normal mode: remove MIP files generated by save_result,
                        # keep only the deconvolved .ome.tiff
                        import glob
                        for pattern in ("mip_*.ome.tiff", "mip_*.png"):
                            for f in glob.glob(str(Path(out_path) / pattern)):
                                try:
                                    os.remove(f)
                                except OSError:
                                    pass
                    else:
                        # Benchmark mode: save_result already wrote MIP TIFF + PNG
                        mip_name = f"mip_{out_name}"
                        mip_png = Path(mip_name).with_suffix(".png")
                        print(f"  MIP:   {mip_png}")

            except Exception as exc:
                print(f"  ERROR processing {img_resource.filename}: {exc}")
                import traceback
                traceback.print_exc()
                continue

            elapsed = time.time() - t0
            print(f"  Time: {elapsed:.1f}s")

        print(f"\n{'=' * 70}")
        print("CIDeconvolve workflow complete.")
        print(f"{'=' * 70}")

        # Clean up tmp folder
        if tmp_path and Path(tmp_path).exists():
            shutil.rmtree(tmp_path, ignore_errors=True)
            print(f"Cleaned up tmp folder: {tmp_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
