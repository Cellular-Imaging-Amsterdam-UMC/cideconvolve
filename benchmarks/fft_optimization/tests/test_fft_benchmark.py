import numpy as np

from fft_shapes import is_smooth, named_shapes, next_smooth
from metrics import comparison_metrics


def test_fft_shapes_are_large_enough_and_smooth():
    minimum = (113, 430, 430)
    shapes = named_shapes(minimum)
    assert shapes["exact"] == minimum
    assert all(a >= b for a, b in zip(shapes["smooth"], minimum))
    assert all(is_smooth(value) for value in shapes["smooth"])
    assert next_smooth(431) == 432


def test_identity_metrics_pass_strict_gate():
    image = np.arange(128, dtype=np.float32).reshape(2, 8, 8)
    metrics = comparison_metrics(image, image.copy())
    assert metrics["quality_pass"]
    assert metrics["ssim_global"] == 1.0
    assert metrics["nrmse"] == 0.0
