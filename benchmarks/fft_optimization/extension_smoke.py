from __future__ import annotations

import sys

from fused_extension import load_fused_extension


def main() -> int:
    module, error = load_fused_extension(verbose=True)
    print(f"extension_loaded: {module is not None}")
    print(f"extension_error: {error}")
    if module is None:
        return 1
    expected = ("ratio_pitched", "update_pitched", "tv_update", "sparse_hessian_gradient")
    exports = [name for name in expected if hasattr(module, name)]
    print(f"extension_exports: {' '.join(exports)}")
    return 0 if len(exports) == len(expected) else 2


if __name__ == "__main__":
    raise SystemExit(main())
