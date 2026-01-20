# core/gpu_env.py
from __future__ import annotations
from typing import Dict


_GPU_ENV_KEYS = ("cuml", "cudf", "cupy", "cuda_runtime")


def collect_gpu_env() -> Dict[str, str]:
    env: Dict[str, str] = {}
    try:
        import cuml  # type: ignore
        env["cuml"] = getattr(cuml, "__version__", "unknown")
    except Exception:
        pass
    try:
        import cudf  # type: ignore
        env["cudf"] = getattr(cudf, "__version__", "unknown")
    except Exception:
        pass
    try:
        import cupy  # type: ignore
        env["cupy"] = getattr(cupy, "__version__", "unknown")
    except Exception:
        pass
    try:
        from numba import cuda  # type: ignore
        version = getattr(cuda, "runtime", None)
        if version is not None and hasattr(version, "get_version"):
            env["cuda_runtime"] = ".".join(str(x) for x in version.get_version())
    except Exception:
        pass
    return env


def diff_gpu_env(saved: Dict[str, str] | None, current: Dict[str, str]) -> Dict[str, Dict[str, str | None]]:
    if not saved:
        return {}
    mismatches: Dict[str, Dict[str, str | None]] = {}
    for key in _GPU_ENV_KEYS:
        saved_val = saved.get(key)
        current_val = current.get(key)
        if saved_val and current_val and saved_val != current_val:
            mismatches[key] = {"saved": saved_val, "current": current_val}
        elif saved_val and not current_val:
            mismatches[key] = {"saved": saved_val, "current": None}
    return mismatches
