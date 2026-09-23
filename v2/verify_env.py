#!/usr/bin/env python3
"""Check the existing smolvla environment for SmolVLA and IsaacLab work."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


LOCAL_LEROBOT_SRC = Path(__file__).resolve().parents[1] / "v1" / "lerobot" / "src"
if LOCAL_LEROBOT_SRC.is_dir():
    sys.path.insert(0, str(LOCAL_LEROBOT_SRC))


def check_import(module_name: str) -> tuple[bool, str]:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # Import errors can be caused by native libraries.
        return False, f"{type(exc).__name__}: {exc}"
    version = getattr(module, "__version__", "version unavailable")
    return True, str(version)


def main() -> int:
    print(f"Python: {sys.executable}")
    print(f"Python version: {sys.version.split()[0]}")

    try:
        import torch

        print(f"[OK] torch {torch.__version__}")
        print(f"[{'OK' if torch.cuda.is_available() else 'FAIL'}] CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"     GPU: {torch.cuda.get_device_name(0)}")
    except Exception as exc:
        print(f"[FAIL] torch: {type(exc).__name__}: {exc}")

    required = ["lerobot", "transformers", "peft", "av"]
    optional = ["omni.isaac.lab", "isaaclab"]
    failures = 0

    for module_name in required + optional:
        ok, detail = check_import(module_name)
        label = "OK" if ok else ("FAIL" if module_name in required else "WARN")
        print(f"[{label}] {module_name}: {detail}")
        if module_name in required and not ok:
            failures += 1

    if failures:
        print("Environment check failed for one or more required packages.")
        return 1
    print("Required SmolVLA dependencies are importable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
