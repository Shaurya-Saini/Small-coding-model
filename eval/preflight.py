#!/usr/bin/env python
"""
preflight.py -- fast sanity check for the EVAL environment, run BEFORE the slow
model load so version/dependency problems surface in seconds, not after a 90s
download. Every failure we hit setting up APPS eval is checked here:

  * GPU present + compute capability >= 7.0
  * transformers on the 4.x line   (5.x removed the load_in_4bit kwarg)
  * datasets on the < 4.0 line      (4.0 removed dataset loading scripts)
  * bitsandbytes importable         (4-bit needs it; missing -> PackageNotFoundError)
  * accelerate present

Usage:
    python eval/preflight.py

Exit code 0 = good to run the harness; non-zero = fix what it prints first.
This checks the environment only; it never downloads a model or dataset.
"""
from __future__ import annotations

import importlib
import sys

GREEN, RED, YELLOW, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[0m"


def _ok(msg: str) -> None:
    print(f"{GREEN}[ OK ]{RESET} {msg}")


def _fail(msg: str, fix: str) -> None:
    print(f"{RED}[FAIL]{RESET} {msg}\n       fix: {fix}")


def _pkg_version(name: str) -> str | None:
    try:
        return importlib.import_module(name).__version__
    except Exception:  # noqa: BLE001
        return None


def main() -> int:
    problems = 0

    # --- GPU ------------------------------------------------------------------
    try:
        import torch
        if not torch.cuda.is_available():
            _fail("No CUDA GPU visible.",
                  "Select a GPU accelerator (T4). Unsloth/quantized eval needs one.")
            problems += 1
        else:
            cap = torch.cuda.get_device_capability(0)
            name = torch.cuda.get_device_name(0)
            n = torch.cuda.device_count()
            if cap[0] < 7:
                _fail(f"{name} compute capability {cap} < 7.0.",
                      "Use a T4 (7.5). P100 (6.0) is unsupported by the quant kernels.")
                problems += 1
            else:
                _ok(f"{n} GPU(s); primary {name}, capability {cap}, "
                    f"torch {torch.__version__} (CUDA {torch.version.cuda}).")
    except Exception as e:  # noqa: BLE001
        _fail(f"Could not import/inspect torch: {e}", "pip install torch (CUDA build).")
        problems += 1

    # --- transformers must be 4.x --------------------------------------------
    tv = _pkg_version("transformers")
    if tv is None:
        _fail("transformers not installed.", 'pip install "transformers>=4.44,<5.0"')
        problems += 1
    elif int(tv.split(".")[0]) >= 5:
        _fail(f"transformers {tv} is 5.x — it removed the load_in_4bit kwarg the "
              "harness uses.", 'pip install "transformers>=4.44,<5.0"')
        problems += 1
    else:
        _ok(f"transformers {tv} (4.x — load_in_4bit kwarg supported).")

    # --- datasets must be < 4.0 ----------------------------------------------
    dv = _pkg_version("datasets")
    if dv is None:
        _fail("datasets not installed.", 'pip install "datasets>=2.16,<4.0"')
        problems += 1
    elif int(dv.split(".")[0]) >= 4:
        _fail(f"datasets {dv} is >= 4.0 — it removed dataset loading scripts, so "
              "the harness's codeparrot/apps load fails.",
              'pip install "datasets>=2.16,<4.0"')
        problems += 1
    else:
        _ok(f"datasets {dv} (< 4.0 — loading scripts supported).")

    # --- bitsandbytes importable (the current blocker) -----------------------
    try:
        import bitsandbytes as bnb  # noqa: F401
        _ok(f"bitsandbytes {getattr(bnb, '__version__', '?')} importable "
            "(4-bit kernels available).")
    except Exception as e:  # noqa: BLE001
        _fail(f"bitsandbytes not importable: {e}",
              "pip install bitsandbytes  (needed for LOAD_IN=4bit).")
        problems += 1

    # --- accelerate -----------------------------------------------------------
    av = _pkg_version("accelerate")
    if av is None:
        _fail("accelerate not installed.", "pip install accelerate")
        problems += 1
    else:
        _ok(f"accelerate {av}.")

    print()
    if problems:
        print(f"{RED}Preflight FAILED with {problems} problem(s). "
              f"Fix the above, then re-run this before the harness.{RESET}")
        return 1
    print(f"{GREEN}Preflight PASSED — safe to run eval/run_apps_eval.sh.{RESET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
