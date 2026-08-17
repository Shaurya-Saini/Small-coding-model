#!/usr/bin/env python
"""
fix_pyext_py312.py -- patch the installed `pyext` module for Python 3.11+.

Why this is needed
------------------
The APPS scorer (`codeparrot/apps_metric`, downloaded by the harness at scoring
time) imports `pyext` to build a RuntimeModule that executes each generated
solution against the hidden tests. `pyext` calls `inspect.getargspec`, which was
deprecated in Python 3.0 and REMOVED in Python 3.11. Kaggle runs Python 3.12, so
the scoring step crashes -- after generation has already finished -- with:

    AttributeError: module 'inspect' has no attribute 'getargspec'

Same issue as FlagOpen/TACO #3 (an APPS-derived benchmark). The documented fix is
to restore `getargspec` from `getfullargspec`. We prepend a shim to pyext.py that
rebuilds the old 4-field namedtuple (so both attribute access and 4-tuple
unpacking keep working). Idempotent: safe to run repeatedly.

`run_apps_eval.sh` calls this automatically; you can also run it by hand:
    python eval/fix_pyext_py312.py
"""
from __future__ import annotations

import importlib.util
import sys

MARKER = "# --- py311+ getargspec shim (SCM) ---"
SHIM = '''# --- py311+ getargspec shim (SCM) ---
import inspect as _inspect, collections as _collections
if not hasattr(_inspect, "getargspec"):
    _ArgSpec = _collections.namedtuple("ArgSpec", "args varargs keywords defaults")
    def _getargspec(func):
        fas = _inspect.getfullargspec(func)
        return _ArgSpec(fas.args, fas.varargs, fas.varkw, fas.defaults)
    _inspect.getargspec = _getargspec
# --- end shim ---
'''


def main() -> int:
    # Locate pyext WITHOUT importing it (importing an unpatched pyext crashes).
    spec = importlib.util.find_spec("pyext")
    if spec is None or not spec.origin:
        print("pyext not found. If the APPS scorer needs it: pip install pyext")
        return 1

    path = spec.origin
    with open(path, encoding="utf-8") as fh:
        src = fh.read()

    if MARKER in src:
        print(f"pyext already patched: {path}")
        return 0

    with open(path, "w", encoding="utf-8") as fh:
        fh.write(SHIM + "\n" + src)
    print(f"Patched pyext for Python 3.11+ : {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
