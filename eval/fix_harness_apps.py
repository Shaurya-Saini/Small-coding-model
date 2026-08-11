#!/usr/bin/env python
"""
fix_harness_apps.py -- patch a bug in bigcode-evaluation-harness's APPS task.

`bigcode_eval/tasks/apps.py::process_results` references a local `level` before
it is assigned:

    def process_results(self, generations, references):
        code_metric = load("codeparrot/apps_metric")
        if level is None:                 # <-- UnboundLocalError at scoring time
            level = self.DATASET_NAME
        results = code_metric.compute(
            predictions=generations, k_list=self.k_list, level=self.DATASET_NAME
        )

The `compute(...)` call already passes `level=self.DATASET_NAME`, so the two-line
`if level is None:` block is dead code that only crashes. We delete it. Idempotent:
once removed, re-running is a no-op.

Usage:
    python eval/fix_harness_apps.py --apps-file <harness>/bigcode_eval/tasks/apps.py
"""
from __future__ import annotations

import argparse
import re
import sys

# `if level is None:\n    level = self.DATASET_NAME` with flexible indentation.
BUGGY = re.compile(
    r"[ \t]*if level is None:[ \t]*\n[ \t]*level[ \t]*=[ \t]*self\.DATASET_NAME[ \t]*\n"
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apps-file", required=True,
                    help="Path to the harness's bigcode_eval/tasks/apps.py")
    args = ap.parse_args()

    try:
        with open(args.apps_file, encoding="utf-8") as fh:
            src = fh.read()
    except FileNotFoundError:
        print(f"apps.py not found: {args.apps_file}")
        return 1

    new, n = BUGGY.subn("", src)
    if n == 0:
        print(f"No buggy `level` block found (already patched?): {args.apps_file}")
        return 0

    with open(args.apps_file, "w", encoding="utf-8") as fh:
        fh.write(new)
    print(f"Patched harness apps.py (removed {n} buggy `level` block): {args.apps_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
