#!/usr/bin/env python
"""
fix_harness_apps.py -- patch two problems in bigcode-evaluation-harness's APPS task
(bigcode_eval/tasks/apps.py). Both are applied idempotently.

1) BUG: `process_results` references a local `level` before assignment:
       if level is None:                 # <-- UnboundLocalError at scoring time
           level = self.DATASET_NAME
   `compute(...)` already passes `level=self.DATASET_NAME`, so this dead block only
   crashes. We delete it.

2) OUTPUT CLEANUP: the harness's default eos is `<|endoftext|>`, but a Qwen chat
   model ends its turn with `<|im_end|>`. That token therefore leaks into the
   generated code, which then fails to compile (SyntaxError) -> every problem
   scores as a "compile error". We override `postprocess_generation` to:
     * keep the original `split("\nANSWER:")` behavior,
     * cut everything from the first `<|im_end|>` / `<|endoftext|>`,
     * if the model wrapped code in a markdown fence (the base instruct model
       tends to), extract the fenced block.
   Overriding on `GeneralAPPS` covers `APPS` (which subclasses it).

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

OVERRIDE_MARKER = "# --- SCM postprocess override ---"
# Appended to the module AFTER the classes are defined. Note: this text is written
# verbatim into apps.py, so backslashes are doubled here to land as single ones.
OVERRIDE = '''

# --- SCM postprocess override ---
# Strip chat end-markers (and extract fenced code) so instruct-model output is
# valid Python before scoring. See eval/fix_harness_apps.py for the why.
import re as _scm_re


def _scm_postprocess_generation(self, generation, idx):
    try:
        generation = generation.split("\\nANSWER:", 1)[1]
    except IndexError:
        pass
    for _tok in ("<|im_end|>", "<|endoftext|>"):
        generation = generation.split(_tok)[0]
    _fence = _scm_re.search(r"```(?:python|py)?\\s*\\n?(.*?)```", generation, _scm_re.DOTALL)
    if _fence:
        generation = _fence.group(1)
    return generation


GeneralAPPS.postprocess_generation = _scm_postprocess_generation
# --- end SCM postprocess override ---
'''


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

    changed = []

    src, n_bug = BUGGY.subn("", src)
    if n_bug:
        changed.append("removed buggy `level` block")

    if OVERRIDE_MARKER not in src:
        src = src.rstrip("\n") + "\n" + OVERRIDE
        changed.append("added postprocess override (strip <|im_end|> / fences)")

    if not changed:
        print(f"Already patched (no changes): {args.apps_file}")
        return 0

    with open(args.apps_file, "w", encoding="utf-8") as fh:
        fh.write(src)
    print(f"Patched harness apps.py [{'; '.join(changed)}]: {args.apps_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
