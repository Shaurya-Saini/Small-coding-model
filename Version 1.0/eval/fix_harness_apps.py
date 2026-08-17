#!/usr/bin/env python
"""
fix_harness_apps.py -- patch two problems in bigcode-evaluation-harness's APPS task
(bigcode_eval/tasks/apps.py). Both are applied idempotently.

1) BUG: `process_results` references a local `level` before assignment:
       if level is None:                 # <-- UnboundLocalError at scoring time
           level = self.DATASET_NAME
   `compute(...)` already passes `level=self.DATASET_NAME`, so this dead block only
   crashes. We delete it.

2) PROMPT + OUTPUT (in-distribution eval): the model was fine-tuned INSIDE Qwen's
   chat template with the exact QUESTION/ANSWER text from data/prepare_apps.py, but
   the harness feeds a bare `QUESTION:/ANSWER:` string with no chat structure. That
   drives the model off-distribution -> systematically broken code (e.g. an extra
   `)` on nearly every line) -> ~all problems fail to compile. We override:
     * `get_prompt` -> rebuild the training-time prompt: chat template + system
       message + `<|im_start|>assistant` cue, same QUESTION/ANSWER body.
     * `postprocess_generation` -> take the assistant turn (or fall back to the
       ANSWER split), cut the `<|im_end|>`/`<|endoftext|>` markers, and extract a
       markdown code fence if present (the base instruct model tends to add one).
   Overriding on `GeneralAPPS` covers `APPS` (which subclasses it). The base model
   uses the same chat template, so the before/after comparison stays fair.

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

OVERRIDE_MARKER = "# --- SCM prompt+postprocess override ---"
# Appended to the module AFTER the classes are defined. A raw string (r'''...''')
# so the `\n`, `\s` inside stay as literal backslash-escapes in the written file.
OVERRIDE = r'''

# --- SCM prompt+postprocess override ---
# The fine-tuned model was trained INSIDE Qwen's chat template with the exact
# QUESTION/ANSWER text from data/prepare_apps.py. The harness feeds a bare
# QUESTION:/ANSWER: string with no chat structure, which drives the model
# off-distribution (systematic broken output). We rebuild the training-time prompt
# (chat template + system msg + assistant cue), and extract the assistant turn back
# out. The base instruct model uses the same template, so the comparison stays fair.
import re as _scm_re
import json as _scm_json

_SCM_SYS = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."


def _scm_get_prompt(self, doc):
    question = doc.get("question", "") or ""
    starter = doc.get("starter_code", "") or ""
    fn_name = None
    io_raw = doc.get("input_output")
    if io_raw:
        try:
            _io = _scm_json.loads(io_raw, parse_int=str)
            if isinstance(_io, dict):
                fn_name = _io.get("fn_name")
        except Exception:
            pass
    starter_block = ("\n" + starter + "\n") if starter.strip() else "\n"
    io_hint = "Use Call-Based format\n" if fn_name else "Use Standard Input format\n"
    body = "QUESTION:\n" + question + "\n" + starter_block + io_hint + "ANSWER:\n"
    return ("<|im_start|>system\n" + _SCM_SYS + "<|im_end|>\n"
            "<|im_start|>user\n" + body + "<|im_end|>\n<|im_start|>assistant\n")


def _scm_postprocess_generation(self, generation, idx):
    # Prefer the assistant turn (chat-templated prompt); fall back to ANSWER split.
    if "<|im_start|>assistant" in generation:
        generation = generation.split("<|im_start|>assistant")[-1]
        if generation.startswith("\n"):
            generation = generation[1:]
    else:
        try:
            generation = generation.split("\nANSWER:", 1)[1]
        except IndexError:
            pass
    for _tok in ("<|im_end|>", "<|endoftext|>"):
        generation = generation.split(_tok)[0]
    _fence = _scm_re.search(r"```(?:python|py)?\s*\n?(.*?)```", generation, _scm_re.DOTALL)
    if _fence:
        generation = _fence.group(1)
    return generation


GeneralAPPS.get_prompt = _scm_get_prompt
GeneralAPPS.postprocess_generation = _scm_postprocess_generation
# --- end SCM prompt+postprocess override ---
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

    original = src

    # Remove any previously-appended SCM override block, so re-patching (e.g. after
    # we revise the override) replaces it cleanly instead of stacking duplicates.
    src = re.split(r"\n+# --- SCM (?:prompt\+)?postprocess override ---",
                   src, maxsplit=1)[0]

    # Remove the upstream buggy `level` block from process_results (if still there).
    src, _ = BUGGY.subn("", src)

    # Append the current override.
    src = src.rstrip("\n") + "\n" + OVERRIDE

    if src == original:
        print(f"Already patched (no changes): {args.apps_file}")
        return 0

    with open(args.apps_file, "w", encoding="utf-8") as fh:
        fh.write(src)
    print(f"Patched harness apps.py (chat-template prompt + postprocess override, "
          f"level-bug removed): {args.apps_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
