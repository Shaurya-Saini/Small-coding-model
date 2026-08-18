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
#
# TWO prompt styles, selected at import time by env var SCM_EVAL_STYLE:
#   * "v1" (default) -- the v1/base model: bare-instruct QUESTION/ANSWER body in the
#     Qwen chat template. Correct for Qwen2.5-Coder-7B-Instruct and the v1 APPS
#     fine-tune (both trained/used with this exact text).
#   * "v2" -- the v2 reasoning fine-tune (…-ocr-qlora): rebuilds the OpenCodeReasoning
#     training prompt (INSTRUCTION + question, chat template, default system msg),
#     appends the APPS Standard-Input/Call-Based I/O hint, and on the way OUT strips
#     the <think>…</think> scratchpad and extracts the LAST ```python fence (the
#     final solution after the reasoning). Feeding the v2 model the v1 prompt, or a
#     small token budget, or grabbing the FIRST fence (which may be exploratory code
#     inside <think>) would reproduce v1's train/eval-mismatch failure -- see
#     CLAUDE.md §8. The wrapper raises --max_length_generation for v2 accordingly.
# The base instruct model uses the same chat template in both styles, so the
# before/after comparison stays fair.
OVERRIDE = r'''

# --- SCM prompt+postprocess override ---
import os as _scm_os
import re as _scm_re
import json as _scm_json

_SCM_SYS = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

# MUST stay byte-identical to INSTRUCTION in data/prepare_reasoning_traces.py --
# any drift puts the v2 model off-distribution at eval (the whole v1 bug).
_SCM_V2_INSTRUCTION = (
    "You are an expert competitive programmer. Solve the following problem.\n"
    "Reason step by step inside <think> </think> tags, then give the complete, "
    "correct solution as a single Python code block.\n\n"
)

_SCM_FENCE_RE = r"```(?:python|py)?\s*\n?(.*?)```"


def _scm_io_fn_name(doc):
    io_raw = doc.get("input_output")
    if io_raw:
        try:
            _io = _scm_json.loads(io_raw, parse_int=str)
            if isinstance(_io, dict):
                return _io.get("fn_name")
        except Exception:
            pass
    return None


def _scm_get_prompt(self, doc):
    question = doc.get("question", "") or ""
    starter = doc.get("starter_code", "") or ""
    fn_name = _scm_io_fn_name(doc)
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
    _fence = _scm_re.search(_SCM_FENCE_RE, generation, _scm_re.DOTALL)
    if _fence:
        generation = _fence.group(1)
    return generation


def _scm_v2_get_prompt(self, doc):
    # Reproduce the OpenCodeReasoning training prompt (INSTRUCTION + question) and
    # append the APPS I/O contract so stdin/stdout vs call-based is unambiguous.
    question = doc.get("question", "") or ""
    starter = doc.get("starter_code", "") or ""
    fn_name = _scm_io_fn_name(doc)
    starter_block = ("\n\n" + starter.strip() + "\n") if starter.strip() else ""
    io_hint = ("\nUse Call-Based format.\n" if fn_name
               else "\nUse Standard Input format.\n")
    body = _SCM_V2_INSTRUCTION + question + starter_block + io_hint
    return ("<|im_start|>system\n" + _SCM_SYS + "<|im_end|>\n"
            "<|im_start|>user\n" + body + "<|im_end|>\n<|im_start|>assistant\n")


def _scm_v2_postprocess_generation(self, generation, idx):
    # Assistant turn only, markers stripped.
    if "<|im_start|>assistant" in generation:
        generation = generation.split("<|im_start|>assistant")[-1]
        if generation.startswith("\n"):
            generation = generation[1:]
    for _tok in ("<|im_end|>", "<|endoftext|>"):
        generation = generation.split(_tok)[0]
    # Drop the reasoning scratchpad: keep only what follows the LAST </think>.
    _low = generation.lower()
    if "</think>" in _low:
        generation = generation[_low.rindex("</think>") + len("</think>"):]
    # The final solution is the LAST fence (earlier fences may be exploratory code
    # the model wrote while reasoning). No fence -> return as-is (a genuine miss).
    _fences = _scm_re.findall(_SCM_FENCE_RE, generation, _scm_re.DOTALL)
    if _fences:
        generation = _fences[-1]
    return generation


_SCM_STYLE = _scm_os.environ.get("SCM_EVAL_STYLE", "v1").strip().lower()
if _SCM_STYLE == "v2":
    GeneralAPPS.get_prompt = _scm_v2_get_prompt
    GeneralAPPS.postprocess_generation = _scm_v2_postprocess_generation
else:
    GeneralAPPS.get_prompt = _scm_get_prompt
    GeneralAPPS.postprocess_generation = _scm_postprocess_generation
print("[SCM] APPS prompt/postprocess style =", _SCM_STYLE)
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
