#!/usr/bin/env python
"""
prepare_apps.py -- Download the APPS dataset and format the TRAIN split into
prompt->solution pairs for QLoRA supervised fine-tuning.

NON-NEGOTIABLE RULES (see CLAUDE.md):
  * TRAIN split only is written here. The APPS TEST split is NEVER written to a
    training file -- it is left untouched for the internal held-out eval, which
    bigcode-evaluation-harness runs directly against the APPS test split.
  * This script must never import or touch LiveCodeBench data. Keeping the two
    physically separate (different scripts, different variables, different
    output files) is the single most important correctness invariant here.

Output (default): data/apps_train.jsonl
  One JSON object per training example, with fields:
    problem_id : original APPS problem id
    difficulty : normalized tier -- easy | medium | hard
    io_format  : "call_based" | "stdin_stdout"
    prompt     : instruction + question (+ starter code + IO-format hint)
    solution   : one correct Python solution

The training script (training/train_qlora.py) is responsible for wrapping
{prompt, solution} in the model's chat template. Keeping that step with the
tokenizer avoids baking a chat format into the data file.

Usage:
    python prepare_apps.py                         # full train split
    python prepare_apps.py --max-samples 50        # quick check
    python prepare_apps.py --max-solutions-per-problem 3
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter

# APPS on the Hugging Face Hub. Splits: "train" (5000) and "test" (5000).
APPS_DATASET = "codeparrot/apps"

# APPS difficulty labels -> our normalized tiers (used for stratified eval).
DIFFICULTY_MAP = {
    "introductory": "easy",
    "interview": "medium",
    "competition": "hard",
}


def load_apps_split(split: str):
    """Load a split of codeparrot/apps WITHOUT its dataset script.

    `datasets` >= 4.0 removed support for dataset loading scripts (and the
    `trust_remote_code` arg), and codeparrot/apps is script-based (apps.py), so
    the old `load_dataset("codeparrot/apps", trust_remote_code=True)` now fails
    with "Dataset scripts are no longer supported". We instead read the Hub's
    auto-generated Parquet export (the `refs/convert/parquet` branch) directly --
    same data, no script, and forward-compatible.
    """
    from huggingface_hub import HfApi
    from datasets import load_dataset

    rev = "refs/convert/parquet"
    files = [f for f in HfApi().list_repo_files(APPS_DATASET, repo_type="dataset",
                                                revision=rev) if f.endswith(".parquet")]
    if not files:
        raise RuntimeError(f"No Parquet export found for {APPS_DATASET}@{rev}.")

    # codeparrot/apps exposes configs: all | introductory | interview | competition.
    # "all" is the full set (with a `difficulty` column) -- exactly what we want.
    configs = sorted({f.split("/")[0] for f in files})
    config = "all" if "all" in configs else ("default" if "default" in configs else configs[0])

    sel = [f for f in files if f.startswith(config + "/")
           and (f"/{split}/" in f or f"/{split}-" in f or f.endswith(f"/{split}.parquet"))]
    if not sel:
        raise RuntimeError(
            f"Could not find split '{split}' in config '{config}'. "
            f"configs={configs}; sample files={files[:6]}")

    data_files = [f"hf://datasets/{APPS_DATASET}@{rev}/{f}" for f in sel]
    # data_files without split keys load under the 'train' key regardless of the
    # underlying split name -- so we always request split="train" here.
    return load_dataset("parquet", data_files=data_files, split="train")


def _safe_json(raw, default):
    """APPS stores solutions / input_output as JSON strings; some are empty or
    malformed. Never let one bad row crash the whole run.

    `parse_int=str` keeps integer literals as strings instead of converting them
    to Python ints. Some APPS test cases embed enormous integers (thousands of
    digits), which trips Python 3.11+'s int-string-length guard
    (`ValueError: Exceeds the limit (4300 digits) ...`). We only read `fn_name`
    from this structure, so we never need the values as ints anyway."""
    if not raw:
        return default
    try:
        return json.loads(raw, parse_int=str)
    except (json.JSONDecodeError, TypeError, ValueError):
        return default


def detect_io_format(example) -> str:
    """A non-empty `fn_name` in input_output means the problem is graded by
    calling a function (call-based); otherwise it's stdin/stdout."""
    io = _safe_json(example.get("input_output"), {})
    return "call_based" if isinstance(io, dict) and io.get("fn_name") else "stdin_stdout"


def build_prompt(example, io_format: str) -> str:
    """APPS-style QUESTION/ANSWER prompt with an IO-format hint. This mirrors how
    bigcode-evaluation-harness presents APPS at eval time, so the model sees a
    consistent format during training and evaluation."""
    question = example.get("question", "") or ""
    starter = example.get("starter_code", "") or ""
    starter_block = f"\n{starter}\n" if starter.strip() else "\n"
    io_hint = "Use Call-Based format\n" if io_format == "call_based" else "Use Standard Input format\n"
    return f"QUESTION:\n{question}\n{starter_block}{io_hint}ANSWER:\n"


def select_solutions(example, max_solutions: int) -> list[str]:
    """Pick up to `max_solutions` correct solutions per problem, shortest first
    (shorter competitive solutions are usually cleaner and tokenize smaller)."""
    sols = _safe_json(example.get("solutions"), [])
    if not isinstance(sols, list):
        return []
    sols = [s for s in sols if isinstance(s, str) and s.strip()]
    sols.sort(key=len)
    return sols[:max_solutions]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", default=os.path.join("data", "apps_train.jsonl"),
                        help="Output JSONL path (default: data/apps_train.jsonl).")
    parser.add_argument("--max-solutions-per-problem", type=int, default=1,
                        help="How many solutions to emit per problem (default: 1).")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Cap the number of source problems (for quick tests).")
    parser.add_argument("--max-prompt-chars", type=int, default=12000,
                        help="Skip problems whose prompt exceeds this length "
                             "(a few APPS questions are enormous).")
    parser.add_argument("--max-solution-chars", type=int, default=12000,
                        help="Skip solutions longer than this.")
    args = parser.parse_args()

    # ---- TRAIN SPLIT ONLY. This is the guardrail; do not change to "test". ----
    split = "train"
    print(f"Loading {APPS_DATASET} split='{split}' (via Parquet export) ...")
    ds = load_apps_split(split)
    if args.max_samples is not None:
        ds = ds.select(range(min(args.max_samples, len(ds))))
    print(f"  {len(ds)} source problems.")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    n_written = 0
    n_skipped_no_sol = 0
    n_skipped_len = 0
    tier_counts: Counter = Counter()

    with open(args.output, "w", encoding="utf-8") as fh:
        for ex in ds:
            difficulty = DIFFICULTY_MAP.get(ex.get("difficulty", ""), "unknown")
            io_format = detect_io_format(ex)
            prompt = build_prompt(ex, io_format)

            if len(prompt) > args.max_prompt_chars:
                n_skipped_len += 1
                continue

            solutions = select_solutions(ex, args.max_solutions_per_problem)
            if not solutions:
                n_skipped_no_sol += 1
                continue

            wrote_for_problem = False
            for sol in solutions:
                if len(sol) > args.max_solution_chars:
                    continue
                record = {
                    "problem_id": ex.get("problem_id"),
                    "difficulty": difficulty,
                    "io_format": io_format,
                    "prompt": prompt,
                    "solution": sol,
                }
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                n_written += 1
                wrote_for_problem = True
            if wrote_for_problem:
                tier_counts[difficulty] += 1
            else:
                n_skipped_len += 1

    print("\nDone.")
    print(f"  examples written      : {n_written}  -> {args.output}")
    print(f"  problems by tier      : {dict(tier_counts)}")
    print(f"  skipped (no solution) : {n_skipped_no_sol}")
    print(f"  skipped (too long)    : {n_skipped_len}")
    print("\nReminder: the APPS *test* split was NOT touched. It is reserved for "
          "the internal held-out eval (bigcode-evaluation-harness).")


if __name__ == "__main__":
    main()
