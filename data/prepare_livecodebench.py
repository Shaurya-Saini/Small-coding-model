#!/usr/bin/env python
"""
prepare_livecodebench.py -- Download LiveCodeBench and record a manifest.

============================================================================
 LIVECODEBENCH IS EVALUATION-ONLY. IT IS NEVER FED INTO TRAINING. EVER.
 (See CLAUDE.md rule #1 and SCM.md §2/§5.)
============================================================================

Because of that rule, this script deliberately does NOT emit prompt->solution
pairs, does NOT format anything for SFT, and shares no code with
prepare_apps.py. Its only jobs are:
  1. Confirm the LiveCodeBench data is reachable and cache it.
  2. Write a small manifest (problem ids, difficulty, contest date, counts) so
     we can reason about the difficulty distribution and record exactly which
     release version we evaluate on -- reproducibility, not training.

The actual evaluation is run later by LiveCodeBench's own `lcb_runner`, which
loads the dataset itself; this script does not produce anything the evaluator
consumes. It is a sanity/caching/record-keeping step.

Output (default): data/livecodebench/manifest.json

Usage:
    python prepare_livecodebench.py --version release_v5
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter

# The "lite" code-generation config is the standard one used for eval and is
# lighter to download. Versions are exposed via the `version_tag` config.
LCB_DATASET = "livecodebench/code_generation_lite"

# NOTE: pick the LATEST stable release and RECORD it in PROGRESS.md -- the score
# is only comparable against leaderboard numbers for the *same* version. Verify
# the newest available tag when you actually run this (release_v6, ... may exist).
DEFAULT_VERSION = "release_v5"


def load_lcb_test(version: str):
    """Load LiveCodeBench WITHOUT its dataset script.

    Same situation as APPS: `datasets` >= 4.0 dropped loading scripts, and
    livecodebench/code_generation_lite is script-based, so we read the Hub's
    auto-generated Parquet export instead. LiveCodeBench exposes each release as
    a config (release_v1 ... release_vN), so we select the config matching
    `version`.
    """
    from huggingface_hub import HfApi
    from datasets import load_dataset

    rev = "refs/convert/parquet"
    files = [f for f in HfApi().list_repo_files(LCB_DATASET, repo_type="dataset",
                                                revision=rev) if f.endswith(".parquet")]
    if not files:
        raise RuntimeError(f"No Parquet export found for {LCB_DATASET}@{rev}. "
                           f"lcb_runner loads the data natively during eval; this "
                           f"manifest step is optional -- you can skip it.")

    configs = sorted({f.split("/")[0] for f in files})
    if version not in configs:
        raise RuntimeError(
            f"Version config '{version}' not in the Parquet export. "
            f"Available: {configs}. Pick one of these (and record it in PROGRESS.md).")

    sel = [f for f in files if f.startswith(version + "/")]
    data_files = [f"hf://datasets/{LCB_DATASET}@{rev}/{f}" for f in sel]
    return load_dataset("parquet", data_files=data_files, split="train")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", default=DEFAULT_VERSION,
                        help=f"LiveCodeBench version_tag (default: {DEFAULT_VERSION}).")
    parser.add_argument("--output-dir", default=os.path.join("data", "livecodebench"),
                        help="Where to write the manifest (default: data/livecodebench).")
    args = parser.parse_args()

    print("=" * 70)
    print(" LiveCodeBench -- EVALUATION ONLY. This data is never used for training.")
    print("=" * 70)
    print(f"Loading {LCB_DATASET} version='{args.version}' (via Parquet export) ...")
    ds = load_lcb_test(args.version)
    print(f"  {len(ds)} problems.")

    # Difficulty distribution (LiveCodeBench already labels easy/medium/hard).
    difficulties = [str(ex.get("difficulty", "unknown")).lower() for ex in ds]
    tier_counts = Counter(difficulties)

    # Manifest holds only metadata -- NO problem statements, NO solutions.
    # (Even though this is eval-only, we keep the manifest content-free so there
    #  is zero chance of this file being mistaken for a training corpus.)
    records = []
    for ex in ds:
        records.append({
            "question_id": ex.get("question_id") or ex.get("problem_id"),
            "difficulty": str(ex.get("difficulty", "unknown")).lower(),
            "contest_date": str(ex.get("contest_date", "")),
            "platform": ex.get("platform", ""),
        })

    os.makedirs(args.output_dir, exist_ok=True)
    manifest_path = os.path.join(args.output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump({
            "dataset": LCB_DATASET,
            "version_tag": args.version,
            "n_problems": len(ds),
            "difficulty_counts": dict(tier_counts),
            "problems": records,
            "_warning": "EVALUATION ONLY -- never use for training.",
        }, fh, indent=2)

    print("\nDone.")
    print(f"  difficulty counts : {dict(tier_counts)}")
    print(f"  manifest written  : {manifest_path}")
    print(f"  RECORD THIS in PROGRESS.md -> evaluated version_tag = {args.version}")


if __name__ == "__main__":
    main()
