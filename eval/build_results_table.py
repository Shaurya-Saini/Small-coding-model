#!/usr/bin/env python
"""
build_results_table.py -- assemble the difficulty-stratified pass@k table
(SCM.md §7) from a single normalized scores file, and write results/report.md.

Why a normalized file: bigcode-evaluation-harness and lcb_runner emit different
JSON shapes that also drift across versions. Rather than parse both here, we read
one hand-normalized `results/scores.json` (copy results/scores.template.json and
fill it from the metrics JSONs the eval scripts produce). This keeps the table
builder stable and makes exactly which numbers we publish auditable -- and every
number in it must come from our own eval runs (frontier cells are marked cited).

Usage:
    python build_results_table.py                       # reads results/scores.json
    python build_results_table.py --scores path.json --out results/report.md
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

TIER_ORDER = ["easy", "medium", "hard"]
TIER_LABELS = {
    "easy": "Easy / Introductory",
    "medium": "Medium / Interview",
    "hard": "Hard / Competition",
}
BENCH_LABELS = {"apps": "APPS (held-out test)", "livecodebench": "LiveCodeBench"}


def fmt_cell(entries: list[dict]) -> str:
    """entries: list of {k, pass_at_k, source}. -> 'pass@1 12.3 · pass@10 25.0'."""
    if not entries:
        return "—"
    entries = sorted(entries, key=lambda e: e.get("k", 1))
    parts = []
    cited = False
    for e in entries:
        val = e.get("pass_at_k")
        val_s = f"{val:.1f}" if isinstance(val, (int, float)) else str(val)
        parts.append(f"pass@{e.get('k', 1)} {val_s}")
        cited = cited or (e.get("source") == "cited")
    text = " · ".join(parts)
    return text + " *(cited)*" if cited else text


def build_table(records, columns, benchmark) -> list[str]:
    # cell[(tier, model_key)] -> list of entries
    cell = defaultdict(list)
    n_problems = {}
    for r in records:
        if r.get("benchmark") != benchmark:
            continue
        tier = r.get("tier")
        cell[(tier, r.get("model"))].append(r)
        if r.get("n_problems") is not None:
            n_problems[tier] = r["n_problems"]

    tiers = [t for t in TIER_ORDER if any(k[0] == t for k in cell)]
    if not tiers:
        return []

    header = "| Difficulty | " + " | ".join(c["label"] for c in columns) + " |"
    sep = "|" + "---|" * (len(columns) + 1)
    lines = [f"### {BENCH_LABELS.get(benchmark, benchmark)}", "", header, sep]
    for tier in tiers:
        row = [TIER_LABELS.get(tier, tier)]
        for c in columns:
            row.append(fmt_cell(cell.get((tier, c["key"]), [])))
        lines.append("| " + " | ".join(row) + " |")
    # footnote with n_problems per tier if present
    if n_problems:
        counts = ", ".join(f"{TIER_LABELS.get(t, t)}: {n}" for t, n in n_problems.items())
        lines += ["", f"_Problems per tier — {counts}._"]
    lines.append("")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", default=os.path.join("results", "scores.json"))
    ap.add_argument("--out", default=os.path.join("results", "report.md"))
    args = ap.parse_args()

    if not os.path.exists(args.scores):
        raise SystemExit(f"{args.scores} not found. Copy results/scores.template.json "
                         f"to results/scores.json and fill it from your eval runs.")

    with open(args.scores, encoding="utf-8") as fh:
        data = json.load(fh)

    columns = data["model_columns"]
    records = data["records"]
    benchmarks = []
    for r in records:
        if r["benchmark"] not in benchmarks:
            benchmarks.append(r["benchmark"])

    out = ["# Results — difficulty-stratified pass@k", ""]
    if data.get("notes"):
        out += [data["notes"], ""]
    for bench in benchmarks:
        out += build_table(records, columns, bench)

    out += [
        "---",
        "",
        "**Honesty notes (SCM.md §7, §10):**",
        "- Only cells sourced from our own eval pipeline are unmarked; *(cited)* "
        "cells are published leaderboard numbers for the same split.",
        "- The hard/competition-tier gap is reported as-is, not downplayed.",
        "- pass@k settings (k, samples, temperature) are recorded per run in the "
        "`results/apps/` and `results/livecodebench/` metrics files.",
        "",
    ]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
