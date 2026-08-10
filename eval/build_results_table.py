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


# --- Diagram generation ------------------------------------------------------
# Categorical palette slots 1..6 from the data-viz reference palette (light
# surface). Validated CVD-safe as a set (worst adjacent ΔE 21.6, well over the
# >=12 target). Slots 2/3 sit below 3:1 contrast on the light surface, so the
# "relief rule" applies -- we draw a value label on every bar, which also just
# makes the chart easier to read.
CATEGORICAL = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948"]
INK, SECONDARY, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"


def _primary_value(entries: list[dict]):
    """The single pass@k value a bar shows: prefer pass@1, else the smallest k."""
    if not entries:
        return None
    picks = [e for e in entries if e.get("k") == 1] or sorted(
        entries, key=lambda e: e.get("k", 1))
    val = picks[0].get("pass_at_k")
    return val if isinstance(val, (int, float)) else None


def render_charts(records, columns, benchmarks, figures_dir):
    """One grouped bar chart per benchmark: pass@1 by difficulty tier, one bar
    per model column. Returns {benchmark: png_path}. Degrades gracefully (and
    non-fatally) if matplotlib isn't installed -- the table still builds."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless: works on Kaggle/Colab with no display
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001 - charts are optional, never fatal
        print(f"matplotlib unavailable ({e}); skipping charts "
              f"(`pip install matplotlib` to enable).")
        return {}

    os.makedirs(figures_dir, exist_ok=True)
    made: dict[str, str] = {}

    for bench in benchmarks:
        recs = [r for r in records if r.get("benchmark") == bench]
        tiers = [t for t in TIER_ORDER if any(r.get("tier") == t for r in recs)]
        cols = [c for c in columns if any(r.get("model") == c["key"] for r in recs)]
        if not tiers or not cols:
            continue

        cell = defaultdict(list)
        for r in recs:
            cell[(r.get("tier"), r.get("model"))].append(r)

        n = len(cols)
        group_w = 0.8
        bar_w = group_w / n
        x = list(range(len(tiers)))

        fig, ax = plt.subplots(figsize=(1.9 * len(tiers) + 2.4, 4.4), dpi=200)
        fig.patch.set_facecolor(SURFACE)
        ax.set_facecolor(SURFACE)

        top = 1.0
        for i, c in enumerate(cols):
            centers = [xi - group_w / 2 + bar_w * (i + 0.5) for xi in x]
            vals = [_primary_value(cell.get((t, c["key"]), [])) for t in tiers]
            heights = [v if v is not None else 0 for v in vals]
            top = max([top] + [h for h in heights])
            bars = ax.bar(centers, heights, width=bar_w * 0.86,
                          color=CATEGORICAL[i % len(CATEGORICAL)],
                          label=c["label"], zorder=3)
            for rect, v in zip(bars, vals):
                if v is None:
                    continue
                ax.annotate(f"{v:.1f}",
                            (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=8, color=INK)

        ax.set_xticks(x)
        ax.set_xticklabels([TIER_LABELS.get(t, t) for t in tiers],
                           fontsize=9, color=INK)
        ax.set_ylabel("pass@1 (%)", fontsize=9, color=SECONDARY)
        ax.set_ylim(0, top * 1.18)
        ax.set_title(f"{BENCH_LABELS.get(bench, bench)} — pass@1 by difficulty",
                     fontsize=11, color=INK, pad=10)

        # Recessive chrome: drop the box, hairline y-grid behind the bars.
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(BASELINE)
        ax.tick_params(colors=MUTED, length=0)
        ax.yaxis.grid(True, color=GRID, linewidth=1, zorder=0)
        ax.set_axisbelow(True)
        # Legend always present for >=2 series (identity never color-alone).
        ax.legend(frameon=False, fontsize=8.5, ncol=min(n, 3),
                  loc="upper center", bbox_to_anchor=(0.5, -0.12), labelcolor=INK)

        path = os.path.join(figures_dir, f"{bench}_pass_at_1.png")
        fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        made[bench] = path
        print(f"Wrote {path}")

    return made


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", default=os.path.join("results", "scores.json"))
    ap.add_argument("--out", default=os.path.join("results", "report.md"))
    ap.add_argument("--figures-dir", default=os.path.join("results", "figures"),
                    help="Where the generated chart PNGs are written.")
    ap.add_argument("--no-charts", action="store_true",
                    help="Skip diagram generation (table only).")
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

    charts = {} if args.no_charts else render_charts(
        records, columns, benchmarks, args.figures_dir)

    out = ["# Results — difficulty-stratified pass@k", ""]
    if data.get("notes"):
        out += [data["notes"], ""]
    for bench in benchmarks:
        out += build_table(records, columns, bench)
        if bench in charts:
            rel = os.path.relpath(charts[bench],
                                  os.path.dirname(os.path.abspath(args.out)))
            rel = rel.replace(os.sep, "/")  # forward slashes render everywhere
            out += [f"![{BENCH_LABELS.get(bench, bench)} — pass@1 by difficulty]({rel})", ""]

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
