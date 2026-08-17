#!/usr/bin/env python
"""
diagnose_apps.py -- autopsy of a saved APPS eval run. Answers ONE question:
"When a solution was scored 0, WHY?" -- so we can tell a real model failure apart
from a measurement/scoring artifact before investing in v2.

Motivation (see CLAUDE.md eval-overhaul): v1's *base* model scored 0.0% strict /
0.84% avg on APPS-introductory, yet a static pass showed 146/150 of its
generations compile and 145/150 call print(). A strong 7B writing valid
stdin->stdout solutions to *introductory* problems cannot truly score 0 -- that
smells like a scoring/format bug, not ability. This script confirms it by
RE-EXECUTING the already-saved generations against the real APPS hidden tests and
categorising every failure.

The decisive signal: each solution is scored TWICE --
  * strict   : exact string match (what the harness does)
  * normalized: rstrip each line + drop trailing blank lines, then compare
If many problems fail strict but PASS normalized, the failures are a
formatting/comparison artifact (the fix is in the eval/prompt, not the model).
If they fail both, the model is genuinely wrong.

Runs in the CLEAN eval env on Kaggle (needs `datasets`; no GPU, no model load --
it only reads saved generations + the dataset's test cases). It does NOT
re-generate anything.

  !! SAFETY: this EXECUTES model-generated code in subprocesses (isolated `-I`,
  per-test timeout, no network assumed). Only run it on saved generations you
  trust (your own runs), in a throwaway environment (Kaggle).

Usage (per tier, per label):
    python eval/diagnose_apps.py \
        --generations results/apps/base/apps-introductory_generations_apps-introductory.json \
        --tier introductory --label base --limit 30

    # static only (no execution) -- safe anywhere, no dataset needed:
    python eval/diagnose_apps.py --generations <path> --tier introductory --static-only
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

# APPS test split difficulty tiers -> the value the dataset uses.
TIER_ALIASES = {
    "introductory": "introductory",
    "interview": "interview",
    "competition": "competition",
}

# Outcome categories, ordered from "measurement problem" to "real model problem".
CATS = ["PASS", "PASS_NORMALIZED", "WRONG_OUTPUT", "NO_OUTPUT",
        "RUNTIME_ERROR", "TIMEOUT", "SYNTAX_ERROR", "EMPTY", "CALL_BASED_SKIP"]


def _load_generations(path: str) -> list[str]:
    """Saved generations are a list (per problem) of lists (per sample). We take
    sample 0 (v1 ran n_samples=1). They are already post-processed (extracted
    code), so we use them verbatim."""
    raw = json.load(open(path, encoding="utf-8"))
    out = []
    for cell in raw:
        out.append(cell[0] if isinstance(cell, list) else cell)
    return out


def _load_apps_tests(tier: str, limit: int, limit_start: int):
    """Return the APPS TEST-split rows for a tier, in the SAME order the
    bigcode harness's `apps-<tier>` task uses -- i.e. the difficulty CONFIG
    (introductory-only), NOT the full 5000-problem 'all' config. Getting this
    wrong misaligns generation[i] with problem[i] and makes every score bogus.
    parse_int=str guards the thousands-of-digit integers in APPS I/O that blow up
    Python's int() (same guard as data/prepare_apps.py)."""
    from datasets import load_dataset
    name = TIER_ALIASES[tier]
    try:
        # Canonical: the difficulty config gives only that tier's problems in the
        # harness's order. (This is what `--tasks apps-introductory` iterates.)
        ds = load_dataset("codeparrot/apps", name, split="test",
                          trust_remote_code=True)
    except Exception:
        # Fallback: filter the 'all' config by the difficulty column, preserving
        # original order.
        full = load_dataset("codeparrot/apps", split="test", trust_remote_code=True)
        ds = full.filter(lambda r: r.get("difficulty") == name)
    print(f"  loaded codeparrot/apps [{name}] test: {len(ds)} problems "
          f"(expect ~1000 for introductory, NOT 5000)")
    rows = []
    for i in range(limit_start, min(limit_start + limit, len(ds))):
        row = ds[i]
        io_raw = row.get("input_output") or ""
        try:
            io = json.loads(io_raw, parse_int=str) if io_raw else {}
        except Exception:
            io = {}
        rows.append({
            "question": row.get("question", "") or "",
            "starter_code": row.get("starter_code", "") or "",
            "io": io,
        })
    return rows


def _as_text(x) -> str:
    """APPS inputs/outputs are sometimes a string, sometimes a list of lines."""
    if isinstance(x, list):
        return "\n".join(str(e) for e in x)
    return str(x)


def _normalize(s: str) -> str:
    """rstrip trailing whitespace per line; drop trailing blank lines."""
    lines = [ln.rstrip() for ln in s.replace("\r\n", "\n").split("\n")]
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines)


def _run_one(code: str, stdin_text: str, timeout: float):
    """Execute `code` as a script with stdin piped in. Returns
    (status, stdout, stderr) where status in {ok, runtime, timeout}."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False,
                                     encoding="utf-8") as fh:
        fh.write(code)
        script = fh.name
    try:
        proc = subprocess.run(
            [sys.executable, "-I", script],
            input=stdin_text, capture_output=True, text=True, timeout=timeout,
        )
        if proc.returncode != 0:
            return "runtime", proc.stdout, proc.stderr
        return "ok", proc.stdout, proc.stderr
    except subprocess.TimeoutExpired:
        return "timeout", "", ""
    finally:
        try:
            Path(script).unlink()
        except OSError:
            pass


def _classify(code: str, io: dict, timeout: float, probe: dict | None = None,
              max_tests: int = 0):
    """Score one generation against a problem's hidden tests, strict AND
    normalized. Returns (category, detail). A problem passes only if ALL tests
    pass. If `probe` is passed, the first test's (input, expected, got) is stored
    into it for the --show-io dump. `max_tests`>0 caps tests/problem for speed --
    which OVER-counts passes (a later hidden test might fail), so a capped run
    answers 'is it ~0 or clearly >0?', not the exact final number."""
    if not code.strip():
        return "EMPTY", ""
    try:
        compile(code, "<gen>", "exec")
    except SyntaxError as e:
        return "SYNTAX_ERROR", str(e).splitlines()[0]

    # Call-based problems need a driver (import + call fn_name); we skip exec here
    # and just note them -- introductory is overwhelmingly stdin-based.
    if io.get("fn_name"):
        return "CALL_BASED_SKIP", io["fn_name"]

    inputs = io.get("inputs") or []
    outputs = io.get("outputs") or []
    if not inputs:
        return "NO_OUTPUT", "no test cases in dataset row"
    if max_tests and max_tests > 0:
        inputs, outputs = inputs[:max_tests], outputs[:max_tests]

    strict_all = True
    norm_all = True
    first_detail = ""
    for t, (inp, exp) in enumerate(zip(inputs, outputs)):
        stdin_text = _as_text(inp)
        expected = _as_text(exp)
        status, got, stderr = _run_one(code, stdin_text, timeout)
        if probe is not None and t == 0:
            probe.update(stdin=stdin_text, expected=expected,
                         got=(got if status == "ok" else f"<{status}> {stderr[-200:]}"))
        if status == "timeout":
            return "TIMEOUT", ""
        if status == "runtime":
            last = (stderr.strip().splitlines() or ["?"])[-1]
            return "RUNTIME_ERROR", last[:200]
        if got.strip() == "" and expected.strip() != "":
            return "NO_OUTPUT", ""
        exact = got == expected
        norm = _normalize(got) == _normalize(expected)
        if not exact:
            strict_all = False
            if not first_detail:
                first_detail = (f"expected={_normalize(expected)[:80]!r} "
                                f"got={_normalize(got)[:80]!r}")
        if not norm:
            norm_all = False
    if strict_all:
        return "PASS", ""
    if norm_all:
        return "PASS_NORMALIZED", "format-only mismatch (whitespace)"
    return "WRONG_OUTPUT", first_detail


def _static_report(gens: list[str]) -> None:
    import re
    n = len(gens)
    compiles = reads = prints = 0
    for code in gens:
        if not code.strip():
            continue
        try:
            compile(code, "<g>", "exec"); compiles += 1
        except SyntaxError:
            pass
        if re.search(r"\binput\s*\(|sys\.stdin|stdin\b", code):
            reads += 1
        if re.search(r"\bprint\s*\(", code):
            prints += 1
    print(f"  static: compiles {compiles}/{n} | reads stdin {reads}/{n} | "
          f"calls print() {prints}/{n}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--generations", required=True, help="saved *_generations_*.json")
    ap.add_argument("--tier", required=True, choices=list(TIER_ALIASES))
    ap.add_argument("--label", default="?", help="just for the header, e.g. base")
    ap.add_argument("--limit", type=int, default=30, help="problems to autopsy")
    ap.add_argument("--limit-start", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=6.0, help="per-test seconds")
    ap.add_argument("--max-tests", type=int, default=0,
                    help="cap hidden tests/problem for speed (0=all). Capped runs "
                         "OVER-count passes; use for 'is it ~0 or >0?', not finals.")
    ap.add_argument("--static-only", action="store_true",
                    help="compile/structure only; no execution, no dataset needed")
    ap.add_argument("--show", type=int, default=6,
                    help="how many per-problem detail lines to print")
    ap.add_argument("--show-io", action="store_true",
                    help="for each shown problem, dump question + first test's "
                         "input/expected/got + the code's I/O lines (alignment check)")
    args = ap.parse_args()

    gens = _load_generations(args.generations)
    print(f"=== APPS diagnose :: label={args.label} tier={args.tier} "
          f"n_generations={len(gens)} ===")
    _static_report(gens)

    if args.static_only:
        print("static-only: done (pass --generations without --static-only to "
              "execute against real tests).")
        return 0

    rows = _load_apps_tests(args.tier, args.limit, args.limit_start)
    counts = Counter()
    shown = 0
    for i, row in enumerate(rows):
        code = gens[args.limit_start + i]
        probe: dict = {} if args.show_io else None
        cat, detail = _classify(code, row["io"], args.timeout, probe=probe,
                                max_tests=args.max_tests)
        counts[cat] += 1
        if shown < args.show and cat != "PASS":
            print(f"  [{args.limit_start + i:>3}] {cat:<16} {detail}")
            if args.show_io:
                io = row["io"]
                q = " ".join(row["question"].split())
                io_lines = [ln for ln in code.splitlines()
                            if ("input(" in ln or "stdin" in ln or "print(" in ln)]
                print(f"        Q: {q[:220]}")
                print(f"        fn_name={io.get('fn_name')} "
                      f"n_tests={len(io.get('inputs') or [])}")
                if probe:
                    print(f"        stdin[0]   : {probe.get('stdin','')[:120]!r}")
                    print(f"        expected[0]: {probe.get('expected','')[:120]!r}")
                    print(f"        got[0]     : {probe.get('got','')[:120]!r}")
                print(f"        code I/O   : {' | '.join(l.strip() for l in io_lines[:6])}")
            shown += 1

    print("\n--- outcome histogram (worst-per-problem) ---")
    total = sum(counts.values())
    for cat in CATS:
        if counts[cat]:
            print(f"  {cat:<16} {counts[cat]:>3} / {total}")

    strict_pass = counts["PASS"]
    artifact = counts["PASS_NORMALIZED"]
    print("\n--- verdict ---")
    print(f"  strict pass@1 on this subset : {strict_pass}/{total}")
    print(f"  would-pass if whitespace-normalized: "
          f"{strict_pass + artifact}/{total}  (+{artifact} recovered)")
    if artifact and artifact >= max(1, strict_pass):
        print("  => FORMAT/SCORING ARTIFACT dominates: fix comparison/prompt, "
              "not the model.")
    elif counts["SYNTAX_ERROR"] >= total * 0.5:
        print("  => REAL model failure: majority don't even compile "
              "(e.g. the golfed-code bracket artifact).")
    else:
        print("  => Mixed; read the per-problem details above.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
