#!/usr/bin/env python
"""
score_apps.py -- the v2 APPS SCORER. Replaces bigcode-evaluation-harness's APPS
scoring, which in v1 reported 0.0% on introductory even though the base model's
saved generations are ~12% correct (proved by eval/diagnose_apps.py). Generation
was never the problem; the harness's SCORING/orchestration was. So we keep the
saved generations and re-score them ourselves, aligned and verified.

What it does, per (label, tier):
  * loads the SAVED post-processed generations (no GPU, no re-generation),
  * loads APPS via the difficulty CONFIG so generation[i] lines up with
    problem[i] (the alignment bug that made v1's numbers bogus -- see
    diagnose_apps.py),
  * executes each solution against the hidden tests in an isolated subprocess
    (stdin/stdout AND call-based `fn_name` problems), with a per-test timeout,
  * reports the SAME two metrics as the report: strict pass@1 (all tests pass)
    and avg test-case pass rate (partial credit),
  * writes per-tier `*_metrics.rescored.json` (audit) and a consolidated
    `results/scores.rescored.json` that `build_results_table.py` renders directly.

Comparison is whitespace-normalized (rstrip lines, drop trailing blank lines) --
the fair/standard APPS convention.

  !! SAFETY: EXECUTES model-generated code. Run only on your own saved
  generations, in a throwaway env (Kaggle). Needs `datasets`; no GPU/model load.

VALIDATION GATE: with `--max-tests 25`, base/introductory should reproduce
~12% strict (matches diagnose_apps.py). If it doesn't, stop and investigate the
scorer before trusting interview/competition.

Usage (re-score both models, all tiers, from the saved generations):
    python eval/score_apps.py --results-dir results/apps \
        --labels base,finetuned --max-tests 50
    python eval/build_results_table.py --scores results/scores.rescored.json \
        --out results/report.rescored.md
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

TIERS = ["introductory", "interview", "competition"]
# report/scores.json uses easy/medium/hard; APPS tasks use these names.
TIER_TO_BUCKET = {"introductory": "easy", "interview": "medium",
                  "competition": "hard"}


def _normalize(s: str) -> str:
    lines = [ln.rstrip() for ln in s.replace("\r\n", "\n").split("\n")]
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines)


def _as_text(x) -> str:
    return "\n".join(str(e) for e in x) if isinstance(x, list) else str(x)


def _find_generations(results_dir: str, label: str, tier: str) -> str | None:
    # Harness saved e.g. apps-introductory_generations_apps-introductory.json.
    hits = glob.glob(os.path.join(results_dir, label,
                                  f"apps-{tier}_generations*.json"))
    return sorted(hits)[0] if hits else None


def _load_generations(path: str) -> list[str]:
    raw = json.load(open(path, encoding="utf-8"))
    return [c[0] if isinstance(c, list) else c for c in raw]


def _load_apps(tier: str):
    """Difficulty CONFIG (aligned order the harness generated in)."""
    from datasets import load_dataset
    try:
        ds = load_dataset("codeparrot/apps", tier, split="test",
                          trust_remote_code=True)
    except Exception:
        full = load_dataset("codeparrot/apps", split="test", trust_remote_code=True)
        ds = full.filter(lambda r: r.get("difficulty") == tier)
    rows = []
    for r in ds:
        io_raw = r.get("input_output") or ""
        try:
            io = json.loads(io_raw, parse_int=str) if io_raw else {}
        except Exception:
            io = {}
        rows.append(io)
    return rows


def _run_stdin(code: str, stdin_text: str, expected: str, timeout: float) -> bool:
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False,
                                     encoding="utf-8") as fh:
        fh.write(code)
        script = fh.name
    try:
        p = subprocess.run([sys.executable, "-I", script], input=stdin_text,
                           capture_output=True, text=True, timeout=timeout)
        if p.returncode != 0:
            return False
        return _normalize(p.stdout) == _normalize(expected)
    except (subprocess.TimeoutExpired, Exception):  # noqa: BLE001
        return False
    finally:
        try:
            Path(script).unlink()
        except OSError:
            pass


_CALL_DRIVER = """
import json, sys
{code}

def _scm_run():
    args = json.loads(sys.stdin.read())
    fn = globals().get({fn!r})
    if fn is None:
        try:
            fn = getattr(Solution(), {fn!r})   # some APPS call-based use a class
        except Exception:
            sys.exit(7)
    res = fn(*args)
    sys.stdout.write(json.dumps(res, default=str))

_scm_run()
"""


def _run_callbased(code: str, fn: str, args, expected, timeout: float) -> bool:
    driver = _CALL_DRIVER.format(code=code, fn=fn)
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False,
                                     encoding="utf-8") as fh:
        fh.write(driver)
        script = fh.name
    try:
        p = subprocess.run([sys.executable, "-I", script],
                           input=json.dumps(args, default=str),
                           capture_output=True, text=True, timeout=timeout)
        if p.returncode != 0:
            return False
        try:
            got = json.loads(p.stdout)
        except Exception:
            got = p.stdout
        # APPS call-based expected is the return value, sometimes wrapped in [ ].
        exp_variants = [expected]
        if isinstance(expected, list) and len(expected) == 1:
            exp_variants.append(expected[0])
        for exp in exp_variants:
            if got == exp:
                return True
            if _normalize(json.dumps(got, default=str)) == \
               _normalize(json.dumps(exp, default=str)):
                return True
        return False
    except (subprocess.TimeoutExpired, Exception):  # noqa: BLE001
        return False
    finally:
        try:
            Path(script).unlink()
        except OSError:
            pass


def _score_problem(code: str, io: dict, timeout: float, max_tests: int):
    """Return (strict_pass, avg_frac, n_tests_run)."""
    if not code.strip():
        return False, 0.0, 0
    try:
        compile(code, "<g>", "exec")
    except SyntaxError:
        return False, 0.0, 0
    inputs = io.get("inputs") or []
    outputs = io.get("outputs") or []
    if not inputs:
        return False, 0.0, 0
    fn = io.get("fn_name")
    if max_tests and max_tests > 0:
        inputs, outputs = inputs[:max_tests], outputs[:max_tests]
    total = len(inputs)
    passed = 0
    for inp, exp in zip(inputs, outputs):
        if fn:
            ok = _run_callbased(code, fn, inp, exp, timeout)
        else:
            ok = _run_stdin(code, _as_text(inp), _as_text(exp), timeout)
        if ok:
            passed += 1
    return (passed == total), (passed / total if total else 0.0), total


def _score_one(task):
    """Top-level worker (must be importable for ProcessPoolExecutor)."""
    i, code, io, timeout, max_tests = task
    strict, avg, _ = _score_problem(code, io, timeout, max_tests)
    return i, strict, avg, bool(io.get("fn_name"))


def _score_tier(gens: list[str], rows: list[dict], timeout: float,
                max_tests: int, limit: int, workers: int, tag: str = ""):
    import concurrent.futures as cf
    n = min(len(gens), len(rows), limit) if limit else min(len(gens), len(rows))
    tasks = [(i, gens[i], rows[i], timeout, max_tests) for i in range(n)]
    strict_hits = 0
    avg_sum = 0.0
    callbased = 0
    done = 0
    # Parallel across CPU cores; each worker still runs its own per-test
    # subprocesses. This is the big speedup vs the single-threaded v1 of this file.
    with cf.ProcessPoolExecutor(max_workers=workers) as ex:
        for _i, strict, avg, cb in ex.map(_score_one, tasks, chunksize=1):
            strict_hits += 1 if strict else 0
            avg_sum += avg
            callbased += 1 if cb else 0
            done += 1
            if done % 25 == 0 or done == n:
                print(f"    [{tag}] {done}/{n} scored "
                      f"(running strict={100*strict_hits/done:.1f}%)", flush=True)
    return {
        "n": n,
        "strict_accuracy": strict_hits / n if n else 0.0,
        "avg_accuracy": avg_sum / n if n else 0.0,
        "callbased": callbased,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="results/apps")
    ap.add_argument("--labels", default="base,finetuned")
    ap.add_argument("--tiers", default=",".join(TIERS))
    ap.add_argument("--limit", type=int, default=0,
                    help="score only the first N problems/tier (0=all saved gens)")
    ap.add_argument("--max-tests", type=int, default=50,
                    help="cap hidden tests/problem for speed (0=all). Capped runs "
                         "slightly OVER-count. 25 reproduces the diagnose_apps gate.")
    ap.add_argument("--timeout", type=float, default=4.0,
                    help="per-test seconds; looping wrong solutions burn this, so "
                         "keep it modest")
    ap.add_argument("--workers", type=int, default=0,
                    help="parallel worker processes (0 = os.cpu_count())")
    ap.add_argument("--scores-out", default="results/scores.rescored.json")
    args = ap.parse_args()

    workers = args.workers or (os.cpu_count() or 4)

    labels = [x.strip() for x in args.labels.split(",") if x.strip()]
    tiers = [x.strip() for x in args.tiers.split(",") if x.strip()]

    # cache the dataset per tier (shared across labels)
    tier_rows = {t: _load_apps(t) for t in tiers}

    records = []
    print(f"\n{'label':<10} {'tier':<13} {'strict pass@1':<14} "
          f"{'avg test-rate':<14} {'n':>4} {'call-based':>10}")
    print("-" * 70)
    for label in labels:
        for tier in tiers:
            gpath = _find_generations(args.results_dir, label, tier)
            if not gpath:
                print(f"{label:<10} {tier:<13} (no generations file found)")
                continue
            gens = _load_generations(gpath)
            res = _score_tier(gens, tier_rows[tier], args.timeout,
                              args.max_tests, args.limit, workers,
                              tag=f"{label}/{tier}")
            strict_pct = 100 * res["strict_accuracy"]
            avg_pct = 100 * res["avg_accuracy"]
            print(f"{label:<10} {tier:<13} {strict_pct:>6.2f}%        "
                  f"{avg_pct:>6.2f}%        {res['n']:>4} {res['callbased']:>10}")

            # per-tier audit file (keep v1 *_metrics.json untouched for the record)
            audit = {
                f"apps-{tier}": {
                    "avg_accuracy": res["avg_accuracy"],
                    "strict_accuracy": res["strict_accuracy"],
                    "pass_at_k": None,
                },
                "config": {"model": label, "scorer": "score_apps.py",
                           "max_tests": args.max_tests, "n_problems": res["n"],
                           "comparison": "whitespace-normalized"},
            }
            outp = os.path.join(args.results_dir, label,
                                f"apps-{tier}_metrics.rescored.json")
            json.dump(audit, open(outp, "w", encoding="utf-8"), indent=2)

            bucket = TIER_TO_BUCKET[tier]
            records.append({"benchmark": "apps", "model": label, "tier": bucket,
                            "k": 1, "pass_at_k": round(strict_pct, 2),
                            "n_problems": res["n"], "source": "our-eval"})
            records.append({"benchmark": "apps_avg", "model": label, "tier": bucket,
                            "k": 1, "pass_at_k": round(avg_pct, 2),
                            "n_problems": res["n"], "source": "our-eval"})

    scores = {
        "notes": ("APPS test split, re-scored by eval/score_apps.py (the v1 "
                  "bigcode-harness APPS scorer was broken -- it reported 0.0% on "
                  "introductory for generations that are ~12% correct). Same saved "
                  "generations (4-bit, greedy temp 0.2, single sample, Qwen chat "
                  "template); scoring is aligned via the difficulty config and "
                  "whitespace-normalized. Metrics over up to "
                  f"{args.max_tests or 'all'} hidden tests/problem."),
        "model_columns": [
            {"key": "base", "label": "Base (Qwen2.5-Coder-7B-Instruct)"},
            {"key": "finetuned", "label": "Fine-tuned v1 (QLoRA/APPS)"},
        ],
        "records": records,
    }
    scores_dir = os.path.dirname(os.path.abspath(args.scores_out))
    os.makedirs(scores_dir, exist_ok=True)
    json.dump(scores, open(args.scores_out, "w", encoding="utf-8"), indent=2)
    print(f"\nWrote {args.scores_out} and per-tier *_metrics.rescored.json.")
    print("Render:  python eval/build_results_table.py "
          f"--scores {args.scores_out} --out results/report.rescored.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
