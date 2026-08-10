# 03 — Evaluation Environment

> **Status: DRAFT (pre-verification).** Correct after the first real eval run.

The eval environment is **separate** from the training environment (SCM.md §3):
Unsloth and the eval tools want different library versions. Use a fresh Colab
notebook / Kaggle notebook / venv. Do **not** `pip install unsloth` here.

Two independent tools:
- **bigcode-evaluation-harness** → APPS pass@k (held-out test split).
- **LiveCodeBench `lcb_runner`** → LiveCodeBench pass@1 (eval-only benchmark).

Run each of `base` and `finetuned` through both, then assemble the table.

---

## Step 0 — GPU + base extras

Any T4 works (inference is lighter than training). Enable Internet.

```bash
pip install -r requirements-eval.txt   # datasets, transformers, accelerate, ...
```

## Step 1 — APPS via bigcode-evaluation-harness

```bash
git clone https://github.com/bigcode-project/bigcode-evaluation-harness
cd bigcode-evaluation-harness
pip install -e .
pip install -r requirements.txt
# CRITICAL, run LAST: the harness loads codeparrot/apps via its loading SCRIPT,
# which datasets >= 4.0 removed. Pin below 4.0 AFTER every other install so
# nothing upgrades it back:
pip install "datasets>=2.16,<4.0"
```

> The harness calls `load_dataset("codeparrot/apps", name=<tier>)` with **no**
> `trust_remote_code`. Two things make that work: `datasets < 4.0` (scripts still
> exist) and `HF_DATASETS_TRUST_REMOTE_CODE=1` (grants script trust without
> editing the harness) — `run_apps_eval.sh` exports the env var for you. Without
> the pin you get `AttributeError: 'APPS' object has no attribute 'dataset'` (the
> dataset load failed silently, then the task tried to use it).

Sanity-check the exact APPS task names for your version (hyphen vs underscore):

```bash
python main.py --help 2>/dev/null | grep -i apps || true
```

Then run the harness wrapper for each model (from the repo root that has `main.py`):

```bash
# fine-tuned
MODEL=Shaurya-saini/qwen2.5-coder-7b-apps-qlora LABEL=finetuned \
  HARNESS_MAIN=$(pwd)/main.py  bash /path/to/SCM/eval/run_apps_eval.sh
# base
MODEL=Qwen/Qwen2.5-Coder-7B-Instruct LABEL=base \
  HARNESS_MAIN=$(pwd)/main.py  bash /path/to/SCM/eval/run_apps_eval.sh
```

`--allow_code_execution` runs generated code against APPS's real hidden tests.
No Docker in notebooks (SCM.md §5) — the harness's per-problem timeout stays on
so a runaway generation can't hang the run. Metrics land in
`results/apps/<label>/<task>_metrics.json`.

## Step 2 — LiveCodeBench via lcb_runner

```bash
git clone https://github.com/LiveCodeBench/LiveCodeBench
cd LiveCodeBench
pip install -e .
```

- **Record the release version** you use (must match `data/livecodebench/manifest.json`
  and `PROGRESS.md`). Leaderboard numbers are version-specific.
- **Custom HF model registration** — a non-registry model may need an entry in
  `lcb_runner/lm_styles.py` (or a local-model flag). Determine the exact
  mechanism for the installed version and record it here.

```bash
MODEL=Shaurya-saini/qwen2.5-coder-7b-apps-qlora LABEL=finetuned RELEASE=release_v5 \
  bash /path/to/SCM/eval/run_livecodebench_eval.sh
MODEL=Qwen/Qwen2.5-Coder-7B-Instruct LABEL=base RELEASE=release_v5 \
  bash /path/to/SCM/eval/run_livecodebench_eval.sh
```

## Step 3 — Frontier numbers

Pull published pass@1 for the **same** LiveCodeBench release/tiers from the public
leaderboard. Spot-check ~20–30 problems yourself via API to confirm scoring
alignment. Put these in the `frontier` rows of `results/scores.json` with
`"source": "cited"`.

## Step 4 — Assemble the table

Copy `results/scores.template.json` → `results/scores.json`, fill every cell from
the metrics JSONs above (and the manifest for LiveCodeBench difficulty joins),
then:

```bash
pip install matplotlib                  # for the diagrams (in requirements-eval.txt)
python eval/build_results_table.py     # -> results/report.md + results/figures/*.png
```

This writes the stratified pass@k table **and** a grouped-bar diagram per
benchmark (pass@1 by tier, base/finetuned/frontier) into `results/figures/`,
embedded in `report.md`. Add `--no-charts` for a table-only build.

---

## Success criteria

- [ ] Eval env is clean (no Unsloth); both tools import.
- [ ] APPS metrics JSONs produced for base **and** finetuned, all three tiers.
- [ ] LiveCodeBench results produced for base **and** finetuned; release recorded.
- [ ] `results/report.md` generated with real numbers.

## Pitfalls seen

- **2026-08-11 — `AttributeError: 'APPS' object has no attribute 'dataset'`**
  (after a clean model load). Root cause: `datasets >= 4.0` in the env, so the
  harness's script-based `codeparrot/apps` load failed (swallowed as a warning),
  leaving `self.dataset` unset. Fix: `pip install "datasets>=2.16,<4.0"` as the
  final install step; `run_apps_eval.sh` already exports
  `HF_DATASETS_TRUST_REMOTE_CODE=1`. Then re-run — no kernel restart needed
  (the harness runs in a fresh subprocess via `accelerate launch`).
- **Unauthenticated HF requests warning** — set `HF_TOKEN` (Kaggle Secret) in the
  env before running for higher rate limits / faster downloads. Not fatal.
- _(still to record: APPS task-name spelling confirmed by `main.py --help`,
  lcb_runner custom-model registration, OOM at long generations, timeout tuning)_
