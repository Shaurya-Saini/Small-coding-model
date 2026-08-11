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
# CRITICAL, run LAST — the deps the harness needs, installed AFTER everything else
# so nothing upgrades them back:
#   * datasets < 4.0     : harness loads codeparrot/apps via its loading SCRIPT,
#                          which datasets 4.0 removed.
#   * transformers < 5.0 : harness passes `load_in_4bit=True` straight to
#                          from_pretrained, a kwarg transformers 5.0 removed.
#   * bitsandbytes       : provides the 4-bit kernels; missing it ->
#                          "PackageNotFoundError: ... bitsandbytes".
pip install "datasets>=2.16,<4.0" "transformers>=4.44,<5.0" bitsandbytes
```

**Preflight — always run this before the harness** (fails in seconds on any
version/dep problem, instead of crashing 90s into a model download):

```bash
python /path/to/SCM/eval/preflight.py   # want: "Preflight PASSED"
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

**One-time: repair the fine-tuned model's tokenizer for transformers 4.x.** The
training push (Unsloth + transformers 5.x) saved the special tokens as a v5-style
`extra_special_tokens` **list**, which transformers 4.x can't parse
(`AttributeError: 'list' object has no attribute 'keys'`). Rename it to the
4.x-native `additional_special_tokens` (the base model already uses that form, so
it needs no repair):

```bash
# needs HF_TOKEN (write) in env; edits the Hub repo once. --dry-run to preview.
python /path/to/SCM/eval/fix_tokenizer_config.py \
  --repo Shaurya-saini/qwen2.5-coder-7b-apps-qlora
```

Then run the harness wrapper for each model (from the repo root that has `main.py`).
**Always do a `LIMIT=10` smoke first** (single-GPU by default — one clean traceback
if anything's wrong), then the full run, then switch to `NUM_PROCESSES=2` for ~2×:

```bash
# 1) 10-problem smoke (fast; proves the whole loop end-to-end)
LIMIT=10 MODEL=Shaurya-saini/qwen2.5-coder-7b-apps-qlora LABEL=finetuned_smoke \
  HARNESS_MAIN=$(pwd)/main.py  bash /path/to/SCM/eval/run_apps_eval.sh

# 2) full fine-tuned run (add NUM_PROCESSES=2 to use both T4s once the smoke passes)
MODEL=Shaurya-saini/qwen2.5-coder-7b-apps-qlora LABEL=finetuned \
  HARNESS_MAIN=$(pwd)/main.py  bash /path/to/SCM/eval/run_apps_eval.sh
# 3) full base run (the before/after baseline)
MODEL=Qwen/Qwen2.5-Coder-7B-Instruct LABEL=base \
  HARNESS_MAIN=$(pwd)/main.py  bash /path/to/SCM/eval/run_apps_eval.sh
```

Defaults are the **safe** ones: `LOAD_IN=4bit`, `NUM_PROCESSES=1` (single GPU).
Knobs: `NUM_PROCESSES=2` (both T4s, ~2×), `NUM_PROCESSES=""` (accelerate auto),
`LOAD_IN=8bit|none`, `LIMIT=N`.

`--allow_code_execution` runs generated code against APPS's real hidden tests.
No Docker in notebooks (SCM.md §5) — the harness's per-problem timeout stays on
so a runaway generation can't hang the run. Metrics land in
`results/apps/<label>/<task>_metrics.json`.

**Precision / VRAM:** the wrapper defaults to `LOAD_IN=4bit`. A 7B model in
16-bit (~15 GB) does **not** leave a 16 GB T4 room to generate — you get
`CUBLAS_STATUS_ALLOC_FAILED` (an OOM). 4-bit weights (~4.5 GB) fit with room, and
with T4 x2 each GPU holds its own replica (2× throughput). Evaluate **base and
fine-tuned identically** (both 4-bit) so the before/after is apples-to-apples —
this is also the realistic way a 7B gets deployed locally. Record it in the
report. Knobs: `LOAD_IN=8bit|none`, `NUM_PROCESSES=1` (force single-GPU),
`LIMIT=10` (quick sanity pass on 10 problems before the full multi-hour run).

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
- **2026-08-11 — `RuntimeError: CUDA error: CUBLAS_STATUS_ALLOC_FAILED` at the
  first forward pass**, after a clean model load. This is OOM: the harness loaded
  the 7B in 16-bit (~15 GB) on a 16 GB T4, leaving nothing for the KV-cache. Fix:
  `LOAD_IN=4bit` (now the wrapper default). With T4 x2 the harness places one
  full 4-bit replica per GPU (`device_map={"": process_index}`), so both GPUs are
  used and it fits.
- **2026-08-11 — `TypeError: Qwen2ForCausalLM.__init__() got an unexpected
  keyword argument 'load_in_4bit'`.** Cause: transformers 5.x removed the direct
  `load_in_4bit` kwarg (now needs `BitsAndBytesConfig`), but the harness still
  passes it. Fix: pin `transformers>=4.44,<5.0` in the eval env (re-run the bash
  cell after; the harness runs in a fresh subprocess, no kernel restart needed).
- **2026-08-11 — `PackageNotFoundError: No package metadata was found for
  bitsandbytes`.** Cause: 4-bit needs `bitsandbytes`, which the clean eval env
  never installed (training got it via Unsloth). Fix: `pip install bitsandbytes`
  (now in `requirements-eval.txt`; `eval/preflight.py` catches it up front). See
  transformers issue #25210.
- **Multi-GPU noise:** `accelerate launch` auto-ran 2 processes and wrapped the
  real error in `ChildFailedError` across both ranks. The wrapper now defaults to
  `NUM_PROCESSES=1` so failures show as a single clean traceback; opt into
  `NUM_PROCESSES=2` for speed once a run is known-good.
- **2026-08-11 — `AttributeError: 'list' object has no attribute 'keys'` in
  `_set_model_specific_special_tokens`** while loading the **fine-tuned**
  tokenizer (model loaded fine first). Cause: the push saved special tokens as a
  transformers-5 `extra_special_tokens` list; transformers 4.x wants a dict.
  Fix: `eval/fix_tokenizer_config.py --repo <finetuned>` renames it to
  `additional_special_tokens` (matches the base model; loads on 4.x and 5.x).
  transformers issue #45376. The base model is unaffected.
- **2026-08-11 — `AttributeError: module 'inspect' has no attribute
  'getargspec'`** at the **scoring** step (generation for all 10 problems
  succeeded first). Cause: the APPS scorer (`codeparrot/apps_metric`) imports
  `pyext`, which uses `inspect.getargspec` — removed in Python 3.11; Kaggle runs
  3.12. Fix: `eval/fix_pyext_py312.py` prepends a `getargspec` shim to the
  installed `pyext` (idempotent). **`run_apps_eval.sh` now runs this
  automatically** before each launch; `preflight.py` also checks it. Same issue
  as FlagOpen/TACO #3.
- **2026-08-11 — `UnboundLocalError: cannot access local variable 'level'`** in
  the harness's `bigcode_eval/tasks/apps.py::process_results` (scoring step).
  It's an upstream bug: a dead `if level is None: level = self.DATASET_NAME`
  block references `level` before assignment, while `compute()` already passes
  `level=self.DATASET_NAME`. Fix: `eval/fix_harness_apps.py` deletes the block;
  **`run_apps_eval.sh` applies it automatically** (path derived from
  `HARNESS_MAIN`).
- **2026-08-11 — 100% "compile errors", `avg_accuracy 0.0`, `runtime errors 0`.**
  NOT a model-quality result. The fine-tuned model emits raw code ending in the
  Qwen turn marker `<|im_end|>`, but the harness's eos is `<|endoftext|>`, so
  `<|im_end|>` leaks into the code and every `compile()` throws `SyntaxError`.
  Fix: `fix_harness_apps.py` also overrides `postprocess_generation` to cut at
  `<|im_end|>`/`<|endoftext|>` (and extract a markdown code block if present, for
  the base instruct model). Auto-applied by `run_apps_eval.sh`. **Diagnose any
  future 0.0 by dumping a generation** (`repr(gens[0][0])`) before trusting it.
- **APPS task-name spelling: confirmed `apps-introductory` (hyphen)** — the
  harness accepted it (`Selected Tasks: ['apps-introductory']`).
- **Unauthenticated HF requests warning** — set `HF_TOKEN` (Kaggle Secret) in the
  env before running for higher rate limits / faster downloads. Not fatal.
- _(still to record: lcb_runner custom-model registration, timeout tuning)_
