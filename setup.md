# SCM — Setup & Reproduction Guide (v2)

Concise, ordered steps to reproduce the **v2** pipeline from this repo. Training
runs on Kaggle (GPU **T4 x2**); evaluation runs in a separate clean environment.

> **v1** (the APPS/golfed-solutions attempt) is frozen in [`Version 1.0/`](./Version%201.0/);
> its own `Version 1.0/setup.md` reproduces it. This guide is v2 (OpenCodeReasoning
> reasoning traces).

## Conventions

- On Kaggle you **`git clone` this repo** into the notebook, then run scripts from
  the repo root. Examples below assume you are inside the cloned `SCM/` directory.
- In a notebook cell: `!python file.py` runs a Python file, `!bash file.sh` runs a
  shell script. Set env vars inline: `!VAR=x python file.py`.
- **Internet must be ON** in the Kaggle notebook (data streams from the HF Hub).

---

## 0. Clone the repo (Kaggle)

```bash
!git clone https://github.com/Shaurya-Saini/SCM.git   # or your fork/URL
%cd SCM
```

(If you push under a different remote, clone that instead — the layout is the same.)

---

## 1. Training (Kaggle, GPU T4 x2) — data prep + fine-tune in one notebook

The v2 data-prep step uses the same libraries the trainer does (Unsloth pulls a
compatible `transformers`/`datasets`/`trl`), so install once and run both steps in
the **same** T4 x2 notebook.

### 1.1 Session
- Right sidebar → **Accelerator = GPU T4 x2** (NOT P100). **Internet = on**.
- Verify: `import torch; print(torch.cuda.get_device_capability(0))  # (7, 5)`

### 1.2 Install
```bash
!pip install -r requirements/train.txt   # unsloth (+ compatible torch/transformers/trl/datasets)
```

### 1.3 HF token (write) as a Kaggle Secret
- huggingface.co → Settings → Access Tokens → new **Write** token.
- Kaggle → Add-ons → Secrets → add `HF_TOKEN`.
```python
import os
from kaggle_secrets import UserSecretsClient
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
```

### 1.4 Build the v2 corpus (OpenCodeReasoning reasoning traces)
Streams `nvidia/OpenCodeReasoning` (`split_0`) and writes reasoning-trace examples,
each **guaranteed to fit the training budget**, to `data/reasoning_train.jsonl`.

> **⚠️ v2.1 — this step had a bug that broke the first v2 model. Read this.** The
> length filter used to silently pass EVERY row (on transformers≥5,
> `len(apply_chat_template(tokenize=True))` returns `2`), so oversized traces
> (median ~7500, max ~20000 tokens) slipped in and got their `</think>`+code tail
> truncated at train time — the model then learned to reason forever and never emit
> code. **Fixed** (correct length measurement + a startup probe). With the fix the
> filter actually bites, so **use `--max-seq-len 4000`** (margin under the 4096 train
> cap) and expect the numbers described below.

> **What's actually in split_0 (verified by a full 567k-row scan):** its
> `split=='train'` partition is effectively **`code_contests` only** — DeepMind's
> aggregate of Codeforces / AtCoder / CodeChef problems (so you still get
> cross-judge variety), each with a DeepSeek-R1 reasoning trace. There is **no
> APPS/TACO here** (they're in split_1, which needs a join — deferred to v2.1), and
> the other platforms don't survive the train filter. So `--per-source-frac`
> defaults to **1.0 (no balancing)** — there's a single source to draw from. The
> APPS-test firewall is automatically satisfied (no APPS data present).

```bash
# v2.1: 2500 examples, each <= 4000 tokens (margin under the 4096 train cap).
!python data/prepare_reasoning_traces.py --target 2500 --max-seq-len 4000   # -> data/reasoning_train.jsonl
# Smoke variant: --target 200 --max-seq-len 4000
```
`--target`/`--max-seq-len` live on the **data-prep** step. The **training** step then
uses whatever is in the file (don't confuse with training's `--max-samples`, which
only *caps* rows for a smoke).

**What you MUST see now (proves the fix is live):**
- The log prints `Length filter: exact tokens via … (max_seq_len=4000; probe=NNN
  tokens)` with **NNN a real number in the hundreds** — never `probe` missing or ~2.
- `examples written : 2500`, and **`rows scanned` is MUCH larger than 2500** (tens of
  thousands) — because the working filter now rejects the ~70% of traces that are too
  long. This run therefore takes several minutes and does more parsing than the old
  (broken) ~1–2 min run. That's correct.

Then sanity-inspect (token range must be realistic, NOT `2 - 2`):
```python
import json
from collections import Counter
rows = [json.loads(l) for l in open("data/reasoning_train.jsonl")]
print("count:", len(rows), "| sources:", Counter(r["source"] for r in rows),
      "| tokens:", min(r["n_tokens"] for r in rows), "-", max(r["n_tokens"] for r in rows))
for r in rows[:3]:
    print(r["source"], r["difficulty"], r["n_tokens"])
    print(r["prompt"][:300]); print("---"); print(r["response"][:400]); print("=====")
```
Confirm: `tokens:` spans up into the ~3000–4000s (not `2 - 2`), and each `response`
is `<think> …reasoning… </think>` + a fenced Python solution (reasoning + clean
code). If tokens read `2 - 2`, the fix didn't take — re-clone/re-pull the repo.

### 1.5 Smoke test the training loop FIRST
```bash
!python training/train_qlora.py --max-samples 300 --max-steps 15 --save-steps 10
```
Expect: model loads in 4-bit; the **truncation guard** prints `Token lengths:
max=… over-limit=0/300 (max_seq_len=4096)` (the `0` is what matters — it proves no
example will be tail-truncated); `train_on_responses_only: enabled`; loss logged and
dropping; an `outputs/checkpoint-*` written. **If it aborts with `ABORT: N/… examples
exceed --max-seq-len`, your corpus is still oversized** — you skipped `--max-seq-len
4000` at §1.4 or are running stale code; regenerate the data.

### 1.6 Full run + push to the Hub
```bash
!python training/train_qlora.py \
    --epochs 1 --save-steps 50 --save-total-limit 3 \
    --push --merge-16bit \
    --hf-username <user> --hf-repo qwen2.5-coder-7b-ocr-qlora
```
Defaults are v2-tuned: `--max-seq-len 4096 --batch-size 1 --grad-accum 8 --lr 1e-4`.
Watch for `over-limit=0/2500` right before training starts (the guard) — that is your
confirmation the v2.1 corpus is clean. Overwriting `…-ocr-qlora` is fine (the first
v2 checkpoint is broken); use `…-ocr-qlora-v2` if you want to keep it as evidence.

> **Time budget (measured on this exact setup):** Unsloth's free tier trains on
> **one** T4 only (`Num GPUs used = 1`), at **~92 s/step** @ 4096 tokens (≈ half of
> 8192's 178 s/step). Effective batch 8 → one step per 8 examples. So:
> - **3000 examples** ≈ 375 steps ≈ **9.6 h train**, + model load + the final
>   16-bit merge/upload ≈ **~10 h total**. Fits the 12 h commit cap, modest margin.
> - **2500 examples** ≈ 313 steps ≈ **8 h train** (~8.5 h total) — safer margin,
>   recommended if you want the push to complete comfortably in one session.
>
> `--save-steps 50` checkpoints throughout, so a timeout is recoverable with
> `--resume` (§1.7). The LoRA adapter is also saved locally before the push.

### 1.6a Interactive vs. committed run — which to use

Do BOTH, in this order:

1. **Interactive first (a few minutes), as a pre-flight — not the full train.**
   With the notebook open interactively, run the install → clone → HF-token → data
   prep cells, confirm `examples written : 2500`, then start training and watch the
   **first 2–3 steps**: you want `Num GPUs used = 1`, `train_on_responses_only:
   enabled`, loss printing, and **~90 s/step** with no CUDA OOM. Once those look
   right, **interrupt/stop** — don't babysit 8 h in an interactive tab.
2. **Then Save Version → Save & Run All (Commit) for the real 8 h run.** A committed
   run executes the whole notebook headless, survives you closing the browser, and
   saves `outputs/` + logs as a version. An **interactive** session can die on
   disconnect/idle and would lose the run — so the long train MUST be a commit.

Notes for the committed run:
- The commit re-runs every cell from scratch, so it re-does data prep (~2 min) and
  training in order — that's fine and fully reproducible.
- **Internet = on** and the **`HF_TOKEN` secret** must be enabled for the committed
  session (Notebook settings), or the OCR stream and the `--push` will fail.
- Check progress under the notebook's **Logs** tab while it runs.

### 1.7 Resume after a disconnect
```bash
!python training/train_qlora.py --epochs 1 ... --resume   # auto-detects outputs/checkpoint-*
```

### 1.8 Confirm push
Hub should show `…/qwen2.5-coder-7b-ocr-qlora` (merged 16-bit) and `…-ocr-qlora-lora`
(adapter). This is a **distinct** repo from the v1 model — v1 is not overwritten.

---

## 2. Evaluation (fresh environment, NO Unsloth)

> **Phase 3 (v2 eval overhaul) is in progress.** The APPS re-scoring path below
> (generate with the harness, score with `eval/score_apps.py`) is validated and
> gives the direct v1→v2 continuity numbers (4-bit, same 150/tier subset). The v2
> **headline** eval — LiveCodeBench (bf16) + HumanEval+/MBPP+ sanity + avg@k — is
> being added; see `V2_PROGRESS.md` Phase 3.

### 2.1 Install (pins matter — install LAST so nothing upgrades them back)
```bash
pip install -r requirements/eval.txt
git clone https://github.com/bigcode-project/bigcode-evaluation-harness
cd bigcode-evaluation-harness && pip install -e . && pip install -r requirements.txt
pip install "datasets>=2.16,<4.0" "transformers>=4.44,<5.0" bitsandbytes
```

### 2.2 Preflight + one-time tokenizer repair
```bash
python /path/to/SCM/eval/preflight.py                                   # want "Preflight PASSED"
# Repair the v2 tokenizer on the Hub (Unsloth pushes extra_special_tokens as a v5
# list; transformers 4.x wants a dict). REQUIRED for the fine-tuned model or it
# fails to load. Base model is unaffected.
python /path/to/SCM/eval/fix_tokenizer_config.py --repo <user>/qwen2.5-coder-7b-ocr-qlora
```
If preflight fails only on `pyext` (`inspect.getargspec`), run
`python /path/to/SCM/eval/fix_pyext_py312.py` and re-run preflight — it edits the
installed `pyext.py` in place (idempotent).

### 2.3 Generate on APPS test (harness), then SCORE with score_apps.py
`run_apps_eval.sh` auto-applies the harness patches and saves generations.
**Do NOT trust the harness's own `*_metrics.json`** — its APPS scorer is broken for
this setup (see CLAUDE.md §8). Score the saved generations with `score_apps.py`.

> **`HARNESS_MAIN` must point at the harness's real `main.py`.** A fresh `!bash`
> cell resets cwd to `/kaggle/working`, so `$(pwd)/main.py` only works if you first
> **`%cd` into the cloned harness dir** (below). A wrong path both crashes AND
> silently skips the critical `fix_harness_apps.py` patch (`WARN: harness apps.py
> not found …`) — watch for `[SCM] APPS prompt/postprocess style = …` in the log to
> confirm the patch ran.

> **v2 is a reasoning model** — it emits a long `<think>` before the code. Pass
> `PROMPT_STYLE=v2` for the fine-tune: the wrapper then rebuilds the
> OpenCodeReasoning training prompt, raises `--max_length_generation` to 6144 (a
> 2048 budget truncates before any code → spurious 0%), strips `<think>…</think>`,
> and extracts the **last** ```python fence. The base model uses the default
> `PROMPT_STYLE=v1`. `score_apps.py` scores the already-extracted code.

```bash
%cd /kaggle/working/bigcode-evaluation-harness   # so $(pwd)/main.py resolves

# ALWAYS smoke single-GPU first (LIMIT=10, NUM_PROCESSES=1): one clean traceback on
# error instead of a buried ChildFailedError, and it confirms the patch/prompt.
LIMIT=10 NUM_PROCESSES=1 MODEL=Qwen/Qwen2.5-Coder-7B-Instruct LABEL=base \
  HARNESS_MAIN=$(pwd)/main.py bash /path/to/SCM/eval/run_apps_eval.sh

# Only after a clean smoke: full tiers, 150/problem, T4 x2. Base = v1 prompt (default).
LIMIT=150 NUM_PROCESSES=2 MODEL=Qwen/Qwen2.5-Coder-7B-Instruct LABEL=base \
  HARNESS_MAIN=$(pwd)/main.py bash /path/to/SCM/eval/run_apps_eval.sh
# v2 fine-tune = reasoning prompt + big gen budget. Smoke it single-GPU first too.
LIMIT=150 NUM_PROCESSES=2 PROMPT_STYLE=v2 MODEL=<user>/qwen2.5-coder-7b-ocr-qlora LABEL=finetuned \
  HARNESS_MAIN=$(pwd)/main.py bash /path/to/SCM/eval/run_apps_eval.sh

# score (replaces the harness scorer); base/introductory should be ~16%
python /path/to/SCM/eval/score_apps.py \
  --results-dir results/apps --labels base,finetuned --max-tests 25 \
  --scores-out /path/to/SCM/results/scores.rescored.json
```

> **GATE — before the v2 *full* run, confirm the fine-tune actually TERMINATES.**
> The first v2 model looped forever (never closed `<think>`, never emitted code,
> ~8 min/problem, ~0%). After the v2 **smoke** (`LIMIT=2 NUM_PROCESSES=1
> PROMPT_STYLE=v2`), dump a generation and check it reaches the code:
> ```python
> import json
> g = json.load(open("results/apps/finetuned/apps-introductory_generations_apps-introductory.json"))[0][0]
> print("chars:", len(g), "| </think>:", "</think>" in g,
>       "| ```python:", "```python" in g, "| <|im_end|>:", "<|im_end|>" in g)
> print(g[-800:])
> ```
> Want `</think>=True`, ```` ```python=True ````, `<|im_end|>=True`, and **length well
> under the 6144 budget** (fast, ~1–2 min/problem). If it still loops (no `</think>`,
> runs to budget), the retrained model still isn't terminating — do NOT launch the
> 150/tier run (it'd be ~20 h and score 0). Fallbacks, in order: raise
> `TEMPERATURE=0.6`; if still looping it's a training issue → revisit the corpus
> (a repetition-penalty patch to the harness generation call is the last resort).
> Only once the smoke shows clean, terminating code is the `LIMIT=150 NUM_PROCESSES=2`
> v2 run worth starting (run it as a committed session — it's multi-hour).

### 2.4 Build results table + diagram
```bash
python eval/build_results_table.py --scores results/scores.rescored.json \
  --out results/report.md        # -> results/report.md + results/figures/*.png
```

---

## 3. LiveCodeBench + frontier (Phase 3, in progress)

Clone `LiveCodeBench`, `pip install -e .`, run `eval/run_livecodebench_eval.sh`
(record the release version), and add published leaderboard numbers as a frontier
column. **Pin the LCB version window to dates that post-date the OpenCodeReasoning
corpus** (or decontaminate by problem id) so codeforces/code_contests training
overlap can't contaminate the headline. Never train on LiveCodeBench.
