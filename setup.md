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
Streams `nvidia/OpenCodeReasoning` (`split_0`), applies the firewall
(`split=='train'` + allowed sources only), and writes ~10k reasoning-trace
examples ≤ 8192 tokens each.
```bash
!python data/prepare_reasoning_traces.py            # -> data/reasoning_train.jsonl
```
Sanity-inspect a couple of rendered rows before training:
```python
import json
rows = [json.loads(l) for l in open("data/reasoning_train.jsonl")][:2]
for r in rows:
    print(r["source"], r["difficulty"], r["n_tokens"])
    print(r["prompt"][:300]); print("---"); print(r["response"][:400]); print("=====")
```
Confirm each `response` is `<think> …reasoning… </think>` + a fenced Python
solution (NOT a golfed one-liner).

### 1.5 Smoke test the training loop FIRST (the Kaggle trial run)
Prove the loop end-to-end on a tiny slice before the full run:
```bash
!python training/train_qlora.py --max-samples 300 --max-steps 15 --save-steps 10
```
Expect: model loads in 4-bit, `train_on_responses_only: enabled`, loss logged and
dropping, an `outputs/checkpoint-*` written. If this passes, do the full run.

### 1.6 Full run + push to the Hub
```bash
!python training/train_qlora.py \
    --epochs 1 --save-steps 50 --save-total-limit 3 \
    --push --merge-16bit \
    --hf-username <user> --hf-repo qwen2.5-coder-7b-ocr-qlora
```
Defaults are v2-tuned: `--max-seq-len 8192 --batch-size 1 --grad-accum 8 --lr 1e-4`
(long sequences on a 16 GB T4; lower LR to curb forgetting). Use **Save Version →
Save & Run All (Commit)** to run in the background.

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
python /path/to/SCM/eval/fix_tokenizer_config.py --repo <user>/qwen2.5-coder-7b-ocr-qlora
```

### 2.3 Generate on APPS test (harness), then SCORE with score_apps.py
`run_apps_eval.sh` auto-applies the harness patches and saves generations.
**Do NOT trust the harness's own `*_metrics.json`** — its APPS scorer is broken for
this setup (see CLAUDE.md §8). Score the saved generations with `score_apps.py`.

> **Reasoning models emit long `<think>` before the code.** For the v2 model, give
> generation a large `--max_new_tokens` and ensure code extraction takes the **last**
> ```python fence after the reasoning (Phase 3 wires this into the generation step;
> `score_apps.py` scores already-extracted code).

```bash
# generate (base and v2 fine-tune), 150/tier
LIMIT=150 NUM_PROCESSES=2 MODEL=<user>/qwen2.5-coder-7b-ocr-qlora LABEL=finetuned \
  HARNESS_MAIN=$(pwd)/main.py bash /path/to/SCM/eval/run_apps_eval.sh
LIMIT=150 NUM_PROCESSES=2 MODEL=Qwen/Qwen2.5-Coder-7B-Instruct LABEL=base \
  HARNESS_MAIN=$(pwd)/main.py bash /path/to/SCM/eval/run_apps_eval.sh

# score (replaces the harness scorer); base/introductory should be ~16%
python /path/to/SCM/eval/score_apps.py \
  --results-dir /path/to/SCM/results/apps --labels base,finetuned --max-tests 25 \
  --scores-out /path/to/SCM/results/scores.rescored.json
```

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
