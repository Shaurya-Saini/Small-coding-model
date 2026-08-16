# CLAUDE.md — SCM Project Brief & Resume Document

This is the **single source of truth** for resuming the SCM project in a future
session. It is self-contained: read it fully before any non-trivial work.
`README.md` is the public-facing documentation; `setup.md` is the reproduction
guide. `V2_PROGRESS.md` is the living execution tracker for v2 (checklist +
status); this file remains the durable brief — read it fully before any work.

---

## 1. Project in one line

Fine-tune **Qwen2.5-Coder-7B-Instruct** with **QLoRA (4-bit)** on the **APPS**
competitive-programming train split, then measure — with a difficulty-stratified
**pass@1** table — how it compares to the un-tuned base model (and, in v2, to
published frontier numbers).

**Honest framing:** this is *not* "beat Claude." The defensible result is the
before/after vs. the base checkpoint, reported transparently including where the
7B loses. Only publish numbers our own eval pipeline produced.

---

## 2. Current status (v1 COMPLETE — 2026-08-12)

- **Trained:** QLoRA, 1 epoch, 620 steps (~2h34m on a single Kaggle T4), final
  train loss 0.6514.
- **Published:** merged 16-bit model at
  `Shaurya-saini/qwen2.5-coder-7b-apps-qlora` (+ `…-lora` adapter).
- **Evaluated:** base vs. fine-tuned on APPS test, 150 problems/tier, 4-bit,
  chat-templated prompt.
- **Result (honest):** the v1 fine-tune **underperformed the base on every tier.**

| Difficulty (APPS test, pass@1 = strict; avg test-case rate in parens) | Base | Fine-tuned v1 |
|---|---|---|
| Easy / Introductory | 0.0% (0.84%) | 0.0% (0.22%) |
| Medium / Interview | 7.3% (30.2%) | 2.0% (7.4%) |
| Hard / Competition | 0.0% (0.54%) | 0.0% (0.0%) |

Two metrics are reported: **pass@1** (strict accuracy — all hidden tests pass) and
**average test-case pass rate** (partial credit). Artifacts: `results/report.md`
(two tables + two diagrams), `results/figures/apps.png` (pass@1) and
`results/figures/apps_avg.png` (avg test-case rate), `results/apps/{base,finetuned}/*`
(raw metrics + generations), and the README analysis.

- **Not done in v1 (deferred to v2):** LiveCodeBench + frontier column; a live demo.
- **Git:** not initialized (user choice). **HF username:** `Shaurya-saini`.

---

## 3. Why v1 underperformed (root cause) and the v2 plan

### Root cause
1. **Trained on the single *shortest* solution per problem.** `prepare_apps.py`
   sorts candidate solutions by length and keeps the shortest. Shortest APPS
   solutions are golfed, cryptic, competition-style — imitating them **narrowed**
   a strong instruct model toward terse, cruder code. This is a data-selection
   mistake, not a method mistake. **The single most important thing to change.**
2. **Catastrophic forgetting / distribution narrowing** — even one epoch eroded
   the base's broad coding ability in exchange for a narrow style.
3. **A learned syntax artifact** — the fine-tune over-produces closing brackets
   (e.g. `input().split()))`), causing compile errors the base avoids.

### v2 research findings (web survey, 2026-08-17)
The modern winning recipe for competitive programming is **not** "imitate final
solutions." It is: **start from a reasoning model → train it to produce a long
chain-of-thought scratchpad → optimize for whether the final code passes hidden
tests.** Reference points (LiveCodeBench pass@1):
- **Qwen2.5-Coder-7B-Instruct** (our v1 base, *non-reasoning*): ~37% published —
  but our v1 eval read it far lower (see eval overhaul). Non-reasoning models have
  a low ceiling on the hard tier no matter how they're tuned.
- **DeepSeek-R1-Distill-Qwen-7B** (reasoning, same size): ~37.6%.
- **OpenCodeReasoning-Nemotron-7B** — **SFT only, no RL**, trained on 736K R1
  *reasoning traces* over 28.9K CP problems: **51.3%** (+13.7 over R1-distill-7B).
  Proves imitation *still works* when the target is reasoning, not golfed code.
- **DeepCoder-14B** (RL/GRPO+ on 24K verified problems, 32×H100 × 2.5 weeks):
  **60.6%**, ~o3-mini, Codeforces 1936. Best ceiling; compute-infeasible for us.

RL primer: RL/GRPO (a.k.a. RLVR) drops the labeled target entirely — the model
generates its own attempts and is rewarded only when the code passes tests. It
composes with QLoRA (Unsloth supports GRPO+QLoRA) but needs many sampled
generations + a code sandbox *per step* → far more compute than SFT. **Out of
scope for a free T4** beyond a tiny didactic run.

### v2 plan (pragmatic, fits our hardware)
- **Primary path — SFT on reasoning traces (proven, same stack as v1):** QLoRA SFT
  on **OpenCodeReasoning**-style long reasoning traces instead of golfed shortest
  solutions. *Still supervised imitation — but of the reasoning process + a clean
  solution, not a cryptic destination.* Directly fixes root causes 1 & 3.
- **Base model (decide before training):** keep Qwen2.5-Coder-7B-Instruct (clean
  v1→v2 ablation) *or* switch to **DeepSeek-R1-Distill-Qwen-7B** (reasoning base,
  higher hard-tier ceiling).
- **Anti-forgetting:** lower LR (~1e-4), fewer steps, optionally mix in general
  instruction data; track a general-code sanity metric (below) to prove no
  regression.
- **RL (stretch/didactic only):** a small GRPO+QLoRA run with a strict +1/0
  pass-all-tests reward on a few hundred *clean* problems (≥5 tests each to avoid
  reward hacking) — for the write-up, not to chase DeepCoder's numbers.

### Eval overhaul (v1 eval understated everything — fix before comparing)
The v1 numbers are depressed by the *measurement*, not just the model. Tell-tale:
the **base** model scored **0.0% on APPS *introductory*** — a strong 7B
(HumanEval ~88%) must solve some beginner problems, so a flat 0 signals a broken /
over-strict pipeline, not ability. Suspects: 4-bit degradation, a single greedy
sample (high variance), tight generation/timeout budget, strict all-hidden-tests
scoring colliding with stdin/stdout format mismatches, and/or extraction grabbing
the wrong code block.

v2 eval changes:
1. **Add a sanity benchmark (HumanEval+ / MBPP+).** If the base doesn't land ~85%
   there, the *pipeline* is broken — fix it before trusting any CP number. Doubles
   as the forgetting guardrail.
2. **Make LiveCodeBench the headline** via `lcb_runner` (robust extraction +
   execution, versioned, contamination-free, frontier numbers built in). Pin one
   LCB **version/date window** (v5 or v6) and report base, v2, and 2-3 frontier
   models on the *same* window. (LCB is a rolling benchmark — v6 covers May 2023–
   Apr 2025, 1055 problems; the "no post-2023 data" impression came from reading the
   frozen original-paper abstract, not the live leaderboard.)
3. **pass@1 = avg@k with temperature** (k≈5-10), not one greedy sample — cuts the
   variance that makes 150-problem tiers swing wildly.
4. **Headline eval in bf16** (report 4-bit separately as the "local-deploy" story)
   so we don't cripple our model in a frontier comparison.
5. **Adequate token budget** (reasoning models need large max_new_tokens) + verify
   a known-correct solution scores 1.0 before trusting a run.
6. **APPS demoted** to optional secondary/sanity, or dropped — it is old (2021),
   partly contaminated, and its harness was the fragile part of v1.

Note: eval fixes change *measured* numbers (likely revealing a much stronger base
and a fairer v2 read), not the underlying model — but since the entire deliverable
*is* an honest measurement, a broken eval is a first-class bottleneck.

See `V2_PROGRESS.md` for the execution checklist and pending decisions.

---

## 4. Non-negotiable rules

1. **Never train on LiveCodeBench — ever.** Eval only. Training on it invalidates
   every published frontier number.
2. **Keep the three datasets physically separate:** APPS-train → training,
   APPS-test → held-out eval, LiveCodeBench → eval only. Never share a file or
   variable during preprocessing.
3. **Kaggle GPU must be `T4 x2`, never P100.** Unsloth needs compute capability
   ≥ 7.0; P100 (6.0) fails, T4 (7.5) works.
4. **Checkpoint aggressively during training** (free sessions can die).
5. **Report the hard-tier gap honestly**, don't cherry-pick.
6. **Only publish numbers our own eval pipeline produced.**

---

## 5. Architecture — two isolated pipelines

- **Training** (Kaggle, T4 x2): Unsloth + QLoRA (4-bit) → push to HF Hub.
- **Evaluation** (fresh env, NO Unsloth): bigcode-evaluation-harness (APPS) +,
  later, `lcb_runner` (LiveCodeBench). Pull the model from the Hub.

They are kept apart because Unsloth and the eval tools want conflicting library
versions. The Hub is the handoff point.

---

## 6. Tech stack & key decisions (with reasons)

| Choice | Reason |
|---|---|
| **QLoRA, not full FT** | 7B in 16-bit needs ~28 GB; the free T4 has 16 GB. Required, not preference. |
| **4-bit (Unsloth pre-quantized base)** | Fits training in 16 GB. We also eval in 4-bit (both models) — the realistic local-deploy setting. |
| **LoRA r=16** | Train ~0.53% of weights (40M/7.66B), freeze the rest. |
| **Base: Qwen2.5-Coder-7B-Instruct** | Already code-specialized, Apache-2.0. Sharpen a skill, not teach from zero. |
| **Training data: APPS train split only** | Plugs into the harness cleanly. |
| **Eval prompt = Qwen chat template** | The model was fine-tuned inside the chat template; evaluating on a bare prompt drives it off-distribution. Same template for both models = fair. |
| **pass@1 = strict accuracy, stratified** | Standard, execution-based, honest; per-tier shows where it wins/loses. |
| **150 problems/tier subset** | Full 5000-problem set is impractical per free session; report n. |
| **No custom sandbox** | The harness runs code with a per-problem timeout (no Docker in notebooks). |
| Scope | Python only; QLoRA only. |

---

## 7. Repo structure

```
SCM/
├── README.md                 # public documentation (plan, decisions, results, links)
├── CLAUDE.md                 # this brief (resume document)
├── setup.md                  # concise reproduction guide
├── requirements/
│   ├── data.txt              # datasets, huggingface_hub
│   ├── train.txt             # unsloth
│   └── eval.txt              # datasets<4, transformers<5, accelerate, bitsandbytes, matplotlib
├── data/
│   ├── prepare_apps.py       # APPS TRAIN -> data/apps_train.jsonl (Parquet loader; parse_int=str)
│   └── prepare_livecodebench.py  # metadata manifest only; NEVER trained on
├── training/
│   └── train_qlora.py        # Unsloth QLoRA; prompt-masking; checkpoint/resume; HF push
├── eval/
│   ├── run_apps_eval.sh      # harness wrapper; auto-applies the fix_* patches below
│   ├── run_livecodebench_eval.sh  # lcb_runner wrapper (v2)
│   ├── preflight.py          # verify GPU + transformers<5 + datasets<4 + bitsandbytes + pyext
│   ├── fix_tokenizer_config.py    # one-time Hub repair: extra_special_tokens list -> additional_special_tokens
│   ├── fix_pyext_py312.py    # shim inspect.getargspec for Python 3.11+ (auto-run by wrapper)
│   ├── fix_harness_apps.py   # patch harness apps.py: level bug + chat-template prompt + output extraction (auto-run)
│   └── build_results_table.py     # scores.json -> report.md + bar diagram (matplotlib)
└── results/
    ├── report.md             # generated table + diagram (no prose)
    ├── scores.json           # normalized numbers feeding the table
    ├── scores.template.json  # schema template
    ├── figures/              # generated PNGs
    └── apps/{base,finetuned}/*    # raw metrics + generations
```

Directories are created by the phase that needs them.

---

## 8. Issues hit in v1 and their fixes (all now baked into the code)

### Training / data
- **`ValueError: Exceeds the limit (4300 digits) for integer string conversion`**
  (`prepare_apps.py`, Python 3.12). APPS `input_output` has thousands-of-digit
  integers. Fix: `json.loads(raw, parse_int=str)` in `_safe_json` (we only read
  `fn_name`).
- **`TypeError: unsloth_push_to_hub() takes 2 positional args`** — Unsloth's
  `push_to_hub` takes only the repo id; push the tokenizer separately with
  `tokenizer.push_to_hub(...)`. (`push_to_hub_merged` *does* take the tokenizer.)
- **`datasets >= 4.0` removed loading scripts + `trust_remote_code`** — our
  scripts load APPS/LCB from the Hub **Parquet export** (`refs/convert/parquet`).
  Do NOT reintroduce `trust_remote_code`.
- **TRL API:** `dataset_text_field` / `max_length` / `packing` live in `SFTConfig`,
  not `SFTTrainer` kwargs.

### Evaluation environment (bigcode-evaluation-harness on a modern Kaggle stack)
- **`AttributeError: 'APPS' object has no attribute 'dataset'`** — harness loads
  `codeparrot/apps` via its script, which `datasets>=4.0` removed. Fix: pin
  `datasets>=2.16,<4.0`; wrapper exports `HF_DATASETS_TRUST_REMOTE_CODE=1`.
- **`TypeError: ... unexpected keyword argument 'load_in_4bit'`** — transformers
  5.x removed the direct kwarg the harness passes. Fix: pin `transformers>=4.44,<5.0`.
- **`PackageNotFoundError: ... bitsandbytes`** — 4-bit needs it; the clean eval env
  lacked it. Fix: `pip install bitsandbytes`.
- **`CUBLAS_STATUS_ALLOC_FAILED`** at first forward — OOM: 7B in 16-bit (~15 GB)
  on a 16 GB T4. Fix: `LOAD_IN=4bit` (wrapper default); with T4 x2 each GPU gets a
  4-bit replica.
- **`AttributeError: 'list' object has no attribute 'keys'`** loading the
  fine-tuned tokenizer — the push saved `extra_special_tokens` as a v5-style list;
  transformers 4.x wants a dict. Fix: `fix_tokenizer_config.py` renames it to
  `additional_special_tokens`. Base model unaffected.
- **`AttributeError: module 'inspect' has no attribute 'getargspec'`** at scoring —
  the APPS scorer's `pyext` uses a function removed in Python 3.11. Fix:
  `fix_pyext_py312.py` (auto-run) shims it.
- **`UnboundLocalError: ... 'level'`** in `apps.py::process_results` — upstream
  bug (dead `if level is None` block). Fix: `fix_harness_apps.py` removes it.
- **100% compile errors / `avg_accuracy 0.0`** — NOT model quality. Two causes,
  both fixed by `fix_harness_apps.py`: (a) `<|im_end|>` leaked into code (harness
  eos is `<|endoftext|>`) → strip it; (b) **prompt-format mismatch** — the harness
  fed a bare `QUESTION:/ANSWER:` prompt but the model was trained inside the chat
  template → off-distribution garbage. Fix: override `get_prompt` to rebuild the
  training-time chat prompt and `postprocess_generation` to extract the assistant
  turn. This dropped compile errors 9→0–2/tier and is what made the numbers valid.
- **~45 s/problem after chat-templating** — generation ran to `max_length` because
  the harness eos rarely fires in chat. Fix: wrapper passes `--eos "<|im_end|>"`.
- **Multi-GPU noise:** `accelerate` auto-ran 2 processes and buried real errors in
  `ChildFailedError`. Wrapper defaults to `NUM_PROCESSES=1`; set `2` for speed once
  a run is known-good.
- **Always verify `config.model`** in each metrics JSON — labels got crossed once
  (a "finetuned" run was actually the base model).
- **APPS task names use hyphens** (`apps-introductory`), confirmed.

### The v1 APPS numbers are BROKEN — the harness scorer, not the model (2026-08-17)
- **The bigcode-harness APPS scorer reported 0.0% on introductory for base
  generations that are actually ~12% correct.** Proved by re-scoring the *saved*
  generations with an independent, alignment-verified executor: base introductory
  = 12/100 strict (13 normalized). Generation/postprocessing was fine all along;
  the harness's **scoring/orchestration** is the bug. The backwards v1 signal
  (introductory 0% < interview 7.3%) was the tell.
- **Do not trust ANY v1 APPS number** (base *or* fine-tune, all tiers) — they came
  from that scorer. Re-score everything with `eval/score_apps.py` before reporting.
- **The alignment trap (bit us twice):** `codeparrot/apps` must be loaded via the
  difficulty **config** — `load_dataset("codeparrot/apps","introductory",split=
  "test")` — NOT `difficulties=[...]` (silently ignored → loads all 5000, misaligns
  generation[i] with problem[i], every score bogus). `--show-io` in `diagnose_apps.py`
  catches this (question must match the code).
- **APPS strict exact-match is partly unfair** — multi-valid-answer problems (e.g.
  "output any 3 vertices achieving the max") can't be credited. Extra reason to make
  **LiveCodeBench (functional checks)** the v2 co-headline.
- **New eval tools:** `eval/diagnose_apps.py` (autopsy: static + execution +
  `--show-io` alignment check) and `eval/score_apps.py` (the standalone verified
  scorer that REPLACES the harness scorer; stdin + call-based; emits
  `results/scores.rescored.json`). Validation gate: `score_apps.py --max-tests 25`
  reproduces base/introductory ≈12%.

### Eval design facts to remember
- The wrapper is self-healing: it auto-runs `fix_pyext_py312.py` and
  `fix_harness_apps.py` before each launch, exports `HF_DATASETS_TRUST_REMOTE_CODE=1`
  and `HF_ALLOW_CODE_EVAL=1`, and passes `--load_in_4bit --eos "<|im_end|>"`.
- Always run `eval/preflight.py` and a `LIMIT=10` smoke before a full run.
- Diagnose any surprising 0.0 by dumping a generation (`print(gens[i][0])`): 100%
  *compile* errors means a format/extraction issue, not model quality.

---

## 9. Working conventions

- Work incrementally; confirm each stage before moving on.
- The eval env is bleeding-edge-fragile — keep the version pins in `requirements/eval.txt`.
- Update this file and `README.md` when status or decisions change.
- Reference model: `Shaurya-saini/qwen2.5-coder-7b-apps-qlora`.
- Kaggle notebooks: training `small-coding-model-v1-0`, push `scm-upload-hf`,
  eval `scm-eval` (links in README).
