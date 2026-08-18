# Small Coding Model (SCM)

Fine-tuning a **7B code LLM** for competitive-programming / LeetCode-style problem
solving, and measuring — with a difficulty-stratified, execution-based **pass@1**
evaluation — how each fine-tune compares to its own un-tuned base checkpoint (and,
from v2 on, to published frontier numbers).

This README is the **complete project documentation**: it covers both model
versions, the methodology, and the honest performance/conclusions for each.

> **Framing.** This is *not* an attempt to "beat" frontier models. Frontier systems
> score in the high-80s pass@1 on public leaderboards, and no LoRA adapter on a 7B
> closes that gap at the hard end. The defensible, interesting result is the
> transparent **before/after vs. the base checkpoint**, reported honestly including
> where the small model loses. Only numbers produced by our own evaluation pipeline
> are ever published.

---

## Project map: two versions

| | **Version 1.0** — done (regressed) | **Version 2.0** — in progress |
|---|---|---|
| **Idea** | QLoRA SFT on APPS, imitating the single *shortest* solution per problem | QLoRA SFT on **reasoning traces** (chain-of-thought → clean solution) |
| **Base** | Qwen2.5-Coder-7B-Instruct | Qwen2.5-Coder-7B-Instruct (unchanged — clean ablation) |
| **Training data** | APPS train split (golfed shortest solutions) | OpenCodeReasoning-style reasoning traces |
| **Result** | **Regressed** vs. base on every tier (see below) | TBD |
| **Where it lives** | Frozen snapshot in [`Version 1.0/`](./Version%201.0/) | Repository **root** (active dev tree) |

- **[`Version 1.0/`](./Version%201.0/)** is a self-contained, frozen snapshot of the
  first attempt — kept as **proof of work** and for possible reproduction. It has
  its own [README](./Version%201.0/README.md) with the full v1 write-up.
- **The repository root is the active v2 development tree.** The existing
  `data/`, `training/`, `eval/`, `results/`, and `requirements/` directories are
  reused and evolved for v2. Execution status lives in
  [`V2_PROGRESS.md`](./V2_PROGRESS.md); the durable design brief is
  [`CLAUDE.md`](./CLAUDE.md).

---

## Objective

Take a strong, already code-specialised 7B model and specialise it further on
competitive-programming problems using a memory-efficient fine-tuning method
(QLoRA), then quantify the effect against the base model on a held-out test set,
stratified by difficulty (introductory / interview / competition). v2 additionally
introduces a contamination-free headline benchmark (LiveCodeBench) and a frontier
comparison on a single pinned window.

---

## Architecture (both versions)

The project is split into two deliberately isolated environments, because the
training and evaluation toolchains require conflicting library versions. The
Hugging Face Hub is the handoff point between them.

1. **Training (Kaggle, GPU T4 x2).** Load Qwen2.5-Coder-7B-Instruct in 4-bit via
   Unsloth, attach LoRA adapters (rank 16), fine-tune, checkpoint regularly, and
   push the merged 16-bit model and the adapter to the Hub.
2. **Evaluation (separate clean environment, no Unsloth).** Pull the base and
   fine-tuned models from the Hub, *generate* solutions on the held-out test split,
   then score by executing each solution against hidden tests with our own
   `eval/score_apps.py` (v2 adds LiveCodeBench via `lcb_runner`).

**Non-negotiable data firewall:** APPS-train → training, APPS-test → held-out eval,
LiveCodeBench → **eval only, never trained on**. The three never share a file or
variable.

### Key implementation decisions (carried across versions)

| Decision | Reason |
|---|---|
| QLoRA rather than full fine-tuning | A 7B in 16-bit needs ~28 GB; the free Kaggle T4 has 16 GB. A hardware requirement, not a preference. |
| 4-bit quantisation | Fits training in 16 GB. v1 also *evaluated* in 4-bit (realistic local-deploy setting); v2 makes the LiveCodeBench headline **bf16** so the model isn't crippled against full-precision APIs. |
| LoRA adapters (rank 16) | Only ~0.53 percent of weights (40M of 7.66B) are trained; the rest stay frozen. |
| Base: Qwen2.5-Coder-7B-Instruct | Already code-specialised, Apache-2.0. Kept identical in v2 so only the *data* changes — a clean before/after ablation. |
| Chat-templated evaluation prompt | The model is fine-tuned inside Qwen's chat template; a bare prompt drives it off-distribution. The same template for both models makes the comparison fair. |
| pass@1 = strict, execution-based, stratified | Generated code must run and pass all hidden tests. Per-tier reporting shows exactly where the model wins and loses. |
| 150 problems per tier | The full 5,000-problem set is impractical per free session; the sample size is always reported. |

---

## Version 1.0 — results (complete)

Evaluated on the APPS held-out test split, 150 problems per difficulty tier. Both
models: 4-bit, single-sample (greedy, temp 0.2), same Qwen chat-template prompt.
Two metrics: **pass@1** (strict — all hidden tests pass) and **average test-case
pass rate** (partial credit).

> **These numbers were re-scored (2026-08-17).** The original v1 run used
> bigcode-evaluation-harness, whose APPS scorer was broken for this setup — it
> reported the base at **0.0 percent** pass@1 on the easy and hard tiers for
> generations that are actually ~16 and ~3 percent correct. The generations were
> fine; the harness mis-aligned each solution against the wrong problem's tests.
> The tables below come from `eval/score_apps.py`, a standalone, alignment-verified
> scorer. See [`Version 1.0/README.md`](./Version%201.0/README.md#the-evaluation-harness-bug).

### pass@1 (strict accuracy)

| Difficulty | Base (Qwen2.5-Coder-7B-Instruct) | Fine-tuned v1 (QLoRA/APPS) |
|---|---|---|
| Easy / Introductory | **16.0** | 0.7 |
| Medium / Interview | **9.3** | 2.0 |
| Hard / Competition | **3.3** | 0.0 |

### Average test-case pass rate

| Difficulty | Base (Qwen2.5-Coder-7B-Instruct) | Fine-tuned v1 (QLoRA/APPS) |
|---|---|---|
| Easy / Introductory | **34.3** | 3.1 |
| Medium / Interview | **36.3** | 6.8 |
| Hard / Competition | **16.2** | 1.3 |

**The v1 QLoRA fine-tune underperformed the base on every tier and under both
metrics.** The base is a respectable coder whose accuracy falls monotonically with
difficulty; the fine-tune never leads and gets far fewer individual tests right,
because much of its output does not even compile. Figures and raw metrics are the
frozen snapshot in [`Version 1.0/results/`](./Version%201.0/results/).

### Why v1 regressed

1. **Training on the shortest (golfed) solution per problem** narrowed a capable
   instruct model toward terse, cryptic, competition-style code. *The single most
   important mistake* — a data-selection issue, not a method issue.
2. **Catastrophic forgetting** — even one epoch eroded the base's broad ability in
   exchange for a narrow style.
3. **A learned syntax artifact** — an over-eager closing-bracket tendency (e.g.
   `input().split()))`) causing compile errors the base avoids.

### v1 conclusions

- QLoRA on this data/config degraded the model rather than improving it; the fault
  is the golfed-solution data and forgetting, **not QLoRA itself**.
- A trustworthy eval requires matching the eval prompt to the training format, and
  **the benchmark harness can itself be the bug** — validate the scorer (a
  known-good model should clear an easy tier) before trusting any comparison.
- Reporting a negative result honestly, with a precise root cause, is a legitimate
  and informative outcome — and it defined the v2 plan.

---

## Version 2.0 — plan and status (in progress)

v2 fixes the two things that sank v1: **(a)** train on *reasoning traces* instead
of golfed solutions, and **(b)** *measure* honestly with a pipeline that isn't
broken. The base model stays Qwen2.5-Coder-7B-Instruct so the only change is the
data — a clean v1→v2 ablation.

### Direction

- **Train on reasoning traces, not golfed solutions.** QLoRA SFT on
  OpenCodeReasoning-style long chain-of-thought traces (`problem → <think>
  reasoning </think> → clean solution`), the same recipe that took a 7B to
  ~51 percent on LiveCodeBench with SFT alone. This directly targets root causes
  #1 and #3.
- **Reduce forgetting:** lower learning rate (~1e-4), fewer steps, optionally mix
  in general instruction data; track a general-code sanity metric as a guardrail.
- **Fixed, broader evaluation:** reuse the corrected `eval/score_apps.py` on the
  same 150 problems per tier for a direct v1→v2 APPS comparison (kept 4-bit for
  apples-to-apples continuity); add a **HumanEval+/MBPP+ sanity bench** (also a
  forgetting guardrail); and make **LiveCodeBench** the **bf16 headline** for a
  contamination-free comparison against 2–3 published frontier models on one pinned
  window. LiveCodeBench is never used for training.
- **Reinforcement learning is v3, conditional.** SFT is capped by the teacher's
  distribution; RL on verifiable rewards (GRPO/RLVR) is the only lever past that
  ceiling. It is attempted only if v2's SFT plateaus, and a full RL run stays out
  of scope for a single free GPU.

### Status

- **Phase 0 (analysis, planning, v1 correction): complete.** Base-model survey, RL
  primer, eval overhaul, and all four open decisions (base model, keep-APPS, eval
  precision, RL timing) are resolved.
- **Phases 1–2 (data + training): first attempt done, and it failed — instructively.**
  A QLoRA fine-tune on 2500 OpenCodeReasoning traces trained and pushed, but at
  evaluation the model reasons endlessly and **never emits a code block** (~0%
  runnable). Root cause was a **data-preparation bug**, not the method: a length
  filter silently became a no-op on newer `transformers` (it measured 2 tokens for
  every example), so over-long traces slipped through and had their solution/close
  tags **truncated during training** — the model literally learned to reason without
  concluding. Both the filter and a new train-time truncation guard are fixed; a
  clean **v2.1 retrain** is the next step. (A textbook case of the project's own
  thesis: measure honestly, and a broken pipeline — here in *data prep* — can masquerade
  as a broken model.)
- **Phase 3 (eval): reasoning-aware APPS path built.** The harness now serves the v2
  model its own training prompt, strips the `<think>` scratchpad, and extracts the
  final code block; the base re-eval is healthy. LiveCodeBench (bf16 headline) +
  HumanEval+/MBPP+ sanity are still to come.
- The live checklist is [`V2_PROGRESS.md`](./V2_PROGRESS.md).

---

## Resources

- **v1 fine-tuned model (merged 16-bit):**
  [Shaurya-saini/qwen2.5-coder-7b-apps-qlora](https://huggingface.co/Shaurya-saini/qwen2.5-coder-7b-apps-qlora)
  (+ the `…-lora` adapter)
- **Training notebook** (v1 QLoRA fine-tune, Kaggle T4):
  [small-coding-model-v1-0](https://www.kaggle.com/code/shauryathemaster/small-coding-model-v1-0)
- **Model-upload notebook** (merge and push to the Hub):
  [scm-upload-hf](https://www.kaggle.com/code/shauryathemaster/scm-upload-hf)
- **Evaluation notebook** (bigcode-evaluation-harness on APPS):
  [scm-eval](https://www.kaggle.com/code/shauryathemaster/scm-eval)

---

## Repository layout

```
SCM/
├── README.md            # this file — complete documentation of v1 + v2
├── CLAUDE.md            # durable design brief / resume document
├── V2_PROGRESS.md       # v2 execution tracker (checklist + status)
├── setup.md             # reproduction guide (active tree)
├── Version 1.0/         # FROZEN v1 snapshot: proof of work + possible reproduction
│   ├── README.md        #   full v1 write-up
│   ├── data/ training/ eval/ results/ requirements/ setup.md
├── data/                # active dev tree (v2): dataset preparation
├── training/            # active dev tree (v2): QLoRA training
├── eval/                # active dev tree (v2): generation + scoring
├── results/             # active dev tree (v2): tables, figures, raw metrics
└── requirements/        # pinned deps (data / train / eval, kept separate)
```

`setup.md` is the concise, step-by-step reproduction guide for the active tree;
`Version 1.0/setup.md` is the frozen v1 version of it.

---

## Non-negotiable rules

1. **Never train on LiveCodeBench — eval only.** Training on it invalidates every
   published frontier number.
2. **Keep the three datasets physically separate** (APPS-train / APPS-test /
   LiveCodeBench).
3. **Kaggle GPU must be `T4 x2`, never P100** (Unsloth needs compute capability
   ≥ 7.0).
4. **Report the hard-tier gap honestly**; only publish numbers our own eval
   pipeline produced.
