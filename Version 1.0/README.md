# SCM — Version 1.0 (frozen snapshot)

This directory is a **self-contained snapshot of Version 1.0** of the Small Coding
Model (SCM) project: the first fine-tuning attempt, its full pipeline, and its
(corrected) results. It is kept as **proof of work** and for **possible
reproduction** of the v1 result. It is not the active development tree — v2 work
happens in the repository root. For the combined, up-to-date project documentation
(v1 **and** v2), see the [root `README.md`](../README.md).

> Everything below describes **v1 only**. The numbers, code, and conclusions here
> are frozen at the state in which v1 was completed and re-scored (2026-08-17).

---

## v1 in one line

Fine-tune **Qwen2.5-Coder-7B-Instruct** with **QLoRA (4-bit)** on the **APPS**
competitive-programming train split, then measure — with a difficulty-stratified,
execution-based **pass@1** table — how the fine-tune compares to its own un-tuned
base checkpoint.

**Outcome:** the v1 fine-tune **regressed** relative to the base on every
difficulty tier and under both metrics. The value of v1 is a clean, honest
before/after measurement plus a precise diagnosis of *why* it regressed — which
directly motivated the v2 redesign.

---

## Objective

Take a strong, already code-specialised 7B model and specialise it further on
competitive-programming problems using a memory-efficient fine-tuning method
(QLoRA), then quantify the effect against the base model on a held-out test set,
stratified by problem difficulty (introductory / interview / competition).

The framing is deliberately modest: frontier models score in the high-80s pass@1
on public leaderboards, and no LoRA adapter on a 7B model closes that gap at the
hard end. The defensible, interesting result lives in the before/after comparison
and in reporting transparently where the small model still fails.

---

## Resources

- **Fine-tuned model (merged 16-bit):**
  [Shaurya-saini/qwen2.5-coder-7b-apps-qlora](https://huggingface.co/Shaurya-saini/qwen2.5-coder-7b-apps-qlora)
  (+ the `…-lora` adapter)
- **Training notebook** (QLoRA fine-tune on Kaggle T4):
  [small-coding-model-v1-0](https://www.kaggle.com/code/shauryathemaster/small-coding-model-v1-0)
- **Model-upload notebook** (merge and push to the Hugging Face Hub):
  [scm-upload-hf](https://www.kaggle.com/code/shauryathemaster/scm-upload-hf)
- **Evaluation notebook** (bigcode-evaluation-harness on APPS):
  [scm-eval](https://www.kaggle.com/code/shauryathemaster/scm-eval)

---

## What was done

- **Trained:** QLoRA (4-bit), 1 epoch, 620 steps (~2h34m on a single Kaggle T4),
  final train loss 0.6514.
- **Published:** merged 16-bit model + LoRA adapter to the Hugging Face Hub.
- **Evaluated:** base vs. fine-tuned on the APPS **test** split, 150 problems per
  tier, 4-bit, chat-templated prompt.
- **Re-scored (2026-08-17):** the original harness scores were invalid; every
  number was re-scored with a standalone verified scorer (`eval/score_apps.py`).

---

## Approach

### Pipeline

The project is split into two deliberately isolated environments, because the
training and evaluation toolchains require conflicting library versions. The
Hugging Face Hub is the handoff point between them.

1. **Training (Kaggle, GPU T4 x2).** Load Qwen2.5-Coder-7B-Instruct in 4-bit via
   Unsloth, attach LoRA adapters, fine-tune on the APPS training split (problem →
   solution pairs), checkpoint regularly, and push the merged model and adapter to
   the Hub.
2. **Evaluation (separate clean environment).** Pull the base and fine-tuned
   models from the Hub and use bigcode-evaluation-harness to *generate* solutions
   on the held-out APPS test split. Scoring is then done by our own
   `eval/score_apps.py`, which executes each solution against the hidden tests and
   computes pass@1 per difficulty tier. (The harness's *own* APPS scorer was found
   broken for this setup and is no longer trusted — see "The evaluation-harness
   bug".)

### Key implementation decisions

| Decision | Reason |
|---|---|
| QLoRA rather than full fine-tuning | A 7B model in 16-bit needs roughly 28 GB; the free Kaggle T4 has 16 GB. QLoRA was a hardware requirement, not a preference. |
| 4-bit quantisation of the base model | Shrinks the base roughly fourfold so it fits in 16 GB. Evaluation is also performed in 4-bit for both models, which is the realistic way a 7B is deployed locally. |
| LoRA adapters (rank 16) | Only about 0.53 percent of the weights (40M of 7.66B) are trained; the rest stay frozen, keeping the base largely intact and the memory footprint small. |
| Base model: Qwen2.5-Coder-7B-Instruct | Already code-specialised and Apache-2.0 licensed. The goal was to sharpen an existing skill, not to teach coding from zero. |
| Train on the APPS training split only | The held-out APPS test split is reserved for evaluation; the two are never mixed. |
| Chat-templated evaluation prompt | The model was fine-tuned inside Qwen's chat template. Evaluating it on a bare prompt drove it off-distribution and produced systematically broken output; matching the training format for both models is what makes the comparison fair and correct. |
| pass@1 as strict accuracy, stratified by difficulty | The standard execution-based metric: generated code must actually run and pass all hidden tests. Splitting by tier shows exactly where the model wins and loses rather than hiding it in a single average. |
| 150 problems per tier | Chat-formatted generation costs several seconds per problem; the full 5,000-problem test set is impractical in a free session. A stratified subset gives a fair signal, with the sample size reported. |

### Evaluation methodology

Both models were evaluated identically: 4-bit weights, fp16 compute, single-sample
generation at temperature 0.2, and the same Qwen chat-template prompt. Generated
code was executed against APPS' hidden test cases. The reported **pass@1 is strict
accuracy** — a problem counts as solved only if every hidden test passes. A softer
secondary metric, the average fraction of individual test cases passed, is used for
additional resolution.

The numbers reported here were **re-scored** by `eval/score_apps.py` after the
original harness scorer was found to be broken. The re-scorer aligns each saved
generation to the correct problem (via the difficulty config), executes it against
the hidden tests with a whitespace-normalized comparison, and evaluates up to 25
tests per problem. The saved generations themselves were unchanged — only the
scoring was fixed.

---

## Results

Evaluated on the APPS held-out test split, 150 problems per difficulty tier. Two
complementary metrics:

- **pass@1 (strict accuracy):** a problem counts as solved only if the generated
  code passes every hidden test. This is the headline, hardest-to-game metric.
- **Average test-case pass rate:** the mean fraction of individual hidden tests
  passed (partial credit); more resolution when strict pass@1 is near zero.

**The v1 QLoRA fine-tune underperformed the base model on every tier and under
both metrics** — but arriving at trustworthy numbers required fixing the
evaluation itself first.

### pass@1 (strict accuracy)

| Difficulty | Base (Qwen2.5-Coder-7B-Instruct) | Fine-tuned v1 (QLoRA/APPS) |
|---|---|---|
| Easy / Introductory | **16.0** | 0.7 |
| Medium / Interview | **9.3** | 2.0 |
| Hard / Competition | **3.3** | 0.0 |

![APPS pass@1 by difficulty — base versus fine-tuned v1](./results/figures/apps.png)

### Average test-case pass rate

| Difficulty | Base (Qwen2.5-Coder-7B-Instruct) | Fine-tuned v1 (QLoRA/APPS) |
|---|---|---|
| Easy / Introductory | **34.3** | 3.1 |
| Medium / Interview | **36.3** | 6.8 |
| Hard / Competition | **16.2** | 1.3 |

![APPS average test-case pass rate by difficulty — base versus fine-tuned v1](./results/figures/apps_avg.png)

The base is a respectable coder whose accuracy falls monotonically with difficulty;
the fine-tune never leads on any tier and gets far fewer individual test cases
right, because much of its output does not even compile. Both tables and figures
are produced by `eval/build_results_table.py` from `results/scores.json`; the full
generated report is in [`results/report.md`](./results/report.md), and raw per-tier
metrics and generations are under `results/apps/`.

---

## The evaluation-harness bug

The original v1 run used bigcode-evaluation-harness, whose APPS scorer was **broken
for this setup**: it reported the base model at **0.0 percent** pass@1 on the easy
and hard tiers for generations that are in fact ~16 and ~3 percent correct. A
capable base model (HumanEval ~88 percent) scoring a flat 0 on *introductory*
problems — and scoring lower on easy than on medium — was the red flag. The
generations were fine; the harness's scoring/orchestration mis-aligned each
solution against the wrong problem's tests and discarded valid answers.

The fix was to replace the harness scorer entirely with `eval/score_apps.py`, a
standalone, alignment-verified scorer that executes each saved solution against the
correct hidden tests (whitespace-normalized, up to 25 tests per problem). The
debugging trail lives in `eval/diagnose_apps.py` and, in the repo root,
`V2_PROGRESS.md`.

**Takeaway that carried into v2:** validate the scorer (a known-good model should
clear an easy tier) before trusting any comparison, and prefer a standalone,
alignment-checked scorer over a fragile third-party harness.

---

## Observations and analysis (why v1 regressed)

Inspecting the raw generations side by side made the cause clear. Three factors, in
order of importance:

1. **Training on the shortest solution per problem.** `prepare_apps.py` selected
   the single shortest correct solution for each APPS problem. In APPS, the
   shortest solutions are heavily golfed, cryptic, competition-style code. Training
   a capable instruct model to imitate that style **narrowed** it: the fine-tune
   produces terse, cramped attempts where the base writes structured, more-correct
   code. This is a data-selection issue and the single largest contributor.
2. **Catastrophic forgetting.** Even a single epoch of imitation on ~5,000
   stylistically-skewed examples eroded the base's broad coding ability in exchange
   for a narrow style that generalises worse.
3. **A learned syntax artifact.** The fine-tune acquired an over-eager
   closing-bracket tendency (e.g. `input().split()))`), producing outright compile
   errors on problems the base handles correctly.

None of this is surprising in retrospect: fine-tuning a strong model on a narrow,
stylistically-skewed target is a well-understood way to regress it.

---

## Conclusions

- QLoRA fine-tuning on this data and configuration did not improve the model; it
  degraded it. The un-tuned Qwen2.5-Coder-7B-Instruct is the stronger model on APPS
  across all three tiers and under both metrics.
- The regression is attributable primarily to training on the shortest (golfed)
  solutions and to catastrophic forgetting, not to the QLoRA method itself.
- A trustworthy evaluation requires matching the evaluation prompt to the training
  prompt format; mismatches can masquerade as catastrophic model failure.
- **The benchmark harness can itself be the bug.** Validate the scorer before
  trusting any comparison.
- Reporting a negative result honestly, with a clear root-cause analysis, is a
  legitimate and informative outcome — and it defined the v2 plan.

---

## Reproduction

`setup.md` in this directory is the concise, step-by-step reproduction guide for
v1 (data → training → evaluation). `requirements/` holds the pinned dependency
lists (kept separate by design for the data, training, and evaluation
environments). `data/`, `training/`, `eval/`, and `results/` hold the v1 pipeline
scripts and outputs exactly as used.

> Note: several scripts here (e.g. `train_qlora.py`, `score_apps.py`, the
> `requirements/`) are shared infrastructure that the v2 tree in the repo root
> continues to evolve. This snapshot preserves the **v1 state** of those files.
