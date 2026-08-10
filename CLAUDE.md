# CLAUDE.md — Operating Guide for the SCM Project

**Project in one line:** Fine-tune Qwen2.5-Coder-7B-Instruct with QLoRA on APPS
competitive-programming problems, then measure — with a difficulty-stratified
pass@k table — how much of the gap to frontier LLMs it closes versus the
un-tuned base model.

> **The full build spec is [`SCM.md`](./SCM.md). Read it before any non-trivial
> work.** This file is the short operating manual; `SCM.md` is the authoritative
> source for scope, rationale, and resolved decisions. Current status lives in
> [`PROGRESS.md`](./PROGRESS.md).

---

## Non-negotiable rules (breaking these can invalidate the project)

1. **Never train on LiveCodeBench — ever, under any circumstance.** It exists to
   be untrained-on; training on it invalidates every published frontier number
   we compare against. Evaluation only.
2. **Keep the three datasets physically separate.** APPS-train → training,
   APPS-test → internal held-out check, LiveCodeBench → eval only. They must
   never share a variable or file during preprocessing. This is the one silent
   mistake that would quietly ruin the whole project.
3. **Kaggle GPU must be `T4 x2`, never the default P100.** Unsloth requires GPU
   compute capability ≥ 7.0; the P100 (6.0) is documented to fail. T4 (7.5)
   works. Easy setting to miss.
4. **Checkpoint aggressively during training.** Free sessions can end without
   warning; losing hours of a multi-hour run to a disconnect is the single most
   avoidable waste of the compute budget.
5. **Report the hard-tier gap honestly, don't downplay it.** The defensible
   result is at the easy/medium tiers and in the before/after vs. the base
   checkpoint. A transparent table that shows where the 7B model still loses is
   more credible than a cherry-picked one. This is not "beat Claude."

---

## Architecture: two pipelines, kept apart

The **training** pipeline (Kaggle, T4 x2, Unsloth/QLoRA) and the **evaluation**
pipeline (a separate, clean environment running bigcode-evaluation-harness for
APPS and `lcb_runner` for LiveCodeBench) are deliberately isolated. Unsloth and
bigcode-evaluation-harness each want their own library versions, and mixing them
in one environment invites hard-to-debug conflicts. Train in one notebook, push
the model to the Hugging Face Hub, evaluate fresh in another.

---

## Tech stack (quick reference — see SCM §4 for the why)

| Layer | Choice |
|---|---|
| Training platform | Kaggle Notebooks, **GPU: T4 x2** (background execution) |
| Prototyping | Google Colab (free T4) — tiny smoke tests only |
| Method | QLoRA (4-bit) via **Unsloth** — required by 16GB VRAM, not optional |
| Base model | Qwen2.5-Coder-7B-Instruct, 4-bit (Unsloth pre-quantized), Apache 2.0 |
| Training data | APPS, **train split only** |
| Eval (APPS) | bigcode-evaluation-harness (`--tasks apps`, `--allow_code_execution`) |
| Eval (frontier) | LiveCodeBench `lcb_runner` + its public leaderboard for cited numbers |
| Model hosting | Hugging Face Hub |
| Optional demo | HF Spaces + ZeroGPU (Gradio) |

**Scope guardrails:** Python only; QLoRA only (no full fine-tune); no custom
execution sandbox (the harness already does it — keep its per-problem timeout on).

---

## Target repo structure (SCM §9)

```
SCM/
├── README.md                    # objective, results table (Phase 7)
├── CLAUDE.md                    # this operating guide
├── SCM.md                       # full build spec
├── PROGRESS.md                  # living status tracker
├── setup/
│   ├── 01_colab_smoketest.md
│   ├── 02_kaggle_training.md
│   ├── 03_evaluation_environment.md
│   └── 04_hf_spaces_demo.md     # optional
├── data/
│   ├── prepare_apps.py          # download + train/test split, formatting
│   └── prepare_livecodebench.py # download only — never fed into training
├── training/
│   └── train_qlora.py           # Unsloth + LoRA config, checkpointing
├── eval/
│   ├── run_apps_eval.sh
│   ├── run_livecodebench_eval.sh
│   └── build_results_table.py
├── results/
│   └── report.md
└── demo/                        # optional Gradio app for HF Spaces
```

Directories are created by the phase that needs them — don't scaffold ahead.

---

## Working conventions

- **Work phase by phase** (SCM §8, Phases 0–9). Don't attempt the whole project
  in one pass. Confirm each phase works before moving on.
- **The `setup/*.md` files are living documents, written *after* a stage
  actually works** — capture what really happened, including errors hit and how
  they were fixed. They are not upfront plans.
- **Update `PROGRESS.md` whenever a phase changes status** (start, finish, skip)
  and log any new decision or blocker there.
- **Only publish numbers your own eval pipeline produced**, and be ready to walk
  through exactly how pass@k was computed.
