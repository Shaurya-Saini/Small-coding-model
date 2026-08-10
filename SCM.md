# Small Coding Model (SCM) — Fine-Tuning a SLM on Competitive Programming, Benchmarked Against Frontier LLMs

---

## 0. How to use this document

This is the build spec for Claude Code. Paste it in as `CLAUDE.md`, and work phase by phase (Section 8) rather than asking for the whole project at once. A few phases explicitly produce a short setup `.md` file of their own (marked below) — these are your step-by-step guides for the fiddly platform-specific parts (Kaggle GPU settings, environment installs, etc.), written *as you build*, not upfront, so they reflect what actually worked rather than what was planned. Section 1 is for you to read once; the rest is reference.

---

## 1. Glossary (plain-language)

- **LoRA / QLoRA:** instead of retraining an entire model's weights, LoRA adds small trainable "adapter" layers on top of a frozen base model. QLoRA does the same thing but loads the frozen base model in 4-bit precision first, cutting memory needs roughly in half — the difference matters here because it determines whether the model even fits on a free GPU.
- **Compute capability:** a number identifying a GPU's hardware generation. Some software (Unsloth, the tool we're using to fine-tune) has a minimum generation requirement — this isn't a setting you can change, it's fixed by which physical GPU you're given.
- **pass@k:** the standard way to score code generation. Generate k candidate solutions per problem; the problem counts as "solved" if any of the k pass all the hidden tests. pass@1 means "did the single best-guess answer work"; pass@10 means "did any of 10 attempts work."
- **Hidden test cases:** the actual input/output pairs used to check whether generated code is correct — not shown to the model, only used after generation to grade it. This is what makes code-generation eval objective (no human or LLM judgment needed) — either the code passes them or it doesn't.
- **Contamination:** when a model has seen the exact evaluation questions during training, making its eval score meaningless (it's recalling, not solving). This is why training data and evaluation data must never overlap.
- **APPS / CodeContests / LiveCodeBench:** three public competitive-programming datasets/benchmarks. APPS and CodeContests come with training splits (for fine-tuning) and are what we'll train on. LiveCodeBench is specifically built to be contamination-free (only uses problems published after major models' training cutoffs) and is what current leaderboards use — this is for evaluation only, never for training.
- **Difficulty stratification:** splitting results by problem difficulty (easy/introductory, medium/interview, hard/competition) instead of one blended average — necessary here because frontier models are near-perfect on easy problems and much weaker on hard ones, so a single average number hides the actually interesting part of the story.
- **bigcode-evaluation-harness:** an existing, open-source tool that already does the "generate code, run it against hidden tests, compute pass@k" loop for you, including for APPS. You configure it and point it at a model; you don't write the execution engine yourself.

---

## 2. Objective and Scope

**Objective:** fine-tune a 7B code-specialized model on competitive programming problems using QLoRA, and measure — honestly, with a difficulty-stratified results table — how much of the gap to frontier LLMs (Claude, GPT, etc.) that closes, compared to the model's own un-tuned starting point.

**The honest framing (carried over from our discussion):** this is not "beat Claude." Frontier models score ~85–90% pass@1 on LiveCodeBench overall, near-perfect on easy problems specifically, and that gap is not something a LoRA adapter on a 7B model closes at the hard end — that's an expected, well-understood capacity difference, not a failure of the method. The genuinely interesting, defensible result lives at the easy/medium tiers and in the before/after comparison against the model's own base checkpoint.

**In scope:**
- QLoRA fine-tuning of Qwen2.5-Coder-7B-Instruct on the APPS training split.
- Evaluation via pass@k on both a held-out slice of APPS and on LiveCodeBench, split by difficulty.
- A comparison table from the start including: the base (un-tuned) model, the fine-tuned model, and published frontier model numbers (pulled from the public LiveCodeBench leaderboard, spot-checked with a small number of your own API calls).
- Stretch goal (only if time/budget remains after the core comparison works): add a same-size or larger model from a different family (e.g., DeepSeek-Coder) as an extra reference point.

**Explicitly out of scope:**
- Full fine-tuning (no QLoRA/LoRA) — not needed, and won't fit the free-tier hardware anyway.
- Training on LiveCodeBench, ever, under any circumstance — it exists specifically to be untrained-on, and using it for training invalidates every published frontier number you're comparing against.
- Multi-language code generation (MultiPL-E-style) — Python only, keeps scope tight.
- Building your own code-execution sandbox from scratch — bigcode-evaluation-harness already does this; configuring it is the task, not writing it.

---

## 3. System Design

**Two separate pipelines, deliberately kept apart (Decision 3 from our discussion):**

**Pipeline A — Training** (Kaggle, T4 GPU)
```
Load Qwen2.5-Coder-7B-Instruct in 4-bit (via Unsloth)
  │
  ▼
Attach LoRA adapters (frozen base model, small trainable layers on top)
  │
  ▼
Fine-tune on APPS training split (problem → correct solution pairs)
  │
  ▼
Save checkpoints regularly (session can disconnect — don't lose progress)
  │
  ▼
Push final adapter + merged model to Hugging Face Hub
```

**Pipeline B — Evaluation** (separate notebook/environment, T4 or CPU — inference only, lighter than training)
```
Pull model from Hugging Face Hub (base model, and separately the fine-tuned model)
  │
  ▼
Run bigcode-evaluation-harness on held-out APPS test problems → pass@k, by difficulty tier
  │
  ▼
Run LiveCodeBench's own eval scripts (lcb_runner) on its problem set → pass@k, by easy/medium/hard split
  │
  ▼
Pull published frontier model scores from the public LiveCodeBench leaderboard for the same splits
  │
  ▼
Assemble one results table: base model | fine-tuned model | frontier models — by difficulty tier
```

Why separate: the training tool (Unsloth) and the evaluation tool (bigcode-evaluation-harness) each want their own set of library versions, and installing both in one environment risks a conflict that's annoying to debug. Train in one notebook, upload the result, evaluate fresh in another.

---

## 4. Tech Stack

| Layer | Choice | Why |
|---|---|---|
| Training platform | Kaggle Notebooks, **GPU: T4 x2** (not the default P100) | Unsloth requires GPU compute capability ≥7.0; the P100 (6.0) is documented to fail training with Unsloth, while T4 (7.5) works. Kaggle also supports background execution, so training keeps running after you close the tab. |
| Prototyping platform | Google Colab (free T4) | Smoke-test the full pipeline on a tiny slice (2–3 problems, a few steps) before committing a real multi-hour run to Kaggle. Interactive and fast to iterate on; not for the real run. |
| Fine-tuning method | QLoRA via Unsloth | A 7B model needs ~18–20GB for plain LoRA — more than the T4's 16GB. QLoRA's 4-bit loading brings this down to ~6–10GB, which fits comfortably. Not optional here — it's required by the hardware. |
| Base model | Qwen2.5-Coder-7B-Instruct, 4-bit (Unsloth's pre-quantized version, ~4–5GB) | Already code-specialized; you're sharpening an existing skill, not teaching code from zero. Apache 2.0 licensed — no restrictions. |
| Training data | APPS (train split only) | Simpler to set up than CodeContests, plugs directly into bigcode-evaluation-harness with no extra parsing work. |
| Evaluation (APPS side) | bigcode-evaluation-harness | Ready-made: generates candidate solutions, executes them against real hidden test cases, computes pass@k — you configure it, you don't build it. |
| Evaluation (frontier comparison side) | LiveCodeBench's own `lcb_runner` scripts, plus its public leaderboard | Contamination-free, currently what real leaderboards use, and lets you cite already-published frontier scores instead of paying for hundreds of API calls. |
| Model hosting | Hugging Face Hub | Free, and both notebooks (training and eval) can pull from/push to it easily. |
| Optional live demo | Hugging Face Spaces + ZeroGPU (free dynamic GPU) | A small Gradio chat demo of the fine-tuned model, hostable at no cost. |

---

## 5. Compatibility & Environment Notes (resolved — no need to re-derive these while building)

- **Kaggle GPU setting:** always select "GPU T4 x2," never leave it on the default P100. This is a one-time notebook setting, easy to miss.
- **Disk space is not a concern:** the 4-bit model (~5GB) + a few thousand training examples (well under 1GB) + checkpoints stays far inside Kaggle's 20GB persistent + larger temp scratch space, and Colab's ~225GB base disk. No cleanup strategy needed at this scale.
- **Library versions:** Unsloth's installer (`pip install unsloth`) automatically pulls compatible versions of PyTorch/transformers/PEFT/TRL — don't hand-pin versions for the training side. For the evaluation side, install bigcode-evaluation-harness fresh in its own notebook/environment rather than alongside Unsloth.
- **Code execution safety:** bigcode-evaluation-harness's recommended safe mode uses Docker containers to sandbox generated code; Kaggle/Colab notebooks don't support running Docker inside them. In practice this is commonly run without the Docker layer for controlled, non-adversarial use like this (you're running your own model's output on known competitive-programming problems, not arbitrary untrusted code) — just keep the harness's per-problem timeout enabled so a stuck/infinite-loop generation can't hang the notebook.
- **Data separation (this is the part you asked about specifically):** APPS train split → training, APPS test split → your own internal held-out check, LiveCodeBench → evaluation only, never touched during training. Keep these physically in different files/variables in your code so it's not possible to accidentally mix them.
- **Licensing:** Qwen2.5-Coder is Apache 2.0. APPS, CodeContests, and LiveCodeBench are all research datasets, free to use for a portfolio project.

---

## 6. How the Hidden-Test Evaluation Actually Works (answering your question directly)

You don't build this by hand. The flow, using bigcode-evaluation-harness:

1. Install it (`git clone` + `pip install -e .`) in your evaluation notebook.
2. Point it at your model (`--model <your-hf-model-path>`) and tell it which benchmark to run (`--tasks apps`, or the APPS difficulty-specific variant).
3. Set `--n_samples` — 1 if you just want strict pass/fail per problem, higher (e.g. 20+) if you want pass@k with k>1.
4. Add `--allow_code_execution` — this tells the harness it's allowed to actually run the generated code against APPS's real hidden test cases (which ship with the dataset, you don't write them).
5. It hands back a pass@k score, automatically, per problem and averaged.

For LiveCodeBench specifically, the process is the same shape but uses that project's own `lcb_runner` scripts instead — clone their repo, run their generation step against your model, then their scoring step, which does the identical "execute against hidden tests" job on LiveCodeBench's own problem set.

Net answer: no manual test-running infrastructure to build. Your work is picking the right flags/config and pointing the tool at the right model — this is genuinely a configuration task, not an engineering-from-scratch one.

---

## 7. Evaluation Plan — the Results Table

Difficulty-stratified from the start, frontier numbers included from the start:

| Difficulty | Base model (Qwen2.5-Coder-7B-Instruct) | Fine-tuned (yours) | Frontier (published, e.g. Claude/GPT) | [Stretch] Other family, same/larger size |
|---|---|---|---|---|
| Easy / Introductory | pass@1, pass@k | pass@1, pass@k | pass@1 (cited) | — |
| Medium / Interview | pass@1, pass@k | pass@1, pass@k | pass@1 (cited) | — |
| Hard / Competition | pass@1, pass@k | pass@1, pass@k | pass@1 (cited) | — |

Also report, same honesty principle as before: how many samples (k) you used, and any cost/latency numbers if you want to make a "cheaper, locally-run alternative" argument alongside the accuracy story.

---

## 8. Build Phases

**Phase 0 — Scaffold + first setup doc.** Repo structure, dependency lists. Produce `setup/01_colab_smoketest.md`: step-by-step for opening a Colab notebook, installing Unsloth, and running a tiny 2–3-problem training smoke test end-to-end.

**Phase 1 — Data pipeline.** Download APPS, split cleanly into train/held-out-test, format as prompt→solution pairs. Verify LiveCodeBench data is downloaded separately and never touched here.

**Phase 2 — Smoke test on Colab.** Run Phase 0's tiny test for real; fix any environment/code issues cheaply before committing Kaggle hours.

**Phase 3 — Real training run + second setup doc.** Produce `setup/02_kaggle_training.md`: the T4-selection reminder, install steps, how to kick off a long background run, and how to check on it later. Then actually run the full QLoRA fine-tune.

**Phase 4 — Push to Hugging Face Hub.** Upload the merged fine-tuned model so the evaluation notebook can pull it fresh.

**Phase 5 — Evaluation setup + third setup doc.** Produce `setup/03_evaluation_environment.md`: installing bigcode-evaluation-harness and lcb_runner in a clean notebook, separate from the training environment.

**Phase 6 — Run the evals.** Base model and fine-tuned model, both benchmarks, all difficulty tiers. Pull frontier numbers from the public leaderboard; spot-check ~20–30 problems yourself via API to confirm scoring alignment.

**Phase 7 — Results + write-up.** Assemble the table from Section 7 into the README.

**Phase 8 — Optional: live demo setup doc.** Produce `setup/04_hf_spaces_demo.md` if you want the Gradio/ZeroGPU demo — only after the core comparison is done and working.

**Phase 9 — Optional stretch.** Only if time/budget remains: add the cross-family/bigger-model row to the table.

---

## 9. Suggested Repo Structure

```
codeforge/
├── README.md                   # objective, results table
├── CLAUDE.md                   # this brief
├── setup/
│   ├── 01_colab_smoketest.md
│   ├── 02_kaggle_training.md
│   ├── 03_evaluation_environment.md
│   └── 04_hf_spaces_demo.md    # optional
├── data/
│   ├── prepare_apps.py          # download + train/test split, formatting
│   └── prepare_livecodebench.py # download only — never fed into training
├── training/
│   └── train_qlora.py           # Unsloth + LoRA config, checkpointing
├── eval/
│   ├── run_apps_eval.sh         # bigcode-evaluation-harness config
│   ├── run_livecodebench_eval.sh # lcb_runner config
│   └── build_results_table.py
├── results/
│   └── report.md                # generated table from Section 7
└── demo/                         # optional Gradio app for HF Spaces
```

---

## 10. Notes That Matter More Than They Look

- **The setup `.md` files are living documents, not upfront specs.** Write each one right after you get that stage actually working, capturing what really happened (including any error you hit and how you fixed it) — that's far more useful to future-you than a plan written before touching the platform.
- **Checkpoint aggressively during training.** Free sessions can end unexpectedly; losing hours of progress to a disconnect is the single most avoidable waste of your limited compute budget.
- **Never let APPS-train, APPS-test, and LiveCodeBench data end up in the same variable/file during preprocessing.** This is the one mistake that would quietly invalidate the entire project if it happened.
- **Report the hard-tier gap honestly, don't downplay it.** A results table that's transparent about where the small model still loses badly is more credible in an interview than one that only shows favorable numbers — and it's exactly the kind of nuance that shows you understand what you built.

---

## 11. Resume-Ready Framing (fill in once you have real numbers)

> Fine-tuned Qwen2.5-Coder-7B via QLoRA on competitive programming problems (APPS); evaluated with pass@k on held-out APPS and LiveCodeBench using execution-based hidden test scoring, stratified by difficulty. Closed **[X]%** of the gap to frontier models (Claude/GPT) on easy/medium-tier problems versus the un-tuned base model, while transparently reporting the remaining gap on hard/competition-tier problems.

As always: only publish numbers your own eval pipeline produced, and be ready to walk through exactly how pass@k was computed if asked.
