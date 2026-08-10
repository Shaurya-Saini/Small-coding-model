# SCM — Small Coding Model

Fine-tuning **Qwen2.5-Coder-7B-Instruct** with **QLoRA** on **APPS**
competitive-programming problems, then measuring — with a difficulty-stratified
**pass@k** table — how much of the gap to frontier LLMs (Claude, GPT, …) the
fine-tune closes versus the un-tuned base model.

> This is **not** "beat Claude." The honest, defensible result lives at the
> easy/medium tiers and in the before/after comparison against the model's own
> base checkpoint. The remaining hard-tier gap is reported transparently.

## Documentation map

| File | Purpose |
|---|---|
| [`SCM.md`](./SCM.md) | Full build spec — scope, rationale, resolved decisions. **Authoritative.** |
| [`CLAUDE.md`](./CLAUDE.md) | Short operating manual + non-negotiable rules. |
| [`PROGRESS.md`](./PROGRESS.md) | Living status tracker (phase by phase). |
| `setup/*.md` | Platform setup guides, written *after* each stage actually works. |

## Pipelines (kept deliberately separate)

- **Training** — Kaggle Notebooks, **GPU T4 x2**, Unsloth + QLoRA (4-bit).
- **Evaluation** — a fresh environment: bigcode-evaluation-harness (APPS) and
  `lcb_runner` (LiveCodeBench). Never share libraries with the training env.

## Data separation (critical)

APPS-train → training · APPS-test → internal held-out check ·
**LiveCodeBench → evaluation only, never trained on.** These never share a file
or variable during preprocessing.

## Results

_Difficulty-stratified pass@k table to be filled in Phase 7 — see_
[`PROGRESS.md`](./PROGRESS.md) _for the placeholder and current status._
