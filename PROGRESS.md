# PROGRESS.md — SCM Project Tracker

Living status tracker for the SCM project. Update this whenever a phase changes
status or a new decision/blocker appears. Full spec: [`SCM.md`](./SCM.md).
Operating rules: [`CLAUDE.md`](./CLAUDE.md).

**Status legend:** ⬜ Not started · 🟨 In progress · ✅ Done · ⏭️ Skipped/optional

> **Current phase: Phase 3 — Real training run on Kaggle (T4 x2).** Phase 2
> **complete**: the self-contained `setup/colab_smoketest.ipynb` ran end-to-end
> on a free Colab T4 (2026-08-10) — 4-bit load, LoRA attach (0.53% trainable),
> APPS-train Parquet load, 20 steps (loss ≈ 0.63), valid generation, and an
> optional merged-16bit HF push all succeeded. Stack: Unsloth 2026.8.11 /
> transformers 5.5.0 / torch 2.11.0 / datasets ≥ 4.0. **Next:** run the real
> QLoRA fine-tune (v1) on Kaggle via `setup/02_kaggle_training.md`.

---

## Phase status (SCM §8)

| Phase | Description | Deliverable(s) | Status | Notes |
|---|---|---|---|---|
| 0 | Scaffold + first setup doc | Repo structure, dependency lists, `setup/01_colab_smoketest.md` | ✅ | README, requirements-{data,train,eval}.txt, setup doc (DRAFT, verify after Phase 2) |
| 1 | Data pipeline | `data/prepare_apps.py`, `data/prepare_livecodebench.py` (download only) | 🟨 | Both scripts written + compile + `--help` OK. Not yet run on real data (write-only mode). Separation guardrails baked in. |
| 2 | Smoke test on Colab | Verified tiny end-to-end run; `setup/colab_smoketest.ipynb` | ✅ | Full pass on Colab T4 (2026-08-10): load→LoRA→APPS Parquet→20 steps (loss 0.63)→generate→HF push. Doc finalized w/ real results |
| 3 | Real training run + second setup doc | `training/train_qlora.py`, `setup/02_kaggle_training.md`, trained adapter | 🟨 | Script + Kaggle doc ready and smoke-validated. **User action:** run v1 fine-tune on Kaggle T4 x2 |
| 4 | Push to Hugging Face Hub | Merged fine-tuned model on HF Hub | 🟨 | Built into `train_qlora.py --push --merge-16bit` → `Shaurya-saini/qwen2.5-coder-7b-apps-qlora`. Needs `HF_TOKEN` + a real run |
| 5 | Evaluation setup + third setup doc | `setup/03_evaluation_environment.md`, `eval/*.sh` | 🟨 | Both harness wrapper scripts + eval-env doc (DRAFT) written. Not yet run |
| 6 | Run the evals | `eval/run_apps_eval.sh`, `eval/run_livecodebench_eval.sh` outputs | ⬜ | **User action:** base + fine-tuned, both benchmarks, all tiers; spot-check ~20–30 via API |
| 7 | Results + write-up | `eval/build_results_table.py`, `results/report.md`, `README.md` | 🟨 | Table builder written + tested on template; `scores.template.json` + `report.md` placeholder in place. Fills once real numbers exist |
| 8 | Optional: live demo doc | `setup/04_hf_spaces_demo.md`, `demo/` Gradio app | ⏭️ | Only after core comparison works |
| 9 | Optional stretch | Cross-family/larger-model row in table | ⏭️ | Only if time/budget remains |

---

## Deliverables checklist

- [x] `setup/00_running_files.md` _(how to run .py/.sh in Colab/Kaggle)_
- [x] `setup/colab_smoketest.ipynb` _(self-contained end-to-end smoke notebook)_
- [x] `setup/01_colab_smoketest.md` _(updated with datasets≥4 + TRL fixes)_
- [x] `setup/02_kaggle_training.md` _(DRAFT — verify after first Kaggle run)_
- [x] `setup/03_evaluation_environment.md` _(DRAFT — verify after first eval run)_
- [ ] `setup/04_hf_spaces_demo.md` (optional)
- [x] `data/prepare_apps.py` _(written; not yet run on real data)_
- [x] `data/prepare_livecodebench.py` _(written; not yet run on real data)_
- [x] `training/train_qlora.py` _(written + compiles; not yet run on GPU)_
- [x] `eval/run_apps_eval.sh` _(written; not yet run)_
- [x] `eval/run_livecodebench_eval.sh` _(written; not yet run)_
- [x] `eval/build_results_table.py` _(written + tested on template)_
- [x] `results/scores.template.json` + `results/report.md` placeholder
- [ ] Fine-tuned model pushed to Hugging Face Hub _(needs real Kaggle run)_
- [ ] `results/report.md` with the completed results table _(needs real numbers)_
- [ ] `README.md` results section filled in _(Phase 7)_
- [ ] `demo/` Gradio app (optional)

---

## Decisions log (resolved — don't re-litigate)

- **Method:** QLoRA via Unsloth — required by the T4's 16GB VRAM, not a preference.
- **Base model:** Qwen2.5-Coder-7B-Instruct, 4-bit (Apache 2.0). Already
  code-specialized — sharpening a skill, not teaching from zero.
- **Training data:** APPS train split only (plugs into bigcode-evaluation-harness
  with no extra parsing).
- **Two isolated pipelines:** train (Unsloth) and eval (harness + lcb_runner) in
  separate environments to avoid dependency conflicts.
- **LiveCodeBench = eval only, forever.** Never touched during training.
- **Kaggle GPU = T4 x2**, never the default P100 (compute capability requirement).
- **Scope:** Python only; QLoRA only; no custom execution sandbox.

### Session decisions (2026-08-10)

- **HF Hub target:** username is **`Shaurya-saini`** (baked into scripts as the
  default, overridable via `HF_USERNAME` env / `--hf-username`). Planned repos:
  `Shaurya-saini/qwen2.5-coder-7b-apps-qlora` (merged 16-bit, for eval) and
  `…-lora` (adapter). Push needs a **write** `HF_TOKEN` (see setup docs 02/03).
- **Training design:** loss masked to solution tokens (`train_on_responses_only`);
  checkpoint every 50 steps with `--save-total-limit 3` and `--resume` auto-detect;
  push merged 16-bit so the eval env can load it without Unsloth.
- **`datasets` >= 4.0 breaking change (2026-08-10):** loading scripts +
  `trust_remote_code` removed. APPS and LiveCodeBench are script-based, so we now
  load both from the Hub **Parquet export** (`refs/convert/parquet`) via
  `load_apps_split` / `load_lcb_test`. Do NOT reintroduce `trust_remote_code`.
- **TRL API:** training uses `SFTConfig` (holds `dataset_text_field`, `max_length`,
  `packing`); `SFTTrainer` no longer takes these as direct kwargs.
- **Version control:** git intentionally NOT initialized yet (user choice).
- **Build mode:** write-only, targeting Colab/Kaggle — no heavy deps installed
  or datasets downloaded on the local Windows dev machine. Scripts verified by
  `py_compile` + `--help` only.
- **APPS solution selection:** shortest-first, 1 solution/problem by default
  (`--max-solutions-per-problem` to raise). Prompt uses APPS-style
  QUESTION/ANSWER + IO-format hint to match the eval harness's format.
- **APPS test split:** the existing `codeparrot/apps` `test` split IS the internal
  held-out set — bigcode-evaluation-harness evaluates on it directly; we do not
  re-slice the train split.

## Open questions / blockers

- **LiveCodeBench version_tag** to evaluate on — scripts default to `release_v5`;
  confirm the latest stable tag when first run and record the chosen version here
  (leaderboard numbers are version-specific).
- **APPS task-name spelling** in the installed bigcode-evaluation-harness
  (`apps-introductory` vs `apps_introductory`) — verify before the eval run.
- **lcb_runner custom-model registration** — exact mechanism for a non-registry
  HF model varies by version; confirm during eval setup.

---

## Results table (SCM §7 — fill in as numbers arrive)

| Difficulty | Base (Qwen2.5-Coder-7B-Instruct) | Fine-tuned (ours) | Frontier (published, cited) | [Stretch] Other family |
|---|---|---|---|---|
| Easy / Introductory | _pass@1, pass@k_ | _pass@1, pass@k_ | _pass@1 (cited)_ | — |
| Medium / Interview | _pass@1, pass@k_ | _pass@1, pass@k_ | _pass@1 (cited)_ | — |
| Hard / Competition | _pass@1, pass@k_ | _pass@1, pass@k_ | _pass@1 (cited)_ | — |

Also record: number of samples (k) used, and any cost/latency numbers for the
"cheaper locally-run alternative" argument.
