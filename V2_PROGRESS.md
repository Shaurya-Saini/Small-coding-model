# V2_PROGRESS.md — SCM v2 Execution Tracker

Living checklist for v2. `CLAUDE.md` holds the durable brief and the *reasoning*
behind these tasks (§3); this file tracks *what's done, what's next, and open
decisions*. Update the status boxes as work progresses.

**Status legend:** `[ ]` todo · `[~]` in progress · `[x]` done · `[!]` blocked/decision needed

**v2 north star:** fix the two things that sank v1 — (a) train on *reasoning
traces*, not golfed shortest solutions; (b) *measure* honestly with a pipeline that
isn't broken — then report base vs v2 vs 2-3 frontier models on one LiveCodeBench
window.

---

## Open decisions (resolve before building) — `[!]`

| # | Decision | Options | Recommendation | Status |
|---|---|---|---|---|
| D1 | **Base model** | (a) keep Qwen2.5-Coder-7B-Instruct (clean v1→v2 ablation) · (b) DeepSeek-R1-Distill-Qwen-7B (reasoning base, higher hard-tier ceiling) | Lean (b) for ceiling; (a) if a controlled ablation matters more | `[!]` |
| D2 | **Eval: keep or drop APPS** | (a) demote to sanity · (b) drop · (c) keep as co-headline | **DECIDED (c):** fix the APPS harness AND add LCB; report both as co-headlines | `[x]` |
| D3 | **Eval precision** | bf16 headline + 4-bit "local-deploy" column · vs · 4-bit only (as v1) | bf16 headline, keep a 4-bit column | `[!]` |
| D4 | **RL stretch goal** | attempt a tiny GRPO+QLoRA run · vs · SFT-only for v2 | SFT-only for the real result; GRPO only as a didactic write-up if time allows | `[!]` |

**Eval approach (DECIDED):** diagnose v1 first, then rebuild. `eval/diagnose_apps.py`
re-executes saved v1 generations against real APPS tests to prove *why* base scored 0%.

---

## Phase 0 — Analysis & planning `[x]`
- [x] Survey base-model CP performance, existing projects, RL techniques, datasets/benchmarks (2026-08-17 web survey → `CLAUDE.md §3`)
- [x] Diagnose v1 eval as a first-class bottleneck (base 0% on APPS-introductory red flag)
- [x] Update `CLAUDE.md §3` with v2 research + plan + eval overhaul
- [ ] Resolve D1–D4 above

## Phase 1 — Data (fix root cause #1) `[ ]`
- [ ] Choose reasoning-trace dataset (OpenCodeReasoning primary; consider CodeContests+/TACO-Verified for cleaner tests)
- [ ] Write `data/prepare_reasoning_traces.py` (Parquet loader; keep `parse_int=str` guard) → `data/*_train.jsonl`
- [ ] Format each example as `problem → <think> trace </think> → clean solution` inside the chat template
- [ ] Sanity: inspect 10 rendered training samples; confirm targets are *reasoning + clean code*, not golfed one-liners
- [ ] Keep the three-dataset firewall: training data ≠ APPS-test ≠ LiveCodeBench (never share a file/variable)

## Phase 2 — Training (fix root cause #2) `[ ]`
- [ ] Adapt `training/train_qlora.py` (same Unsloth+QLoRA+SFTTrainer stack; prompt-masking on the trace region)
- [ ] Anti-forgetting knobs: LR ~1e-4, fewer steps, optional general-instruction mix
- [ ] Checkpoint aggressively (free Kaggle sessions die); resume tested
- [ ] Push merged 16-bit + LoRA adapter to HF Hub (tokenizer pushed separately — see `CLAUDE.md §8`)

## Phase 3 — Eval overhaul (fix the measurement) `[~]`
- [x] `eval/diagnose_apps.py` written; **static autopsy done** (offline, no GPU):
  - Base introductory: **146/150 compile, 145/150 print, yet 0% strict / 0.84% avg** → measurement/scoring bug (not ability).
  - Fine-tuned introductory: **only 18/150 compile, 132/150 syntax errors (`unmatched ')'`)** → REAL bracket artifact from golfed-code training (root cause #3). Eval fixes won't rescue it; v2 data change will.
- [~] **Ran `diagnose_apps.py` WITH execution (base, introductory, 30):** result refined the diagnosis. NOT a whitespace/scoring artifact (0/30 recovered by normalization). Dominant failure = **RUNTIME_ERROR 22/30**, all input-parsing mismatches (code reads wrong stdin shape: `int('7 5')`, unpack-count errors, EOF). Model writes competent algorithms but its input-reader guesses the wrong format → REAL but likely *recoverable* (clearer I/O-contract prompt + reasoning + multi-sample), not "model hopeless."
- [ ] **Confirm generation↔test alignment** with `--show-io` (does each solution match its aligned question?). Rules out an index/prompt bug before attributing failures to the model.
- [ ] Once alignment confirmed: this is a **prompt/eval-setup** issue as much as model — v2 eval must give a clear input-format contract + reasoning room + avg@k, not just relax comparison.
- [ ] Based on that: fix the APPS harness scoring/prompt (comparison strictness and/or output-format contract) so base measures fairly
- [ ] Add **HumanEval+/MBPP+ sanity bench**; confirm *base* scores ~85% → proves the pipeline works (gate before trusting any CP number)
- [ ] Stand up **`lcb_runner`** (LiveCodeBench); pin one version/window (v5 or v6)
- [ ] Verify a known-correct solution scores 1.0 (execution sanity) before any full run
- [ ] Switch to **pass@1 = avg@k** (k≈5-10, temperature) to cut variance
- [ ] **bf16 headline** eval + optional 4-bit "local-deploy" column
- [ ] Give reasoning models adequate `max_new_tokens`; correct stop tokens
- [ ] Run: **base**, **v2 fine-tune**, **2-3 frontier** models on the *same* LCB window
- [ ] (optional) Keep APPS run per D2 for continuity with v1

## Phase 4 — Report & publish `[ ]`
- [ ] Regenerate `results/` (new tables/figures): LCB headline + HumanEval sanity + optional APPS
- [ ] Direct **v1 → v2** comparison + **v2 vs frontier** on LCB
- [ ] Add Codeforces-rating estimate if available (intuitive "what rating is my model")
- [ ] Update `README.md` analysis + `CLAUDE.md §2` status honestly (including where the 7B still loses)
- [ ] Refresh Kaggle notebook links

---

## Non-negotiables carried from v1 (see `CLAUDE.md §4`)
- Never train on LiveCodeBench — eval only.
- Keep the three datasets physically separate.
- Kaggle GPU = `T4 x2`, never P100.
- Report the hard-tier gap honestly; only publish numbers our own pipeline produced.

## Changelog
- 2026-08-17 — Created. Phase 0 complete; v2 direction set (reasoning-trace SFT +
  eval overhaul). D1–D4 pending.
- 2026-08-17 — D2 decided (APPS co-headline) + eval approach (diagnose-first).
  Added `eval/diagnose_apps.py`; static autopsy split the v1 failure: base = scoring
  bug, fine-tune = real bracket artifact. Next: run it with execution on Kaggle.
