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
- [x] **Ran execution + `--show-io`; found and fixed a loader bug IN THE DIAGNOSTIC** (not the harness): `diagnose_apps.py` loaded the full 5000-problem `all` config (the `difficulties=` kwarg was silently ignored) and scored introductory gen[i] against all-tiers problem[i] → total misalignment. `--show-io` exposed it: every solution solved a different problem than its paired question. The earlier "22/30 runtime errors = model misparses input" read was an ARTIFACT of this bug — **disregard it.** Fixed loader to use the difficulty CONFIG (`load_dataset("codeparrot/apps","introductory",split="test")`), the same order the harness iterates.
- [x] **Re-ran aligned (fixed loader): questions now match code.** On 10 aligned introductory: 1 PASS / 5 WRONG / 3 RUNTIME / 1 NO_OUTPUT. So in the v1 setup (4-bit, greedy, no-reasoning, bare prompt) the base genuinely produces wrong/buggy solutions — real, not pure artifact. TWO caveats though:
  - **v1's reported 0.0% is still likely an undercount:** aligned executor already found 1/10; hitting exactly 0/150 if the true rate is ~10% is statistically near-impossible. Also v1 had introductory (0%) < interview (7.3%), which is backwards. → the v1 *introductory* number specifically is suspect.
  - **APPS strict exact-match is partly unfair:** problems with multiple valid answers (e.g. "choose any 3 vertices achieving max") can't be credited by string match → argues for LiveCodeBench (functional checks) as the v2 headline.
- [x] **Confirmed on 100 aligned introductory: base = 12% strict (13% normalized), NOT 0%.** → v1's harness APPS scorer is BROKEN (threw away valid ~12%-correct generations). Root cause of the whole v1 "base looks terrible" impression is a **scoring bug**, not the model. (Also confirms APPS strict exact-match penalizes multi-answer problems.)
- [x] **FIX written: `eval/score_apps.py`** — standalone verified scorer that replaces the fragile bigcode APPS scorer. Re-scores the SAVED generations (no GPU): aligned via difficulty config, stdin + call-based support, whitespace-normalized, emits per-tier `*_metrics.rescored.json` + `results/scores.rescored.json` for `build_results_table.py`. Generation was never broken, so no regeneration needed for a corrected v1 table.
- [x] **Gate PASSED:** `score_apps.py` base/introductory = **16.0% strict, 34.3% avg** (n=150) vs v1's reported 0.0%/0.84%. Confirms the v1 harness scorer was broken and the base is respectable. (Fixed two issues: single-threaded slowness → now parallel across CPU cores w/ live progress + 4s timeout; and a missing output-dir mkdir.)
- [ ] **Full run:** both models × 3 tiers → `results/scores.rescored.json` → corrected v1 comparison table.
- [ ] Render corrected table: `build_results_table.py --scores results/scores.rescored.json`; update `README.md` + `CLAUDE.md §2` with the TRUE v1 numbers (base and fine-tune both re-scored — v1's medium=7.33%/hard=0% came from the same broken scorer and are also suspect).
- [ ] Then v2-quality eval: regenerate with bf16 + reasoning-prompt + avg@k, score with the same `score_apps.py`; add LiveCodeBench (functional) as co-headline.
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
- 2026-08-17 — Diagnosis complete. Confirmed base = 12% (not 0%) on aligned
  introductory → **v1 harness APPS scorer is broken.** Added `eval/score_apps.py`
  (verified standalone scorer) to re-score saved generations GPU-free. Fine-tune's
  bracket artifact remains real. Next: run score_apps.py, render corrected v1 table.
