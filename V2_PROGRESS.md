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
| D1 | **Base model** | (a) keep Qwen2.5-Coder-7B-Instruct · (b) DeepSeek-R1-Distill-Qwen-7B | **DECIDED (a):** keep Qwen2.5-Coder-7B-Instruct — cleanest v1→v2 ablation (only the DATA changes: golfed solutions → reasoning traces), same OpenCodeReasoning recipe that took a 7B to ~51% LCB. | `[x]` |
| D2 | **Eval: keep or drop APPS** | (a) demote to sanity · (b) drop · (c) keep as co-headline | **DECIDED (c):** keep APPS (now scored by `score_apps.py`) AND add LCB, as co-headlines. | `[x]` |
| D3 | **Eval precision** | bf16 headline + 4-bit column · vs · 4-bit only | **DECIDED (split):** the APPS v1→v2 continuity table stays **4-bit** (matches the saved v1 generations, apples-to-apples); the **LiveCodeBench** headline + frontier comparison runs **bf16** (don't cripple our model vs full-precision APIs). | `[x]` |
| D4 | **RL: when?** | in v2 · vs · later | **DECIDED:** RL is **v3, conditional** — attempt only if v2's SFT underperforms/plateaus near the distillation ceiling; skippable if v2 succeeds. v2 is SFT-only. Rationale: SFT is capped by the R1 teacher; RL (ground-truth rewards) is the only lever past that ceiling. See `CLAUDE.md §3` v3 subsection. | `[x]` |

**Eval approach (DECIDED):** diagnose v1 first, then rebuild. Outcome: v1 harness
scorer was broken; replaced by `eval/score_apps.py`; v1 numbers corrected (base
16.0/9.3/3.3, fine-tune 0.7/2.0/0.0 strict).

### v2 design notes settled today (2026-08-17)
- **Data = OpenCodeReasoning (Python).** Pick a session-sized subset (target ~10–20k
  traces to fit a free T4); each example rendered as `problem → <think>reasoning</think>
  → clean solution` inside Qwen's chat template. Prompt-mask everything before the
  assistant turn. Keep `parse_int=str` guard from v1.
- **Long-sequence cost:** reasoning traces are long → need larger `max_seq_len`
  (~4–8k) than v1, which raises VRAM/time on a T4. Budget for it (fewer steps, grad
  accumulation, aggressive checkpointing).
- **Eval implication of reasoning:** at eval time the v2 model EMITS long `<think>`
  before code → generation needs a large `max_new_tokens`, and code extraction must
  take the **last** ```python fence after the reasoning. `score_apps.py` scores
  already-extracted code, so this lives in the generation/postprocess step.
- **Firewall unchanged:** OpenCodeReasoning (train) ≠ APPS-test (eval) ≠ LiveCodeBench
  (eval only, never trained on).

---

## Phase 0 — Analysis & planning `[x]`
- [x] Survey base-model CP performance, existing projects, RL techniques, datasets/benchmarks (2026-08-17 web survey → `CLAUDE.md §3`)
- [x] Diagnose v1 eval as a first-class bottleneck (base 0% on APPS-introductory red flag)
- [x] Update `CLAUDE.md §3` with v2 research + plan + eval overhaul
- [x] Resolve D1–D4 (all decided 2026-08-17 — see table)
- [x] Correct & re-publish v1 results (scores.json, report.md, figures, README, CLAUDE.md, setup.md)

## Phase 1 — Data (fix root cause #1) `[~]`  ← code done; validate in the Kaggle trial
- [x] Dataset chosen: **OpenCodeReasoning** (Python reasoning traces); base stays Qwen2.5-Coder-7B-Instruct (D1)
- [x] **Decided: config `split_0` only** (question inline via `input`; no APPS/TACO join, unlike `split_1`)
- [x] **REVISED after trial: subset ~3000 examples @ `max_seq_len` 4096** (was 10k@8192). Trial showed Unsloth free uses **1 T4 only** (~22 s/ex @ 8192) → 10k@8192 ≈ 62h, infeasible; 3k@4096 ≈ 8–9h fits one session.
- [x] **REVISED after trial: firewall = `split=='train'` only + empty `EXCLUDE_SOURCES` = ALL split_0 platforms, balanced by per-source cap.** Verified split_0 has **no apps/taco** (they're in split_1); its sources are live platforms (codeforces/code_contests/atcoder/codechef/aizu/hackerearth/…). APPS-test firewall now automatic; LCB firewall deferred to eval-window choice (Phase 3).
- [x] Wrote + **debugged** `data/prepare_reasoning_traces.py`. Trial exposed 3 bugs (all fixed): (1) old allow-list {apps,taco,code_contests,codeforces} ∩ split_0 = only code_contests → mono-source output; (2) per-source cap checked AFTER the expensive regex parse → wasted the scan budget on 374k over-cap rows; (3) `--max-scan` too low to reach late-in-stream platforms. Now: cheap firewall+cap BEFORE parse, `--max-scan` 1.5M, deny-list instead of allow-list.
- [x] Format = `problem → <think> trace </think> → clean solution` written to `data/reasoning_train.jsonl` as `{prompt, response}`; training masks everything before the assistant turn
- [x] Firewall + parse split into `firewall_source()` (cheap) / `parse_row()` (expensive); unit-tested offline (test/valid-split, empty-source, no-think, no-input all skip; codeforces/atcoder/train kept)
- [ ] **Kaggle re-run (next):** run fixed prep, confirm `sources={…}` shows a spread (not 100% code_contests) and reaches ~3000; inspect a few rendered rows
- [x] Three-dataset firewall preserved: separate script/output; APPS-test & LiveCodeBench never touched

## Phase 2 — Training (fix root cause #2) `[~]`  ← code done; validate in the Kaggle trial
- [x] Adapted `training/train_qlora.py` (same Unsloth+QLoRA+SFTTrainer stack; configurable `--prompt-field`/`--response-field`; `train_on_responses_only` masks the prompt so loss covers the reasoning + code)
- [x] Anti-forgetting defaults: `--lr 1e-4`, 1 epoch; long-seq T4 defaults `--max-seq-len 8192 --batch-size 1 --grad-accum 8` (effective batch 8)
- [x] Aggressive checkpointing carried over (save_steps/save_total_limit, `--resume`); **resume to be re-tested in the Kaggle trial**
- [x] Push path updated to a **distinct** v2 repo `qwen2.5-coder-7b-ocr-qlora` (v1 model not overwritten); merged-16bit + adapter + tokenizer push logic unchanged (see `CLAUDE.md §8`)
- [x] **Kaggle smoke PASSED:** loaded 4-bit, `train_on_responses_only: enabled`, loss ~1.15→~1.0 over 15 steps, checkpoints written. Confirmed **1 GPU used** (Unsloth free) and **178 s/step** @ 8192 → drove the 4096/3000 revision above.
- [ ] **Kaggle full run (next):** with the fixed data (~3000 @ 4096), run `--epochs 1 --push --merge-16bit`; verify checkpoint/resume once mid-run

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
- [x] **Full run done (both models × 3 tiers, corrected):** base 16.0/9.3/3.3 strict (34.3/36.3/16.2 avg); fine-tune 0.7/2.0/0.0 (3.1/6.8/1.3). All tiers `call-based=0` → all stdin, fully trusted path. **Base beats fine-tune on every tier & both metrics, at correct magnitudes.** (Parallel scorer: base introductory ~fast now; fine-tune tiers fast since most don't compile.)
- [x] **Corrected results committed to repo (done locally from pasted numbers, incl. figures — matplotlib is local):** updated `results/scores.json`, regenerated `results/report.md` + `results/figures/apps.png`/`apps_avg.png`, updated `CLAUDE.md §2` and `README.md` (Results + a new "the harness can be the bug" conclusion). Old bogus numbers preserved as record in `CLAUDE.md`.
- [x] **APPS scorer replaced, not patched:** `score_apps.py` supersedes the broken bigcode APPS scoring path entirely (no need to fix the harness's comparison/prompt).
- [ ] Then v2-quality eval: regenerate with bf16 + reasoning-prompt + avg@k, score with the same `score_apps.py`; add LiveCodeBench (functional) as co-headline.
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
- 2026-08-17 — **Corrected v1 eval published.** Full re-score: base 16.0/9.3/3.3 vs
  fine-tune 0.7/2.0/0.0 (strict). `score_apps.py` parallelized + mkdir fix. Updated
  scores.json/report.md/figures/CLAUDE.md/README.md locally. **Diagnosis+correction
  phase COMPLETE.**
- 2026-08-17 (session wrap) — **All decisions closed (D1–D4).** D1: keep Qwen2.5-
  Coder-7B-Instruct. D3: APPS table 4-bit / LCB headline bf16. D4: SFT-only primary.
  Doc consistency pass done: README got a dedicated "evaluation-harness bug" section
  + corrected observations/conclusions/next-steps; setup.md eval steps rewritten to
  use `score_apps.py`; v2 design notes recorded. **NEXT SESSION: Phase 1** — pick
  OpenCodeReasoning subset size + write `data/prepare_reasoning_traces.py`.
- 2026-08-18 — **Repo reorganized:** v1 frozen into `Version 1.0/` (proof of work +
  self-contained reproduction, own README); repo root is now the v2 dev tree; root
  `README.md` rewritten as combined v1+v2 documentation.
- 2026-08-18 — **Phase 1 + Phase 2 code complete (pending one Kaggle trial).** Three
  data decisions taken with the user: (1) OCR config `split_0` only; (2) firewall =
  `split=='train'` + sources {apps,taco,code_contests,codeforces}; (3) ~10k examples
  @ seq_len 8192. Wrote `data/prepare_reasoning_traces.py` (streaming loader, firewall,
  `<think>`+fenced-code reconstruction, exact-tokenizer length filter) — parsing/firewall
  unit-tested offline. Adapted `training/train_qlora.py` to v2 (`{prompt,response}`,
  seq_len 8192, batch 1 / grad-accum 8, lr 1e-4, distinct v2 Hub repo). Updated
  `requirements/data.txt` (+transformers), `data/README.md`, `setup.md` (Kaggle
  git-clone flow). **NEXT: single Kaggle T4 x2 trial** — data prep → inspect rows →
  training smoke → (if green) full run + push. Then Phase 3 eval overhaul.
- 2026-08-18 — **2nd Kaggle run → split_0 is code_contests-only for train.** A full
  567,850-row scan showed the `split=='train'` partition yields ONLY `code_contests`
  (DeepMind CodeContests aggregate); the HF-stats hint of diverse platforms was
  wrong (other platforms are *-valid/*-test → firewalled). "All-platforms balanced"
  is impossible from split_0. Fix: `--per-source-frac` default → **1.0** (no
  balancing); take ~3000 from code_contests (collection stops early → fast). Data
  is clean (0 too_long, 0 no-reasoning in the sample; ~92 s/step @ 4096). True
  multi-dataset diversity (apps/taco) → **v2.1** via the split_1 join. Docs
  (setup/data-README/CLAUDE) corrected. **NEXT: re-run prep (expect 3000), full
  training + push.**
- 2026-08-18 — **Kaggle trial run → plan revised (still Phase 1/2).** Smoke training
  PASSED (loss drops, checkpoints write) but exposed hard constraints and data bugs:
  (a) **Unsloth free = 1 T4 only**, 178 s/step @ 8192 → 10k@8192 ≈ 62h infeasible →
  **revised to ~3000 examples @ 4096** (≈8–9h). (b) **split_0 has no apps/taco**
  (verified via HF stats) and is source-clustered → the old allow-list yielded 100%
  code_contests → **revised to all-platforms + per-source balancing**, deny-list
  empty. (c) Fixed the sampler: cheap firewall/cap BEFORE the expensive parse,
  `--max-scan` 1.5M. Both user decisions (compute plan; source mix) taken. Updated
  prep + train defaults + all docs. **NEXT: re-run prep (fixed) + full training on
  Kaggle**, then Phase 3.
