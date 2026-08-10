# data/ — dataset preparation

Two scripts, kept **deliberately separate** so training and evaluation data can
never mix (see `CLAUDE.md` rules #1 and #2).

| Script | Dataset | Purpose | Feeds training? |
|---|---|---|---|
| `prepare_apps.py` | `codeparrot/apps` (**train** split) | Format prompt→solution pairs for QLoRA SFT | **Yes** |
| `prepare_livecodebench.py` | `livecodebench/code_generation_lite` | Download + manifest only | **Never** |

The APPS **test** split is intentionally *not* processed by any script here — it
is left untouched for the internal held-out check, which
bigcode-evaluation-harness runs directly against the APPS test split in Phase 6.

## Run (data-prep env only — CPU, no GPU)

```bash
pip install -r ../requirements-data.txt

# Training corpus (APPS train split -> data/apps_train.jsonl)
python prepare_apps.py

# Eval sanity/caching (LiveCodeBench -> data/livecodebench/manifest.json)
python prepare_livecodebench.py --version release_v5
```

## Outputs (git-ignored — regenerate, don't commit)

- `apps_train.jsonl` — SFT records: `{problem_id, difficulty, io_format, prompt, solution}`
- `livecodebench/manifest.json` — metadata only (ids, difficulty, date); **no**
  problem statements or solutions.

Difficulty tiers are normalized to `easy | medium | hard`
(APPS `introductory | interview | competition`).
