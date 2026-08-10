# 02 — Kaggle Training Run

> **Status: VERIFIED (2026-08-11).** v1 ran end-to-end on Kaggle: 620 steps /
> 1 epoch, ~2h34m, final train loss **0.6514**, model pushed to
> `Shaurya-saini/qwen2.5-coder-7b-apps-qlora`. Real timings and the two bugs hit
> are in "Verified run" at the bottom.

The real QLoRA fine-tune runs on **Kaggle Notebooks** (free, supports background
execution so it keeps running after you close the tab). Colab is only for the
smoke test in `01_colab_smoketest.md`.

---

## Step 0 — The one setting you must not miss: GPU = T4 x2

Notebook right sidebar → **Session options → Accelerator → `GPU T4 x2`**.
**Not** the default P100 — Unsloth needs compute capability ≥ 7.0; P100 is 6.0
and is documented to fail. Confirm in a cell:

```python
import torch
print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))  # Tesla T4 (7, 5)
```

Also enable **Internet** (Session options → Internet on) so pip and the Hub work.

## Step 1 — Get the code + install

Either clone your repo or upload `data/prepare_apps.py` and
`training/train_qlora.py`. Then:

```python
%pip install -q unsloth
# Do NOT hand-pin torch/transformers/peft/trl (SCM.md §5). If the resolver
# conflicts, restart the kernel once and re-run.
```

## Step 2 — HF token as a Kaggle Secret (needed only to push the model)

1. huggingface.co → Settings → **Access Tokens** → **Create new token** → role
   **Write** → copy.
2. Kaggle notebook → **Add-ons → Secrets → Add secret**: name `HF_TOKEN`, paste
   the value, and attach it to the notebook.
3. Load it into the environment:

```python
import os
from kaggle_secrets import UserSecretsClient
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
```

## Step 3 — Build the training data (APPS train split only)

```python
!python data/prepare_apps.py          # -> data/apps_train.jsonl
```

(~1.3 GB download; a few minutes. The APPS *test* split is not touched.)

## Step 4 — Kick off the run

Interactive first-check, then let it run:

```python
!python training/train_qlora.py \
    --epochs 1 --batch-size 2 --grad-accum 4 \
    --save-steps 50 --save-total-limit 3 \
    --push --merge-16bit \
    --hf-username Shaurya-saini --hf-repo qwen2.5-coder-7b-apps-qlora
```

Then use **Save Version → Save & Run All (Commit)** to run it in the
**background** — it survives closing the tab. Check progress under the
notebook's **Versions** tab / logs.

## Step 5 — If the session dies mid-run: resume

Checkpoints land in `outputs/checkpoint-*` every `--save-steps`. Re-run the same
command with `--resume` added; it auto-detects the last checkpoint.

```python
!python training/train_qlora.py ... --resume
```

> On Kaggle, `outputs/` under `/kaggle/working` persists across a committed run.
> For extra safety on very long runs, periodically copy checkpoints to a Kaggle
> Dataset or push adapter snapshots to the Hub.

## Step 6 — Confirm the push

After success you should see, on huggingface.co/Shaurya-saini:
- `qwen2.5-coder-7b-apps-qlora-lora` (the adapter)
- `qwen2.5-coder-7b-apps-qlora` (merged 16-bit — what the eval env pulls)

---

## Success criteria

- [ ] `(7, 5)` reported for device capability (T4 x2 selected).
- [ ] `apps_train.jsonl` built from the **train** split only.
- [ ] Training logs loss; checkpoints appear in `outputs/`.
- [ ] Merged model + adapter visible on the Hub under `Shaurya-saini/…`.

## Verified run (2026-08-11)

Workflow used: cloned the GitHub repo into `/kaggle/working`, `git pull` for
fixes, `%pip install -q unsloth`, HF token via `kaggle_secrets`.

- **Data:** `python data/prepare_apps.py` → `data/apps_train.jsonl` from the full
  APPS **train** split (5000 problems, 1 solution each).
- **Training:** `--epochs 1 --batch-size 2 --grad-accum 4 --save-steps 50
  --save-total-limit 3 --push --merge-16bit`. **620 steps**, **~2h34m**
  (~11 s/step), final `train_loss = 0.6514`, single T4 used (`Num GPUs = 1`).
- **Output:** merged 16-bit `Shaurya-saini/qwen2.5-coder-7b-apps-qlora` +
  adapter `…-lora`. The merge/upload tail took the expected ~20 min.

### Bugs hit and fixed (both now in the code)

1. **`ValueError: Exceeds the limit (4300 digits) for integer string conversion`**
   during `prepare_apps.py`. Cause: Python 3.12's int-string guard vs. APPS test
   cases with thousands-of-digits integers. Fix: `json.loads(raw, parse_int=str)`
   in `_safe_json` (we only read `fn_name`, never the values as ints).
2. **`TypeError: unsloth_push_to_hub() takes 2 positional arguments...`** at the
   push step — *after training had fully completed*. Cause: Unsloth's
   `push_to_hub` takes only the repo id positionally, not the tokenizer. Fix:
   push the tokenizer with its own `tokenizer.push_to_hub(...)` call. **Recovery
   without retraining:** the adapter was already saved to `outputs/final_adapter`,
   so we just re-ran the corrected push (model/tokenizer still in memory).

### Pitfalls / notes

- The push is the *last* step and runs after ~2.5h of training — a bug there
  wastes the whole session if you can't recover. The saved `outputs/final_adapter`
  is the safety net; you can reload it and push without retraining.
- Unsloth (free) uses **one** GPU even on T4 x2 — pick T4 x2 anyway (avoids P100,
  gives compute capability 7.5); timings above are single-GPU.
