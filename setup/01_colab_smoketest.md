# 01 — Colab Smoke Test

> **Status: VERIFIED (2026-08-10).** Ran end-to-end on a free Colab T4 via the
> self-contained `setup/colab_smoketest.ipynb`. All steps below match what
> actually happened; real results are in "Verified run" at the bottom.

## Purpose

Prove the *whole* QLoRA loop runs end-to-end on a tiny slice (2–3 APPS problems,
~10 steps) on a **free Colab T4** before spending real Kaggle hours. This is a
plumbing test, not a training run — we only want to see: model loads in 4-bit,
LoRA attaches, a few optimizer steps run without OOM, and the model still
generates code afterward.

> Colab free gives a **single** T4. The real run uses Kaggle **T4 x2** — but for
> a smoke test one T4 is plenty.

---

## Step 0 — Select the GPU

Colab menu: **Runtime → Change runtime type → Hardware accelerator = T4 GPU**.
Confirm:

```python
import torch
print(torch.cuda.get_device_name(0))              # -> Tesla T4
print(torch.cuda.get_device_capability(0))        # -> (7, 5)  (>= 7.0 required)
```

## Step 1 — Install Unsloth

```python
%pip install -q unsloth
# If Colab complains about a resolver conflict, restart the runtime once and
# re-run this cell. Do NOT hand-pin torch/transformers/peft/trl (see SCM.md §5).
```

## Step 2 — Load the base model in 4-bit

```python
from unsloth import FastLanguageModel
import torch

MAX_SEQ_LEN = 2048
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name    = "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit",  # pre-quantized
    max_seq_length= MAX_SEQ_LEN,
    dtype         = None,          # auto (fp16 on T4)
    load_in_4bit  = True,
)
```

## Step 3 — Attach LoRA adapters

```python
model = FastLanguageModel.get_peft_model(
    model,
    r                          = 16,
    target_modules             = ["q_proj","k_proj","v_proj","o_proj",
                                  "gate_proj","up_proj","down_proj"],
    lora_alpha                 = 16,
    lora_dropout               = 0,
    bias                       = "none",
    use_gradient_checkpointing = "unsloth",
    random_state               = 3407,
)
```

## Step 4 — Grab 2–3 APPS **train** examples (train split ONLY)

> **Fix (verified 2026-08-10):** `datasets` >= 4.0 (what Colab installs today)
> removed dataset *scripts* and `trust_remote_code`, so
> `load_dataset("codeparrot/apps", trust_remote_code=True)` fails with
> *"Dataset scripts are no longer supported"*. Load the Hub's Parquet export
> instead — same data, no script.

```python
from huggingface_hub import HfApi
from datasets import load_dataset
import json

def load_apps_split(split, n=None):
    """Load codeparrot/apps from its auto-generated Parquet export (no script)."""
    rev = "refs/convert/parquet"
    files = [f for f in HfApi().list_repo_files("codeparrot/apps",
             repo_type="dataset", revision=rev) if f.endswith(".parquet")]
    configs = sorted({f.split("/")[0] for f in files})
    config = "all" if "all" in configs else configs[0]
    sel = [f for f in files if f.startswith(config + "/")
           and (f"/{split}/" in f or f"/{split}-" in f or f.endswith(f"/{split}.parquet"))]
    data_files = [f"hf://datasets/codeparrot/apps@{rev}/{f}" for f in sel]
    ds = load_dataset("parquet", data_files=data_files, split="train")
    return ds.select(range(n)) if n else ds

# TRAIN split only. Never load the test split here, and never LiveCodeBench.
raw = load_apps_split("train", n=3)

def build_prompt(ex):
    io = json.loads(ex["input_output"]) if ex["input_output"] else {}
    io_hint = "\nUse Call-Based format\n" if io.get("fn_name") else "\nUse Standard Input format\n"
    sc = f"\n{ex['starter_code']}\n" if ex.get("starter_code","").strip() else "\n"
    return f"QUESTION:\n{ex['question']}\n{sc}{io_hint}ANSWER:\n"

def first_solution(ex):
    sols = json.loads(ex["solutions"]) if ex["solutions"] else []
    return sols[0] if sols else None

def to_text(ex):
    sol = first_solution(ex)
    msgs = [{"role":"user","content":build_prompt(ex)},
            {"role":"assistant","content":sol}]
    return {"text": tokenizer.apply_chat_template(msgs, tokenize=False)}

ds = raw.filter(lambda ex: first_solution(ex) is not None).map(to_text)
print(ds[0]["text"][:800])
```

## Step 5 — Run ~10 training steps

> On the current TRL, `dataset_text_field`/`max_length`/`packing` live in
> `SFTConfig`, not as `SFTTrainer` kwargs (this is the working API on the
> Unsloth 2026.x stack).

```python
from trl import SFTTrainer, SFTConfig

trainer = SFTTrainer(
    model         = model,
    tokenizer     = tokenizer,
    train_dataset = ds,
    args = SFTConfig(
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 4,
        warmup_steps                = 1,
        max_steps                   = 10,          # smoke test only
        learning_rate               = 2e-4,
        logging_steps               = 1,
        optim                       = "adamw_8bit",
        seed                        = 3407,
        output_dir                  = "outputs",
        dataset_text_field          = "text",
        max_length                  = MAX_SEQ_LEN,
        packing                     = False,
        fp16                        = not torch.cuda.is_bf16_supported(),
        bf16                        = torch.cuda.is_bf16_supported(),
        report_to                   = "none",
    ),
)
trainer.train()   # want: loss prints for ~10 steps, no OOM
```

## Step 6 — Confirm it still generates code

```python
FastLanguageModel.for_inference(model)
msgs = [{"role":"user","content":"Write a Python function that returns the nth Fibonacci number."}]
inputs = tokenizer.apply_chat_template(msgs, add_generation_prompt=True,
                                       return_tensors="pt").to("cuda")
out = model.generate(input_ids=inputs, max_new_tokens=256)
print(tokenizer.decode(out[0], skip_special_tokens=True))
```

---

## Success criteria

- [x] `get_device_capability` reports `(7, 5)`.
- [x] Model loads in 4-bit with no OOM.
- [x] LoRA attaches (Unsloth prints trainable-params count).
- [x] ~10 steps run, loss logged, no OOM.
- [x] Post-training generation returns syntactically plausible Python.

## Verified run (2026-08-10, `setup/colab_smoketest.ipynb`)

Free Colab **Tesla T4**, capability `(7, 5)`. Stack: **Unsloth 2026.8.11**,
Transformers 5.5.0, Torch 2.11.0+cu128, Triton 3.6.0, `Bfloat16 = FALSE` (fp16).

- APPS **train** Parquet load: 50 problems, 49 usable after prompt-mask filter
  (1 dropped as all-`-100` from truncation — expected).
- LoRA: **40,370,176 / 7,655,986,688** trainable params (**0.53%**).
- Train: 20 steps, final `train_loss ≈ 0.632`, `train_runtime ≈ 236 s`
  (~12 s/step at effective batch 4), no OOM. `train_on_responses_only` active.
- Generation after training: returned valid Python (recursive Fibonacci).
- Optional HF push: merged 16-bit landed at
  `Shaurya-saini/qwen2.5-coder-7b-apps-qlora-smoketest`. **Note:** the merge step
  re-downloads the full-precision base to merge, then uploads ~15 GB — it took
  **~20 min**. Expect the same tail on the real Kaggle run; it is not a hang.
- Unsloth reported `Num GPUs used = 1` (single T4, as expected on Colab).

## Pitfalls seen

- **2026-08-10 — `RuntimeError: Dataset scripts are no longer supported, but
  found apps.py`.** Cause: `datasets` >= 4.0 dropped loading scripts +
  `trust_remote_code`. Fix: load the Parquet export (`refs/convert/parquet`)
  as in Step 4. Applied to `data/prepare_apps.py` and
  `data/prepare_livecodebench.py` too.
- **TRL API:** `dataset_text_field` / `max_length` / `packing` must be passed via
  `SFTConfig`, not as `SFTTrainer` kwargs, on the current stack (Step 5).
- Steps 0–3 (GPU check, `pip install unsloth`, 4-bit load, LoRA attach) verified
  working on Colab free T4 with Unsloth 2026.8.10 / transformers 5.5.0 /
  torch 2.11.0 / datasets >= 4.0.
- _(record anything new from the next run here)_
