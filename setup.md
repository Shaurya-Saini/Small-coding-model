# SCM — Setup & Reproduction Guide

Concise, ordered steps to reproduce the project from this repo. Training runs on
Kaggle (GPU **T4 x2**); evaluation runs in a separate clean environment.

## Conventions

- In a notebook cell: `!python file.py` runs a Python file, `!bash file.sh` runs a
  shell script. Set env vars inline: `!VAR=x bash file.sh`.
- Get the repo onto the machine with `git clone <repo>` (or upload a zip).

---

## 1. Data prep (CPU, any machine)

```bash
pip install -r requirements/data.txt
python data/prepare_apps.py            # -> data/apps_train.jsonl (APPS TRAIN split only)
```

---

## 2. Colab smoke test (optional, free T4)

Purpose: prove the QLoRA loop before spending Kaggle hours.

1. `Runtime -> Change runtime type -> T4 GPU`.
2. `%pip install -q unsloth`
3. Load `unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit` in 4-bit, attach LoRA.
4. Load ~50 APPS train rows (Parquet), format with the chat template.
5. Train ~20 steps with `SFTConfig` (`dataset_text_field`/`max_length`/`packing`
   live in `SFTConfig`), confirm loss drops and generation works.

---

## 3. Training (Kaggle, GPU T4 x2)

### 3.1 Session
- Right sidebar -> **Accelerator = GPU T4 x2** (NOT P100). **Internet = on**.
- Verify: `import torch; print(torch.cuda.get_device_capability(0))  # (7, 5)`

### 3.2 Install
```bash
pip install -r requirements/train.txt   # unsloth (pulls a compatible torch/transformers/trl set)
```

### 3.3 HF token (write) as a Kaggle Secret
- huggingface.co -> Settings -> Access Tokens -> new **Write** token.
- Kaggle -> Add-ons -> Secrets -> add `HF_TOKEN`.
```python
import os
from kaggle_secrets import UserSecretsClient
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
```

### 3.4 Build data + train
```bash
python data/prepare_apps.py
python training/train_qlora.py \
    --epochs 1 --batch-size 2 --grad-accum 4 \
    --save-steps 50 --save-total-limit 3 \
    --push --merge-16bit \
    --hf-username <user> --hf-repo qwen2.5-coder-7b-apps-qlora
```
Use **Save Version -> Save & Run All (Commit)** to run in the background.

### 3.5 Resume after a disconnect
```bash
python training/train_qlora.py ... --resume   # auto-detects outputs/checkpoint-*
```

### 3.6 Confirm push
Hub should show `…/qwen2.5-coder-7b-apps-qlora` (merged 16-bit) and `…-lora` (adapter).

---

## 4. Evaluation (fresh environment, NO Unsloth)

### 4.1 Install (pins matter — install LAST so nothing upgrades them back)
```bash
pip install -r requirements/eval.txt
git clone https://github.com/bigcode-project/bigcode-evaluation-harness
cd bigcode-evaluation-harness && pip install -e . && pip install -r requirements.txt
pip install "datasets>=2.16,<4.0" "transformers>=4.44,<5.0" bitsandbytes
```

### 4.2 Preflight (fails fast on any version/dep issue)
```bash
python /path/to/SCM/eval/preflight.py     # want: "Preflight PASSED"
```

### 4.3 One-time: repair the fine-tuned tokenizer (transformers 4.x)
```bash
python /path/to/SCM/eval/fix_tokenizer_config.py --repo <user>/qwen2.5-coder-7b-apps-qlora
```

### 4.4 Run APPS eval (from the harness dir with `main.py`)
`run_apps_eval.sh` auto-applies the harness patches (`fix_pyext_py312.py`,
`fix_harness_apps.py`) and defaults to `LOAD_IN=4bit`, `NUM_PROCESSES=1`,
`EOS=<|im_end|>`.
```bash
# smoke (10 problems)
LIMIT=10 MODEL=<user>/qwen2.5-coder-7b-apps-qlora LABEL=finetuned_smoke \
  HARNESS_MAIN=$(pwd)/main.py bash /path/to/SCM/eval/run_apps_eval.sh

# full subset runs (NUM_PROCESSES=2 uses both T4s)
LIMIT=150 NUM_PROCESSES=2 MODEL=<user>/qwen2.5-coder-7b-apps-qlora LABEL=finetuned \
  HARNESS_MAIN=$(pwd)/main.py bash /path/to/SCM/eval/run_apps_eval.sh
LIMIT=150 NUM_PROCESSES=2 MODEL=Qwen/Qwen2.5-Coder-7B-Instruct LABEL=base \
  HARNESS_MAIN=$(pwd)/main.py bash /path/to/SCM/eval/run_apps_eval.sh
```
Metrics land in `results/apps/<label>/<task>_metrics.json`.

### 4.5 Build results table + diagram
Fill `results/scores.json` (copy `results/scores.template.json`) from the
`*_metrics.json` files (`pass@1` = `strict_accuracy`), then:
```bash
python eval/build_results_table.py        # -> results/report.md + results/figures/*.png
```

---

## 5. LiveCodeBench + frontier (deferred to v2)

Clone `LiveCodeBench`, `pip install -e .`, run `eval/run_livecodebench_eval.sh`
(record the release version), and add published leaderboard numbers as the
`frontier` column in `results/scores.json`. Never train on LiveCodeBench.
