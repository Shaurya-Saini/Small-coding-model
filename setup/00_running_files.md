# 00 — How to run this project's files (Colab & Kaggle)

Short answer to "how do I run the `.py` / `.sh` files in a notebook?": in a
notebook cell, a line starting with `!` runs a **shell command**, so you run a
Python file with `!python path/to/file.py` and a shell script with
`!bash path/to/file.sh`. You do **not** import them. The only prerequisite is
getting the files onto the machine first (three ways below).

There are also two ways to work:
- **Self-contained notebook** (easiest for the smoke test): all code is *inside*
  notebook cells — no repo files needed. Use `setup/colab_smoketest.ipynb`.
- **Run the repo scripts** (for the real Kaggle run and eval): get the files onto
  the machine, then `!python …` / `!bash …`.

---

## A. Notebook cell basics

| You want to… | Put this in a cell |
|---|---|
| Run Python inline | just write Python (no `!`) |
| Run a shell command | `!ls`, `!pip install foo`, `!python train.py` |
| Run a shell script | `!bash eval/run_apps_eval.sh` |
| Set env vars for one command | `!MODEL=... LABEL=base bash eval/run_apps_eval.sh` |
| Set env vars for the whole session | `import os; os.environ["HF_TOKEN"]="..."` |
| Write a file from a cell | put `%%writefile path.py` as the first line |

`.sh` files are **shell scripts** — a list of terminal commands. Ours just wrap
long `accelerate launch …` / `python -m lcb_runner …` commands so you don't
retype them. They run on Linux (Colab/Kaggle are Linux), via `!bash file.sh`,
and read settings from environment variables you set on the same line.

---

## B. Three ways to get the repo files onto Colab/Kaggle

### Way 1 — Upload a zip (simplest, no git needed)

On your PC, zip the `SCM` folder. Then in Colab:

```python
from google.colab import files
up = files.upload()          # pick SCM.zip in the dialog
!unzip -q SCM.zip -d /content/
%cd /content/SCM
!ls
```

On Kaggle: use the right sidebar **+ Add Data → Upload** a dataset (your zip),
then `!unzip` it into `/kaggle/working`.

### Way 2 — Create the files from cells with `%%writefile`

If you don't want to upload, paste each file into its own cell prefixed with
`%%writefile`. Example:

```python
%%writefile data/prepare_apps.py
# ...paste the full file contents here...
```

(First make the folders: `!mkdir -p data training eval results setup`.)

### Way 3 — git clone (only if you later put this on GitHub)

```python
!git clone https://github.com/<you>/SCM.git
%cd SCM
```

We haven't set up git yet, so use Way 1 or 2 for now.

---

## C. Running each phase

Once the files are present and you're `cd`'d into the repo root:

```python
# Phase 1 — build the training data (APPS train split only)
!pip install -q -r requirements-data.txt
!python data/prepare_apps.py --max-samples 200      # small; drop the flag for full

# Phase 3 — train (smoke first, then real)
!pip install -q unsloth
!python training/train_qlora.py --max-samples 20 --max-steps 10 --save-steps 5
# real-ish (after smoke is clean), with HF_TOKEN set (see setup/02):
!python training/train_qlora.py --epochs 1 --push --merge-16bit
```

```python
# Phase 6 — eval (in a SEPARATE, clean notebook; see setup/03)
# these .sh files run the harnesses; set MODEL + LABEL on the same line:
!MODEL=Shaurya-saini/qwen2.5-coder-7b-apps-qlora LABEL=finetuned bash eval/run_apps_eval.sh
!MODEL=Qwen/Qwen2.5-Coder-7B-Instruct           LABEL=base      bash eval/run_apps_eval.sh
# then fill results/scores.json and:
!python eval/build_results_table.py
```

> Passing flags: `!python file.py --flag value` is exactly the same as typing it
> in a terminal. `--help` on any of our scripts lists every flag:
> `!python training/train_qlora.py --help`.

---

## D. Which environment runs what

- **Colab (free T4)** → the smoke test only (`setup/colab_smoketest.ipynb`).
- **Kaggle (T4 x2)** → the real training run (`setup/02_kaggle_training.md`).
- **A fresh eval notebook** (Colab or Kaggle) → the `eval/*.sh` scripts
  (`setup/03_evaluation_environment.md`). Never install Unsloth here.
