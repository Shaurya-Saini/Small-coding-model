#!/usr/bin/env bash
# run_livecodebench_eval.sh -- pass@1 on LiveCodeBench via its own lcb_runner.
# EVAL ONLY. LiveCodeBench is never trained on (CLAUDE.md rule #1).
#
# Run in the CLEAN eval environment, after:
#   git clone https://github.com/LiveCodeBench/LiveCodeBench
#   cd LiveCodeBench && pip install -e .
#
# lcb_runner loads the dataset itself and does the "generate -> execute against
# hidden tests -> score" loop. Difficulty splits (easy/medium/hard) are then
# joined from data/livecodebench/manifest.json in build_results_table.py, using
# the version recorded there.
#
# Usage:
#   MODEL=Shaurya-saini/qwen2.5-coder-7b-apps-qlora LABEL=finetuned ./run_livecodebench_eval.sh
#   MODEL=Qwen/Qwen2.5-Coder-7B-Instruct           LABEL=base      ./run_livecodebench_eval.sh
set -euo pipefail

MODEL="${MODEL:?set MODEL to an HF model id}"
LABEL="${LABEL:?set LABEL, e.g. base | finetuned}"
RELEASE="${RELEASE:-release_v5}"      # MUST match the version recorded in CLAUDE.md
N_SAMPLES="${N_SAMPLES:-1}"           # 1 -> pass@1
BACKEND="${BACKEND:-vllm}"            # vllm (fast) or hf

OUT_DIR="results/livecodebench/${LABEL}"
mkdir -p "${OUT_DIR}"

# NOTE (finalize in setup.md): a custom HF model must
# be registered with lcb_runner. Depending on the lcb_runner version this is
# either adding an entry to lcb_runner/lm_styles.py or passing a local/custom
# model flag. Confirm the exact flag before the real run.
echo "=== LiveCodeBench :: ${LABEL} :: ${MODEL} :: ${RELEASE} ==="
python -m lcb_runner.runner.main \
  --model "${MODEL}" \
  --scenario codegeneration \
  --release_version "${RELEASE}" \
  --n "${N_SAMPLES}" \
  --evaluate \
  --use_cache

echo "Done. Copy lcb_runner's output JSON into ${OUT_DIR}/ and run"
echo "eval/build_results_table.py (it joins difficulty from the manifest)."
