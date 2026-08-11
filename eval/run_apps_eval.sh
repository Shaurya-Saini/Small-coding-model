#!/usr/bin/env bash
# run_apps_eval.sh -- pass@k on the APPS TEST split via bigcode-evaluation-harness,
# stratified by difficulty tier. Run this in the CLEAN eval environment (never
# the training env -- see SCM.md §3), after `pip install -e .` inside
# bigcode-evaluation-harness.
#
# APPS ships its own hidden test cases; the harness executes generated code
# against them and reports pass@k. We keep the per-problem timeout on (Docker
# sandbox is unavailable in notebooks -- SCM.md §5).
#
# Usage:
#   MODEL=Shaurya-saini/qwen2.5-coder-7b-apps-qlora LABEL=finetuned ./run_apps_eval.sh
#   MODEL=Qwen/Qwen2.5-Coder-7B-Instruct           LABEL=base      ./run_apps_eval.sh
#
# Run once per model (base and fine-tuned). Outputs one metrics JSON per tier
# under results/apps/<label>/.
set -euo pipefail

# bigcode-evaluation-harness loads codeparrot/apps via its loading SCRIPT and does
# NOT pass trust_remote_code. On datasets >= 4.0 scripts are gone entirely (the
# eval env must pin datasets < 4.0 -- see setup/03); on datasets 2.16..3.x a
# script needs trust to run, which this env var grants without editing the
# harness. Harmless on older datasets.
export HF_DATASETS_TRUST_REMOTE_CODE=1

MODEL="${MODEL:?set MODEL to an HF model id}"
LABEL="${LABEL:?set LABEL, e.g. base | finetuned}"
N_SAMPLES="${N_SAMPLES:-1}"           # 1 -> pass@1; raise (e.g. 20) for pass@k
TEMPERATURE="${TEMPERATURE:-0.2}"     # use ~0.6-0.8 when N_SAMPLES>1
MAX_GEN_LEN="${MAX_GEN_LEN:-2048}"
BATCH_SIZE="${BATCH_SIZE:-1}"
PRECISION="${PRECISION:-fp16}"        # T4 has no bf16 -> fp16 (ignored when quantized)
LOAD_IN="${LOAD_IN:-4bit}"            # 4bit | 8bit | none. A 7B in 16-bit (~15GB) OOMs a
                                      # 16GB T4 (CUBLAS_STATUS_ALLOC_FAILED); 4-bit (~4.5GB)
                                      # leaves room to generate. Each GPU gets its own replica.
NUM_PROCESSES="${NUM_PROCESSES:-}"    # blank = auto (all GPUs). Set 1 to force single-GPU.
LIMIT="${LIMIT:-}"                    # blank = full tier. Set e.g. 10 for a quick sanity pass.
HARNESS_MAIN="${HARNESS_MAIN:-main.py}"   # path to bigcode-evaluation-harness/main.py

# 7B in 16-bit does not fit a T4 with room to generate -> quantize for eval.
QUANT_FLAG=()
case "${LOAD_IN}" in
  4bit) QUANT_FLAG=(--load_in_4bit) ;;
  8bit) QUANT_FLAG=(--load_in_8bit) ;;
  none) QUANT_FLAG=() ;;
  *) echo "LOAD_IN must be one of: 4bit | 8bit | none" >&2; exit 1 ;;
esac

# Optional accelerate launch args (force single-GPU) and problem-count cap.
LAUNCH_ARGS=()
[ -n "${NUM_PROCESSES}" ] && LAUNCH_ARGS=(--num_processes "${NUM_PROCESSES}")
LIMIT_FLAG=()
[ -n "${LIMIT}" ] && LIMIT_FLAG=(--limit "${LIMIT}")

# APPS difficulty-specific task names. VERIFY against `python $HARNESS_MAIN --help`
# (older/newer harness versions have used hyphens vs underscores).
TASKS=("apps-introductory" "apps-interview" "apps-competition")

OUT_DIR="results/apps/${LABEL}"
mkdir -p "${OUT_DIR}"

for TASK in "${TASKS[@]}"; do
  echo "=== ${LABEL} :: ${TASK} :: n_samples=${N_SAMPLES} :: load_in=${LOAD_IN} ${LIMIT:+:: limit=${LIMIT}} ==="
  accelerate launch "${LAUNCH_ARGS[@]}" "${HARNESS_MAIN}" \
    --model "${MODEL}" \
    --tasks "${TASK}" \
    --n_samples "${N_SAMPLES}" \
    --temperature "${TEMPERATURE}" \
    --max_length_generation "${MAX_GEN_LEN}" \
    --batch_size "${BATCH_SIZE}" \
    --precision "${PRECISION}" \
    "${QUANT_FLAG[@]}" \
    "${LIMIT_FLAG[@]}" \
    --allow_code_execution \
    --save_generations \
    --save_generations_path "${OUT_DIR}/${TASK}_generations.json" \
    --metric_output_path "${OUT_DIR}/${TASK}_metrics.json"
done

echo "Done. Metrics in ${OUT_DIR}/. Feed them into eval/build_results_table.py."
