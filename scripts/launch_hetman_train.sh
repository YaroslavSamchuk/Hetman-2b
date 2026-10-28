#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SESSION_NAME="${HETMAN_TRAIN_SESSION:-hetman-full-train}"
TRAIN_SHARD="${HETMAN_TRAIN_SHARD:-}"
VAL_SHARD="${HETMAN_VAL_SHARD:-}"
TOKENIZER="${HETMAN_TOKENIZER:-${ROOT_DIR}/data/processed/tokenizer/tokenizer.json}"
OUTPUT_DIR="${HETMAN_OUTPUT_DIR:-${ROOT_DIR}/runs/full-training}"
MODEL_CONFIG="${HETMAN_MODEL_CONFIG:-}"
LOG_DIR="${ROOT_DIR}/logs"
LOG_FILE="${LOG_DIR}/hetman_train.log"

if [[ -z "${TRAIN_SHARD}" || -z "${VAL_SHARD}" ]]; then
  echo "Set HETMAN_TRAIN_SHARD and HETMAN_VAL_SHARD to shard base paths (without .bin/.idx)." >&2
  exit 2
fi
for shard_base in "${TRAIN_SHARD}" "${VAL_SHARD}"; do
  [[ -f "${shard_base}.bin" && -f "${shard_base}.idx" ]] || {
    echo "Shard pair is missing: ${shard_base}.bin/.idx" >&2
    exit 2
  }
done
[[ -f "${TOKENIZER}" ]] || { echo "Tokenizer not found: ${TOKENIZER}" >&2; exit 2; }
if [[ -n "${MODEL_CONFIG}" && ! -f "${MODEL_CONFIG}" ]]; then
  echo "Model config not found: ${MODEL_CONFIG}" >&2
  exit 2
fi
command -v tmux >/dev/null || { echo "tmux is required to run detached." >&2; exit 2; }
command -v flock >/dev/null || { echo "flock is required to prevent concurrent writers to one checkpoint directory." >&2; exit 2; }
if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "A training session named '${SESSION_NAME}' already exists." >&2
  exit 2
fi

"${PYTHON_BIN}" -c 'import torch, tokenizers; assert torch.cuda.is_available(), "CUDA is not available"; print("GPU:", torch.cuda.get_device_name(0)); print("PyTorch:", torch.__version__)'
mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"

worker=("${PYTHON_BIN}" -u "${ROOT_DIR}/src/train_hetman.py"
  --train-shard "${TRAIN_SHARD}"
  --val-shard "${VAL_SHARD}"
  --tokenizer "${TOKENIZER}"
  --output-dir "${OUTPUT_DIR}"
  --device cuda
  --precision "${HETMAN_PRECISION:-bf16}"
  --epochs "${HETMAN_EPOCHS:-1}"
  --chunk-tokens "${HETMAN_CHUNK_TOKENS:-256}"
  --eval-every "${HETMAN_EVAL_EVERY:-100}"
  --checkpoint-every "${HETMAN_CHECKPOINT_EVERY:-100}"
)
if [[ -n "${MODEL_CONFIG}" ]]; then worker+=(--model-config "${MODEL_CONFIG}"); fi
if [[ -n "${HETMAN_MAX_STEPS:-}" ]]; then worker+=(--max-steps "${HETMAN_MAX_STEPS}"); fi
if [[ "${HETMAN_RESUME:-0}" == "1" ]]; then worker+=(--resume); fi

printf -v worker_cmd '%q ' "${worker[@]}"
printf -v log_q '%q' "${LOG_FILE}"
printf -v output_q '%q' "${OUTPUT_DIR}"
tmux new-session -d -s "${SESSION_NAME}" "cd $(printf '%q' "${ROOT_DIR}") && set -o pipefail && flock -n ${output_q}/.train.lock ${worker_cmd} 2>&1 | tee -a ${log_q}"
echo "Started Hetman single-node training in tmux session '${SESSION_NAME}'."
echo "Logs: ${LOG_FILE}"
echo "Resume an interrupted run by rerunning with HETMAN_RESUME=1."
