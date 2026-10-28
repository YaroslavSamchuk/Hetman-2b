#!/usr/bin/env bash
# Start the resumable A/B/C pilot on a provisioned Linux training node.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
SESSION_NAME="hetman-mvp-pilot"
LOG_DIR="${ROOT_DIR}/logs"
mkdir -p "${LOG_DIR}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "ERROR: Python executable not found: ${PYTHON_BIN}" >&2
    exit 2
fi
if ! command -v tmux >/dev/null 2>&1; then
    echo "ERROR: tmux is required for a detached session. Install it on the training node first." >&2
    exit 2
fi
if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    echo "Pilot is already running in tmux session '${SESSION_NAME}'."
    echo "Attach with: tmux attach -t ${SESSION_NAME}"
    exit 0
fi

"${PYTHON_BIN}" -c 'import numpy, torch, tokenizers, pyarrow; print("Python ML dependencies: available"); print("PyTorch:", torch.__version__, "CUDA:", torch.cuda.is_available()); raise SystemExit(0 if torch.cuda.is_available() else "CUDA GPU required for the full pilot; install a CUDA-enabled PyTorch build or use a smaller local smoke test.")'
"${PYTHON_BIN}" src/prepare_raw_corpora.py audit --strict

printf -v worker_cmd 'cd %q && bash %q' "${ROOT_DIR}" "${SCRIPT_DIR}/run_pilot_worker.sh"
tmux new-session -d -s "${SESSION_NAME}" "${worker_cmd}"

echo "Started the resumable A/B/C data-recipe pilot in tmux '${SESSION_NAME}'."
echo "Dataset preparation and recipe training run sequentially; progress: ${LOG_DIR}/pilot_worker.log"
echo "Attach: tmux attach -t ${SESSION_NAME}"
echo "Stop gracefully with Ctrl+C in the attached session; the latest checkpoint is saved on interruption."
