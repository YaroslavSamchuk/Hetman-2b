#!/usr/bin/env bash
# Resumable worker for the local A/B/C data-recipe pilot.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
OUTPUT_DIR="${PILOT_OUTPUT_DIR:-${ROOT_DIR}/runs/pilot_abc}"
MAX_RESTARTS="${PILOT_MAX_RESTARTS:-3}"
PILOT_DEVICE="${PILOT_DEVICE:-cuda}"
LOG_DIR="${ROOT_DIR}/logs"
mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"
if [[ ! "${MAX_RESTARTS}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: PILOT_MAX_RESTARTS must be a non-negative integer." >&2
    exit 2
fi

tokenizer_failures=0
while true; do
    echo "[$(date -Is)] Training or validating the 65,536-entry pilot tokenizer." | tee -a "${LOG_DIR}/pilot_worker.log"
    if "${PYTHON_BIN}" -u src/train_pilot_tokenizer.py 2>&1 | tee -a "${LOG_DIR}/pilot_worker.log"; then
        break
    else
        status=${PIPESTATUS[0]}
    fi
    if (( status == 130 || status == 143 )); then
        echo "[$(date -Is)] Tokenizer training stopped by operator." | tee -a "${LOG_DIR}/pilot_worker.log"
        exit "${status}"
    fi
    tokenizer_failures=$((tokenizer_failures + 1))
    if (( tokenizer_failures > MAX_RESTARTS )); then
        echo "[$(date -Is)] Tokenizer training failed ${tokenizer_failures} times; stopping before shard creation." | tee -a "${LOG_DIR}/pilot_worker.log"
        exit "${status:-1}"
    fi
    delay=$((30 * tokenizer_failures))
    echo "[$(date -Is)] Tokenizer training exited ${status}; retrying in ${delay}s." | tee -a "${LOG_DIR}/pilot_worker.log"
    sleep "${delay}"
done

build_failures=0
while true; do
    echo "[$(date -Is)] Building or validating all pilot shards." | tee -a "${LOG_DIR}/pilot_worker.log"
    if "${PYTHON_BIN}" -u src/build_pilot_datasets.py 2>&1 | tee -a "${LOG_DIR}/pilot_worker.log"; then
        break
    else
        status=${PIPESTATUS[0]}
    fi
    if (( status == 130 || status == 143 )); then
        echo "[$(date -Is)] Dataset build stopped by operator." | tee -a "${LOG_DIR}/pilot_worker.log"
        exit "${status}"
    fi
    build_failures=$((build_failures + 1))
    if (( build_failures > MAX_RESTARTS )); then
        echo "[$(date -Is)] Dataset build failed ${build_failures} times; stopping for operator review." | tee -a "${LOG_DIR}/pilot_worker.log"
        exit "${status:-1}"
    fi
    delay=$((30 * build_failures))
    echo "[$(date -Is)] Dataset build exited ${status}; retrying in ${delay}s." | tee -a "${LOG_DIR}/pilot_worker.log"
    sleep "${delay}"
done

for recipe in recipe_a_baseline recipe_b_cleaned recipe_c_balanced; do
    checkpoint_dir="${OUTPUT_DIR}/${recipe}_checkpoints"
    if ! compgen -G "${checkpoint_dir}/step_*.pt" >/dev/null; then
        acceptance_failures=0
        while true; do
            acceptance_args=(--recipe "${recipe}" --output-dir "${OUTPUT_DIR}" --device "${PILOT_DEVICE}" --max-steps 1 --checkpoint-every 1 --no-eval)
            if compgen -G "${checkpoint_dir}/step_*.pt" >/dev/null; then
                acceptance_args+=(--resume)
            fi
            echo "[$(date -Is)] Running one-step GPU acceptance check for ${recipe}." | tee -a "${LOG_DIR}/pilot_worker.log"
            if "${PYTHON_BIN}" -u src/run_pilot_experiment.py "${acceptance_args[@]}" 2>&1 | tee -a "${LOG_DIR}/pilot_worker.log"; then
                break
            else
                status=${PIPESTATUS[0]}
            fi
            if (( status == 130 || status == 143 )); then
                echo "[$(date -Is)] ${recipe} acceptance check stopped by operator." | tee -a "${LOG_DIR}/pilot_worker.log"
                exit "${status}"
            fi
            acceptance_failures=$((acceptance_failures + 1))
            if (( acceptance_failures > MAX_RESTARTS )); then
                echo "[$(date -Is)] ${recipe} acceptance check failed ${acceptance_failures} times; stopping before full recipe training." | tee -a "${LOG_DIR}/pilot_worker.log"
                exit "${status:-1}"
            fi
            delay=$((30 * acceptance_failures))
            echo "[$(date -Is)] ${recipe} acceptance check exited ${status}; retrying in ${delay}s." | tee -a "${LOG_DIR}/pilot_worker.log"
            sleep "${delay}"
        done
    fi

    failures=0
    while true; do
        args=(--recipe "${recipe}" --output-dir "${OUTPUT_DIR}" --device "${PILOT_DEVICE}")
        if compgen -G "${checkpoint_dir}/step_*.pt" >/dev/null; then
            args+=(--resume)
        fi

        echo "[$(date -Is)] Starting ${recipe}; restart ${failures}/${MAX_RESTARTS}" | tee -a "${LOG_DIR}/pilot_worker.log"
        if "${PYTHON_BIN}" -u src/run_pilot_experiment.py "${args[@]}" 2>&1 | tee -a "${LOG_DIR}/pilot_worker.log"; then
            break
        else
            status=${PIPESTATUS[0]}
        fi
        if (( status == 130 || status == 143 )); then
            echo "[$(date -Is)] ${recipe} stopped by operator after saving its checkpoint." | tee -a "${LOG_DIR}/pilot_worker.log"
            exit "${status}"
        fi
        failures=$((failures + 1))
        if (( failures > MAX_RESTARTS )); then
            echo "[$(date -Is)] ${recipe} failed ${failures} times; stopping for operator review." | tee -a "${LOG_DIR}/pilot_worker.log"
            exit "${status:-1}"
        fi
        delay=$((30 * failures))
        echo "[$(date -Is)] ${recipe} exited ${status}; retrying from its latest checkpoint in ${delay}s." | tee -a "${LOG_DIR}/pilot_worker.log"
        sleep "${delay}"
    done
done

echo "[$(date -Is)] All three data-recipe pilot runs completed." | tee -a "${LOG_DIR}/pilot_worker.log"
