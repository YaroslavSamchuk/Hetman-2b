#!/usr/bin/env bash
# ==============================================================================
# HETMAN-2.0B: Live Training Status & Progress Monitor
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
STATUS_FILE="${ROOT_DIR}/logs/STATUS.txt"
LOG_FILE="${ROOT_DIR}/logs/training.log"
SESSION_NAME="hetman-training"

if [ -f "${STATUS_FILE}" ]; then
    cat "${STATUS_FILE}"
else
    echo "================================================================================"
    echo "                      HETMAN-2.0B TRAINING STATUS MONITOR                       "
    echo "================================================================================"
    echo " [!] STATUS.txt not found yet. Has training been started with launch_autopilot.sh?"
fi

echo ""
if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    echo -e " [Process Status]: \033[1;32mRUNNING (tmux session: ${SESSION_NAME})\033[0m"
else
    echo -e " [Process Status]: \033[1;31mNOT RUNNING (tmux session not found)\033[0m"
fi

if [ -f "${LOG_FILE}" ]; then
    echo "================================================================================"
    echo " Recent Log Entries (Last 10 lines):"
    echo "--------------------------------------------------------------------------------"
    tail -n 10 "${LOG_FILE}"
    echo "================================================================================"
fi
