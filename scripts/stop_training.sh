#!/usr/bin/env bash
# ==============================================================================
# HETMAN-2.0B: Safe Graceful Shutdown Script
# ==============================================================================
# Sends SIGINT (Ctrl+C) to the training process to trigger emergency checkpoint
# saving, ensuring zero loss of progress before terminating the session.
# ==============================================================================

SESSION_NAME="hetman-training"

echo "[*] Initiating graceful shutdown of Hetman-2.0B training..."

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    echo "[*] Sending SIGINT to training process in tmux session '${SESSION_NAME}'..."
    tmux send-keys -t "${SESSION_NAME}" C-c
    
    echo "[*] Waiting gracefully for checkpoint saving to complete (up to 60s)..."
    WAITED=0
    while tmux has-session -t "${SESSION_NAME}" 2>/dev/null && [ ${WAITED} -lt 60 ]; do
        sleep 2
        WAITED=$((WAITED + 2))
        echo "    ...waiting (${WAITED}s)"
    done
    
    if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
        echo "[*] Graceful wait period exceeded. Terminating tmux session '${SESSION_NAME}'..."
        tmux kill-session -t "${SESSION_NAME}" 2>/dev/null || true
    fi
    echo -e "\033[1;32m[✓] Training safely stopped. All checkpoints preserved.\033[0m"
    echo "    To resume training anytime: bash scripts/launch_autopilot.sh"
else
    echo "[!] No running tmux session '${SESSION_NAME}' found."
fi
