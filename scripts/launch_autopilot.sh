#!/usr/bin/env bash
# ==============================================================================
# HETMAN-2.0B: Cloud TPU v5e-128 Master Autopilot Launcher
# ==============================================================================
# Run this script once on your TPU VM and leave it running for ~16 days.
# When it finishes, you will find fully trained model weights and the optimized
# single-file .rada container ready for immediate local deployment.
#
# Features:
#   1. Automatic TPU Environment & Dependency Provisioning (JAX, Flax, libtpu)
#   2. Full 4-Phase Curriculum (Pre-training 4k -> YaRN 256k -> Chat SFT -> RL)
#   3. Auto-Resume Watchdog (survives TPU preemptions and transient cloud glitches)
#   4. Rolling Checkpoint Pruning (keeps disk healthy over 16+ days)
#   5. Automatic Export to .rada binary container upon completion
#   6. Live STATUS.txt dashboard and optional Webhook alerts
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

SESSION_NAME="hetman-training"
LOG_DIR="${ROOT_DIR}/logs"
CKPT_DIR="${ROOT_DIR}/checkpoints"
EXPORT_DIR="${ROOT_DIR}/../rada.cpp/models"

mkdir -p "${LOG_DIR}" "${CKPT_DIR}" "${EXPORT_DIR}"

echo -e "\033[1;33m"
cat << "EOF"
================================================================================
   _   _ _____ _____ __  __    _    _   _       ____   ___  ____  
  | | | | ____|_   _|  \/  |  / \  | \ | |     |___ \ / _ \| __ ) 
  | |_| |  _|   | | | |\/| | / _ \ |  \| | _____ __) | | | |  _ \ 
  |  _  | |___  | | | |  | |/ ___ \| |\  ||_____/ __/| |_| | |_) |
  |_| |_|_____| |_| |_|  |_/_/   \_\_| \_|     |_____|\___/|____/ 
        AUTOPILOT TPU LAUNCHER (Cloud TPU v5e-128 Pod Slice)
================================================================================
EOF
echo -e "\033[0m"

# 1. Check Python & TPU Runtime Environment
echo "[1/4] Verifying Python and Cloud TPU Runtime..."
if ! command -v python3 &> /dev/null; then
    echo "[!] Error: python3 is not installed. Please install Python 3.10+ first."
    exit 1
fi

# Check / Install tmux for persistent background detachment
if ! command -v tmux &> /dev/null; then
    echo "[*] Installing tmux for detached session management..."
    sudo apt-get update -y && sudo apt-get install -y tmux
fi

# Verify JAX & TPU Driver installation
python3 -c "import jax; print('[✓] JAX Detected. Hardware Devices:', len(jax.devices()))" 2>/dev/null || {
    echo "[*] JAX with TPU support not found. Auto-installing Cloud TPU dependencies..."
    pip install --upgrade pip
    pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
    pip install flax optax orbax-checkpoint tokenizers torch numpy
    echo "[✓] Cloud TPU dependencies successfully installed."
}

# 2. Check for active training session
if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    echo -e "\033[1;36m[!] A training session '${SESSION_NAME}' is ALREADY RUNNING!\033[0m"
    echo "    - To view live training:  tmux attach -t ${SESSION_NAME}"
    echo "    - To check status:        bash scripts/status.sh"
    echo "    - To stop safely:         bash scripts/stop_training.sh"
    exit 0
fi

# 3. Create Supervisory Watchdog Wrapper
WATCHDOG_SCRIPT="${ROOT_DIR}/scripts/.watchdog_runner.sh"
cat << 'INNER_EOF' > "${WATCHDOG_SCRIPT}"
#!/usr/bin/env bash
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

LOG_FILE="${ROOT_DIR}/logs/training.log"
CRASH_COUNT=0
MAX_CRASHES=10

echo "[Watchdog] Initializing Autonomous Training Loop at $(date)..." >> "${LOG_FILE}"

# Auto-detect whether to use JAX TPU engine or PyTorch Autopilot
IS_TPU=false
if python3 -c "import jax; devices = jax.devices(); exit(0 if any(d.platform == 'tpu' for d in devices) else 1)" 2>/dev/null; then
    IS_TPU=true
fi

while true; do
    echo "----------------------------------------------------------------" >> "${LOG_FILE}"
    echo "[Watchdog] Starting/Resuming Hetman-2.0B at $(date)..." >> "${LOG_FILE}"
    echo "----------------------------------------------------------------" >> "${LOG_FILE}"

    # Launch master orchestrator (JAX TPU distributed scan if on TPU VM, else PyTorch Autopilot)
    if [ "${IS_TPU}" = true ] || [[ "$*" == *"--engine jax"* ]]; then
        echo "[Watchdog] Executing JAX/Flax TPU v5e Distributed Engine..." >> "${LOG_FILE}"
        python3 -u src/train_tpu_jax.py "$@" 2>&1 | tee -a "${LOG_FILE}"
    else
        echo "[Watchdog] Executing PyTorch Autopilot Engine..." >> "${LOG_FILE}"
        python3 -u src/train_autopilot.py "$@" 2>&1 | tee -a "${LOG_FILE}"
    fi
    EXIT_CODE=${PIPESTATUS[0]}

    if [ ${EXIT_CODE} -eq 0 ]; then
        echo -e "\n[🎉] Training completed successfully with exit code 0 at $(date)!" >> "${LOG_FILE}"
        echo "[✓] Autopilot pipeline finished all 4 phases and exported .rada container." >> "${LOG_FILE}"
        break
    else
        CRASH_COUNT=$((CRASH_COUNT + 1))
        echo -e "\n[⚠️] Process exited with error code ${EXIT_CODE} at $(date) (Event #${CRASH_COUNT})." >> "${LOG_FILE}"
        if [ ${CRASH_COUNT} -ge ${MAX_CRASHES} ]; then
            echo "[!] Maximum crash count (${MAX_CRASHES}) reached. Stopping to prevent loop." >> "${LOG_FILE}"
            exit 1
        fi
        echo "[*] Cloud preemption or driver reset detected. Clearing locks..." >> "${LOG_FILE}"
        
        # Reset any dangling TPU lock files
        sudo rm -f /tmp/libtpu* /tmp/tpu* 2>/dev/null || true
        pkill -9 -f "train_autopilot.py" 2>/dev/null || true
        pkill -9 -f "train_tpu_jax.py" 2>/dev/null || true
        
        echo "[*] Waiting 30 seconds for hardware reinitialization..." >> "${LOG_FILE}"
        sleep 30
        echo "[*] Resuming automatically from latest saved checkpoint..." >> "${LOG_FILE}"
    fi
done
INNER_EOF
chmod +x "${WATCHDOG_SCRIPT}"

# 4. Launch within detached tmux session
echo "[2/4] Starting Autopilot inside persistent tmux session '${SESSION_NAME}'..."
tmux new-session -d -s "${SESSION_NAME}" "bash ${WATCHDOG_SCRIPT} $*"

sleep 3

# Verify launch success
if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    echo -e "\033[1;32m[✓] Hetman-2.0B TPU Training successfully launched in background!\033[0m"
    echo "================================================================================"
    echo " 🚀 AUTOPILOT IS NOW RUNNING INDEPENDENTLY ON TPU v5e-128"
    echo "================================================================================"
    echo " You can safely disconnect your SSH terminal now and leave the machine alone."
    echo " The script will autonomously run all 4 phases, prune checkpoints, and export"
    echo " the final 'hetman-2.0b-ternary.rada' model container when finished (~16 days)."
    echo ""
    echo " Useful Commands:"
    echo "   1. View Live Status:      bash scripts/status.sh"
    echo "   2. Follow Live Logs:      tail -f logs/training.log"
    echo "   3. Attach to Console:     tmux attach -t ${SESSION_NAME}  (Detach with: Ctrl+B then D)"
    echo "   4. Stop Safely:           bash scripts/stop_training.sh"
    echo "================================================================================"
else
    echo -e "\033[1;31m[!] Failed to start tmux session. Check logs/training.log for details.\033[0m"
    exit 1
fi
