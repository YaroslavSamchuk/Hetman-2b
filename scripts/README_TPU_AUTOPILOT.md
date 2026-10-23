# Autonomous Cloud Training Autopilot: Operational Field Guide

> **"Set it, launch it, go to sleep."**  
> This directory contains the autonomous lifecycle scripts that orchestrate the **Hetman-2.0B** pre-training and long-context annealing run across Cloud TPU v5e-128 pods.

---

## 🧭 Why We Built the Autopilot

Training a 2.14B-parameter foundation model across 855 billion tokens takes ~16.5 days of non-stop compute. On cloud clusters (especially when taking advantage of spot or preemptible nodes to keep research costs sustainable), things go wrong:
- Cloud nodes get preempted mid-epoch.
- Transient network blips disrupt host-to-host communications.
- Checkpoint writes can be interrupted mid-flight, leaving corrupted files.
- Workers leak memory over dozens of consecutive hours.

We wrote this Autopilot suite so no human has to sit and babysit terminal windows for 16 days. It handles crash recovery, atomic checkpoint rolling, multi-host JAX barrier re-initialization, and automatic conversion to `.rada` inference format when the final loss step completes.

---

## 🚀 Quickstart

Once compute is provisioned and your Cloud TPU VM is created:

```bash
# 1. Connect to the TPU VM via SSH
ssh <user>@<tpu-vm-ip> # or cloud CLI

# 2. Navigate to project root
cd Hetman-2b

# 3. Test locally first (dry run in 10 seconds)
python src/train_autopilot.py --dry_run --max_steps 20

# 4. Launch the autonomous autopilot in the background
bash scripts/launch_autopilot.sh
```

**That's it!** The launcher will:
1. Automatically inspect TPU hardware (128 v5e accelerator cores).
2. Install/verify `jax[tpu]`, `flax`, `optax`, `orbax-checkpoint` if needed.
3. Spawn a resilient `tmux` session (`hetman-training`) managed by an autonomous watchdog.
4. Execute the complete 4-phase curriculum.

> **💡 Safe Terminal Detachment:** Training runs detached inside `tmux`. You can safely close your SSH connection or power off your local workstation.

---

## 2. Monitoring Progress

To monitor progress at any time, connect to the TPU VM and run:

```bash
# 1. Terminal dashboard (current phase, step, tokens, loss, ETA)
bash scripts/status.sh

# 2. Or view live status log
cat logs/STATUS.txt

# 3. Attach to live training terminal
tmux attach -t hetman-training
# (To detach without stopping: press Ctrl+B, then D)

# 4. Stream recent logs
tail -f logs/training.log
```

---

## 3. Remote Notifications (Telegram / Discord / Slack)

To receive real-time notifications on launch, phase transitions, and final completion:
```bash
# Set your Webhook URL before launching:
export WEBHOOK_URL="https://discord.com/api/webhooks/..."
# Or for Telegram bot:
# export WEBHOOK_URL="https://api.telegram.org/bot<TOKEN>/sendMessage?chat_id=<CHAT_ID>&text="

bash scripts/launch_autopilot.sh
```

---

## 4. Multi-Phase Curriculum Schedule (~16.5 Days)

| Phase | Duration | Token Volume | Description |
| :--- | :--- | :--- | :--- |
| **Phase 1: 4k Pre-training** | ~14.2 days | 820 Billion tokens | Core language, factual knowledge, logic, and code (WSD schedule, peak LR `0.0018`). |
| **Phase 2: 256k Long-Context** | ~1.9 days | 30 Billion tokens | Context window expansion to 262,144 tokens via YaRN RoPE (scale factor $s=64$). |
| **Phase 3: Chat & Reasoning SFT** | ~0.35 days | 5 Billion tokens | ChatML alignment, multi-turn reasoning chains `<think>...</think>`, thought pruning. |
| **Phase 4: RL Alignment** | ~4 hours | 500 steps | Post-training alignment via Sum-Preserving GAR and in-process AST verifier. |
| **Phase 5: Auto-Export** | ~2 minutes | Automatic | Weight conversion into packed 2-bit ternary binary container `hetman-2.0b-ternary.rada`. |

---

## 5. Fault Tolerance (Preemptions & Reboot Watchdog)

Cloud accelerator instances can experience transient preemption or host maintenance.
- **Integrated Watchdog:** Continuously tracks process health.
- If a VM reboots or driver glitches occur:
  1. Cleans locked `/tmp/libtpu*` sockets.
  2. Waits 30 seconds for network stabilization.
  3. **Automatically resumes training from the latest valid checkpoint**.
- **Checkpoint Rotation:** Automatically saves checkpoints and preserves top-3 recent checkpoints plus the best loss checkpoint to protect disk space.

---

## 6. Output Deliverables

Upon completion, `logs/TRAINING_COMPLETED.md` is generated:
```text
================================================================================
 [✓] ALL TRAINING PHASES COMPLETED SUCCESSFULLY! 
================================================================================
[*] Phase 5: Automatically Exporting Trained Model to .rada Binary Container...
[✓] Successfully exported 897 tensors to ../rada.cpp/models/hetman-2.0b-ternary.rada
    Embedded Tokenizer: Present (65,536 vocab)
[🎉] Complete! Ready-to-use model produced at: ../rada.cpp/models/hetman-2.0b-ternary.rada
```

### Running the Trained Model:
```bash
# 1. High-throughput CLI inference (>350 tok/s):
./rada.cpp/build/Release/rada_cli -m rada.cpp/models/hetman-2.0b-ternary.rada

# 2. Local browser web server UI:
./rada.cpp/build/Release/rada_web -m rada.cpp/models/hetman-2.0b-ternary.rada
```
