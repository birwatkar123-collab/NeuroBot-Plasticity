#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail
ROOT="${1:-$HOME/NeuroBot-Plasticity}"
KERNEL="birwatkar/neurobot-plasticity-task-abc"
cd "$ROOT"
python -m json.tool kaggle_task_abc/kernel-metadata.json >/dev/null
python -m py_compile kaggle_task_abc/task_abc_kaggle.py

echo "Pushing corrected self-contained Version 2 to Kaggle..."
kaggle kernels push -p kaggle_task_abc --accelerator NvidiaTeslaT4 --timeout 43200

echo
echo "Status:"
kaggle kernels status "$KERNEL" || true

echo
echo "Follow logs with:"
echo "  kaggle kernels logs $KERNEL --follow --interval 20"
