#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail
cd "$HOME/NeuroBot-Plasticity"
KERNEL="birwatkar/neurobot-plasticity-task-abc"
echo "Pushing corrected semantic Bridge Task A→B→C kernel..."
python -m py_compile kaggle_task_abc/task_abc_kaggle.py
kaggle kernels push -p kaggle_task_abc --accelerator NvidiaTeslaT4 --timeout 43200
echo
kaggle kernels status "$KERNEL" || true
echo
echo "Follow logs:"
echo "  kaggle kernels logs $KERNEL --follow --interval 20"
