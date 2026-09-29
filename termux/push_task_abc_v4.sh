#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail
REPO_DIR="$HOME/NeuroBot-Plasticity"
KERNEL="birwatkar/neurobot-plasticity-task-abc"
cd "$REPO_DIR"
python -m py_compile kaggle_task_abc/task_abc_kaggle.py
python -m json.tool kaggle_task_abc/kernel-metadata.json >/dev/null
echo "Pushing RLDS/OXE Bridge schema fix..."
kaggle kernels push \
  -p kaggle_task_abc \
  --accelerator NvidiaTeslaT4 \
  --timeout 43200
kaggle kernels status "$KERNEL" || true
echo
echo "Follow logs:"
echo "  kaggle kernels logs $KERNEL --follow --interval 20"
