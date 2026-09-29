#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail

ROOT="${1:-$HOME/NeuroBot-Plasticity}"
KERNEL="birwatkar/neurobot-plasticity-task-abc"

if [ ! -f "$ROOT/kaggle_task_abc/kernel-metadata.json" ]; then
  echo "Missing $ROOT/kaggle_task_abc/kernel-metadata.json"
  echo "Extract the TaskABC package into $ROOT first."
  exit 2
fi

cd "$ROOT"
python -m json.tool kaggle_task_abc/kernel-metadata.json >/dev/null

echo "Pushing $KERNEL to Kaggle..."
kaggle kernels push \
  -p kaggle_task_abc \
  --accelerator NvidiaTeslaT4 \
  --timeout 43200

echo
echo "Status:"
kaggle kernels status "$KERNEL" || true

echo
echo "Follow logs with:"
echo "  kaggle kernels logs $KERNEL --follow --interval 20"
echo
echo "Download outputs with:"
echo "  mkdir -p $ROOT/kaggle_outputs/task_abc"
echo "  kaggle kernels output $KERNEL -p $ROOT/kaggle_outputs/task_abc"
