#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail

echo "[1/3] Installing Termux prerequisites"
pkg update -y
pkg install -y python python-pip git unzip clang rust make pkg-config openssl libffi

echo "[2/3] Installing Kaggle CLI"
# Do not self-upgrade pip on Termux; Termux manages pip through its package manager.
python -m pip install --upgrade setuptools wheel packaging
python -m pip install --upgrade kaggle
python --version
kaggle --version

echo "[3/3] Checking Kaggle authentication"
if ! kaggle kernels list -m >/dev/null 2>&1; then
  echo
  echo "Kaggle authentication is required. Run:"
  echo "  kaggle auth login --no-launch-browser"
  echo "Open the printed URL on your phone, approve access, then run this script again."
  exit 2
fi

echo
echo "Kaggle CLI is ready."
