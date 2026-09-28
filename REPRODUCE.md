# Reproducing The Local Experiments

## Setup

```bash
cd D:\NeuroBot-Plasticity
pip install -r requirements.txt
```

## Run Synthetic Experiments

```bash
python src/train.py
python src/continual_learning.py
python src/replay_sweep.py
python src/uci_robot_failures.py
```

## Expected Outputs

The scripts write files to `outputs/`:

- `learning_curves.png`
- `continual_learning_forgetting.png`
- `replay_sweep_tradeoff.png`
- `results_summary.csv`
- `continual_learning_summary.csv`
- `replay_sweep.csv`
- `uci_robot_failures_summary.csv`
- `uci_robot_failures_public_dataset.png`

## Current Key Result

Replay consolidation reduced Task A forgetting from `0.9377` to `0.3205`, a
relative reduction of `65.81%`.

## BridgeData V2 Preparation

The full BridgeData dataset is large. For the next stage, use a small TFDS/RLDS
subset:

```bash
python src/bridgedata_loader.py --output data/bridge_subset.npz --split "train[:1%]" --max-episodes 40
python src/train_bridge_subset.py --data data/bridge_subset.npz
```
