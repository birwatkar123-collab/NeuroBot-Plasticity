# NeuroBot-Plasticity

Neuroplasticity-inspired robot task learning with replay consolidation,
curriculum learning, and continual-learning evaluation.

This project studies whether robot training can improve when neural-network
training borrows ideas from neuroplasticity:

- repetition strengthens useful behavior
- gradual practice improves skill acquisition
- replay helps retain previous experience
- error-focused practice improves adaptation

The project includes three local experiments:

1. Single-task action prediction with curriculum learning and replay.
2. Continual learning across two robot-style tasks, where replay consolidation
   reduces forgetting after learning a new task.
3. A replay-fraction sweep that measures the stability-plasticity tradeoff.

## Research Question

Can neuroplasticity-inspired replay help a robot policy remember older skills
while learning a new task?

## Current Milestone

The current version runs three small experiments:

1. Train a baseline model on robot-style state/action trajectories.
2. Train a neuroplasticity-inspired model with:
   - easy-to-hard curriculum stages
   - replay buffer for previous examples
   - extra sampling from high-error examples
3. Train a continual-learning setup:
   - learn Task A
   - then learn Task B
   - compare forgetting with and without replay consolidation
4. Compare validation loss, forgetting, and learning curves.

This scaffold uses synthetic robot-style data first so the whole pipeline runs
without a large dataset download. The next milestone is to replace the synthetic
loader with a subset of BridgeData V2 or Open X-Embodiment.
The repository now includes starter scripts for extracting and training on a
small BridgeData subset through TensorFlow Datasets/RLDS.

## Initial Results

Single-task training:

- baseline mean final validation MSE: `0.0481`
- plasticity-inspired mean final validation MSE: `0.0519`

Continual learning:

- sequential baseline forgetting: `0.9377`
- replay consolidation forgetting: `0.3205`
- sequential baseline Task B final MSE: `0.0443`
- replay consolidation Task B final MSE: `0.2169`
- best forgetting reduction in sweep: `65.82%` with replay fraction `0.45`

Interpretation: replay consolidation strongly reduces forgetting of the old
task, but it slows adaptation to the new task. This is the classic
stability-plasticity tradeoff and gives the project a clear research direction.

Replay sweep:

| Replay fraction | Task A forgetting | Task B final MSE | Forgetting reduction |
| --- | ---: | ---: | ---: |
| 0.00 | 0.9377 | 0.0443 | 0.00% |
| 0.10 | 0.7592 | 0.0581 | 19.04% |
| 0.20 | 0.6070 | 0.0845 | 35.27% |
| 0.30 | 0.4525 | 0.1369 | 51.74% |
| 0.45 | 0.3205 | 0.2169 | 65.82% |

## Recommended Publication Stack

- Training: Kaggle Notebooks with GPU
- Code: GitHub
- Demo: Hugging Face Spaces
- Write-up: Medium, Hashnode, or arXiv later

Public Kaggle notebook:

```text
https://www.kaggle.com/code/birwatkar/neurobot-plasticity
```

## Repository Status

This is version `0.1.0`: a local, reproducible prototype using synthetic
robot-style data. The next research version should add BridgeData V2 results.

## Files

- `REPORT.md` - first project report with method, results, and interpretation
- `REPRODUCE.md` - local reproduction instructions
- `GITHUB_PUBLISHING.md` - checklist for publishing the repository
- `CITATION.cff` - citation metadata
- `LICENSE` - MIT license
- `src/train.py` - baseline and neuroplasticity-inspired training experiment
- `src/continual_learning.py` - replay consolidation experiment
- `src/replay_sweep.py` - replay-fraction tradeoff sweep
- `src/bridgedata_loader.py` - extracts a small TFDS/RLDS BridgeData subset
- `src/train_bridge_subset.py` - trains on the extracted BridgeData subset
- `requirements.txt` - Python dependencies
- `PROJECT_PLAN.md` - roadmap for dataset integration and publishing

## Quick Start

```bash
pip install -r requirements.txt
python src/train.py
python src/continual_learning.py
python src/replay_sweep.py
```

The scripts write result files to `outputs/`.

## Main Finding

The local results show a stability-plasticity tradeoff. More replay reduces
forgetting of the old task, but slows adaptation to the new task. This is a
strong first research story because it directly connects robot learning to a
core neuroplasticity problem: preserving useful old patterns while remaining
adaptable.

## BridgeData V2 Subset Workflow

The full BridgeData dataset is large, so start with a small subset on Kaggle or
another machine with enough disk space.

```bash
python src/bridgedata_loader.py --output data/bridge_subset.npz --split "train[:1%]" --max-episodes 40
python src/train_bridge_subset.py --data data/bridge_subset.npz
```

If Kaggle already has a TensorFlow Datasets cache attached, pass its directory:

```bash
python src/bridgedata_loader.py --data-dir /kaggle/input/YOUR_TFDS_DIR --output data/bridge_subset.npz
```

## Kaggle Upload Package

The `kaggle_kernel/` folder contains a Kaggle-ready notebook package:

- `kernel-metadata.json`
- `neurobot_plasticity_kaggle.ipynb`
- `requirements.txt`
- `src/`

A zipped copy is available at:

```text
outputs/kaggle_upload_package.zip
```

The Kaggle metadata is configured for:

```text
birwatkar/neurobot-plasticity
```

To publish a new version with Kaggle CLI, run:

```bash
kaggle kernels push -p kaggle_kernel
```
