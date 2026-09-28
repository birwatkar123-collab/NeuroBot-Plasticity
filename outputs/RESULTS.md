# Results Index

This folder contains generated local experiment outputs.

## Plots

- `learning_curves.png` - single-task baseline vs curriculum/replay training
- `continual_learning_forgetting.png` - continual-learning forgetting comparison
- `replay_sweep_tradeoff.png` - replay fraction vs forgetting/new-task error
- `uci_robot_failures_public_dataset.png` - UCI public robot dataset result

## CSV Summaries

- `results_summary.csv` - single-task summary
- `continual_learning_summary.csv` - continual-learning summary
- `replay_sweep.csv` - replay-fraction sweep table
- `uci_robot_failures_summary.csv` - UCI public robot dataset summary

## Key Finding

Replay consolidation reduced Task A forgetting from `0.9377` to `0.3205`, a
relative reduction of `65.81%`. The tradeoff is that Task B final error
increased as replay fraction increased.
