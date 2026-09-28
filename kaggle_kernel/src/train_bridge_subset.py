from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset

from train import Config, plot_history, train_baseline, train_neuroplasticity_inspired


def load_npz_dataset(path: Path, cfg: Config) -> tuple[TensorDataset, TensorDataset, Config]:
    data = np.load(path)
    states = data["states"].astype(np.float32)
    difficulties = data["difficulties"].astype(np.float32)
    actions = data["actions"].astype(np.float32)

    usable = min(len(states), len(actions), len(difficulties))
    states = states[:usable]
    actions = actions[:usable]
    difficulties = difficulties[:usable]

    indices = np.random.default_rng(cfg.seed).permutation(usable)
    split = int(usable * 0.8)
    train_idx = indices[:split]
    val_idx = indices[split:]

    bridge_cfg = replace(cfg, input_dim=states.shape[1], action_dim=actions.shape[1])
    train = TensorDataset(
        torch.from_numpy(states[train_idx]),
        torch.from_numpy(difficulties[train_idx]),
        torch.from_numpy(actions[train_idx]),
    )
    val = TensorDataset(
        torch.from_numpy(states[val_idx]),
        torch.from_numpy(difficulties[val_idx]),
        torch.from_numpy(actions[val_idx]),
    )
    return train, val, bridge_cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Train on extracted BridgeData subset.")
    parser.add_argument("--data", type=Path, default=Path("data/bridge_subset.npz"))
    args = parser.parse_args()

    cfg = Config(epochs=12, seeds=(7,))
    train, val, bridge_cfg = load_npz_dataset(args.data, cfg)
    baseline = train_baseline(train, val, bridge_cfg)
    plasticity = train_neuroplasticity_inspired(train, val, bridge_cfg)
    graph_path = plot_history([baseline], [plasticity])

    print(f"Bridge subset baseline final validation MSE: {baseline[-1]:.4f}")
    print(f"Bridge subset plasticity-inspired final validation MSE: {plasticity[-1]:.4f}")
    print(f"Saved graph: {graph_path}")


if __name__ == "__main__":
    main()
