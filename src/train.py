from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"


@dataclass
class Config:
    seed: int = 7
    seeds: tuple[int, ...] = (7, 19, 31)
    input_dim: int = 12
    action_dim: int = 4
    train_size: int = 8000
    val_size: int = 2000
    batch_size: int = 128
    epochs: int = 24
    lr: float = 3e-4
    replay_fraction: float = 0.45
    hard_example_fraction: float = 0.30
    consolidation_epochs: int = 6


@dataclass
class SyntheticRobotTask:
    weights: torch.Tensor


class PolicyNet(nn.Module):
    def __init__(self, input_dim: int, action_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + 1, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x: torch.Tensor, difficulty: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, difficulty], dim=1))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_robot_task(cfg: Config) -> SyntheticRobotTask:
    weights = torch.randn(cfg.input_dim, cfg.action_dim) / math.sqrt(cfg.input_dim)
    return SyntheticRobotTask(weights=weights)


def make_robot_style_data(
    size: int, cfg: Config, task: SyntheticRobotTask
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create a small proxy for robot state-to-action learning.

    Difficulty controls noise and nonlinearity, imitating easier and harder
    manipulation settings such as larger objects, clutter, or slippery contact.
    """
    x = torch.randn(size, cfg.input_dim)
    difficulty = torch.rand(size, 1)

    nonlinear = torch.sin(x[:, : cfg.action_dim] * (1.0 + 2.0 * difficulty))
    base_action = x @ task.weights + 0.35 * nonlinear
    noise = torch.randn(size, cfg.action_dim) * (0.03 + 0.18 * difficulty)
    action = base_action + noise
    return x.float(), difficulty.float(), action.float()


def validation_loss(model: PolicyNet, val: TensorDataset, cfg: Config) -> float:
    model.eval()
    loader = DataLoader(val, batch_size=cfg.batch_size)
    losses: list[float] = []
    loss_fn = nn.MSELoss()
    with torch.no_grad():
        for x, difficulty, action in loader:
            losses.append(loss_fn(model(x, difficulty), action).item())
    return float(np.mean(losses))


def train_baseline(train: TensorDataset, val: TensorDataset, cfg: Config) -> list[float]:
    model = PolicyNet(cfg.input_dim, cfg.action_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()
    history: list[float] = []

    for _ in range(cfg.epochs):
        model.train()
        loader = DataLoader(train, batch_size=cfg.batch_size, shuffle=True)
        for x, difficulty, action in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(x, difficulty), action)
            loss.backward()
            optimizer.step()
        history.append(validation_loss(model, val, cfg))

    return history


def curriculum_indices(difficulty: torch.Tensor, epoch: int, cfg: Config) -> torch.Tensor:
    curriculum_epochs = cfg.epochs - cfg.consolidation_epochs
    if epoch >= curriculum_epochs:
        return torch.arange(len(difficulty))

    max_difficulty = min(1.0, 0.30 + 0.70 * (epoch + 1) / curriculum_epochs)
    return torch.where(difficulty.squeeze() <= max_difficulty)[0]


def train_neuroplasticity_inspired(train: TensorDataset, val: TensorDataset, cfg: Config) -> list[float]:
    model = PolicyNet(cfg.input_dim, cfg.action_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss(reduction="none")
    history: list[float] = []
    replay_indices: list[int] = []

    x_all, difficulty_all, action_all = train.tensors

    for epoch in range(cfg.epochs):
        model.train()
        current_indices = curriculum_indices(difficulty_all, epoch, cfg).tolist()

        replay_count = int(len(current_indices) * cfg.replay_fraction)
        replay_sample = random.sample(replay_indices, min(replay_count, len(replay_indices)))
        epoch_indices = current_indices + replay_sample
        random.shuffle(epoch_indices)

        for start in range(0, len(epoch_indices), cfg.batch_size):
            batch_indices = torch.tensor(epoch_indices[start : start + cfg.batch_size])
            x = x_all[batch_indices]
            difficulty = difficulty_all[batch_indices]
            action = action_all[batch_indices]

            optimizer.zero_grad()
            per_item = loss_fn(model(x, difficulty), action).mean(dim=1)
            hard_weight = 1.0 + difficulty.squeeze()
            loss = (per_item * hard_weight).mean()
            loss.backward()
            optimizer.step()

            hard_count = max(1, int(len(batch_indices) * cfg.hard_example_fraction))
            hard_local = torch.topk(per_item.detach(), k=min(hard_count, len(per_item))).indices
            replay_indices.extend(batch_indices[hard_local].tolist())
            replay_indices = replay_indices[-4000:]

        history.append(validation_loss(model, val, cfg))

    return history


def plot_history(baseline_runs: list[list[float]], plasticity_runs: list[list[float]]) -> Path:
    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / "learning_curves.png"
    baseline = np.array(baseline_runs)
    plasticity = np.array(plasticity_runs)
    epochs = np.arange(1, baseline.shape[1] + 1)

    plt.figure(figsize=(9, 5))
    plt.plot(epochs, baseline.mean(axis=0), label="Baseline training", linewidth=2)
    plt.fill_between(
        epochs,
        baseline.mean(axis=0) - baseline.std(axis=0),
        baseline.mean(axis=0) + baseline.std(axis=0),
        alpha=0.16,
    )
    plt.plot(epochs, plasticity.mean(axis=0), label="Curriculum + replay", linewidth=2)
    plt.fill_between(
        epochs,
        plasticity.mean(axis=0) - plasticity.std(axis=0),
        plasticity.mean(axis=0) + plasticity.std(axis=0),
        alpha=0.16,
    )
    plt.xlabel("Epoch")
    plt.ylabel("Validation MSE")
    plt.title("Robot Action Prediction: Multi-Seed Training Comparison")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    return output_path


def write_summary(baseline_runs: list[list[float]], plasticity_runs: list[list[float]]) -> Path:
    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / "results_summary.csv"
    baseline_final = np.array([run[-1] for run in baseline_runs])
    plasticity_final = np.array([run[-1] for run in plasticity_runs])
    improvement = (baseline_final - plasticity_final) / baseline_final * 100.0

    lines = [
        "metric,baseline,plasticity_inspired,relative_improvement_percent",
        (
            "final_validation_mse_mean,"
            f"{baseline_final.mean():.6f},"
            f"{plasticity_final.mean():.6f},"
            f"{improvement.mean():.2f}"
        ),
        (
            "final_validation_mse_std,"
            f"{baseline_final.std():.6f},"
            f"{plasticity_final.std():.6f},"
            f"{improvement.std():.2f}"
        ),
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def run_once(seed: int, cfg: Config) -> tuple[list[float], list[float]]:
    run_cfg = Config(seed=seed, seeds=cfg.seeds)
    set_seed(run_cfg.seed)

    task = make_robot_task(run_cfg)
    x_train, d_train, y_train = make_robot_style_data(run_cfg.train_size, run_cfg, task)
    x_val, d_val, y_val = make_robot_style_data(run_cfg.val_size, run_cfg, task)
    train = TensorDataset(x_train, d_train, y_train)
    val = TensorDataset(x_val, d_val, y_val)

    baseline = train_baseline(train, val, run_cfg)
    plasticity = train_neuroplasticity_inspired(train, val, run_cfg)
    return baseline, plasticity


def main() -> None:
    cfg = Config()
    baseline_runs: list[list[float]] = []
    plasticity_runs: list[list[float]] = []

    for seed in cfg.seeds:
        print(f"Running seed {seed}...")
        baseline, plasticity = run_once(seed, cfg)
        baseline_runs.append(baseline)
        plasticity_runs.append(plasticity)
        print(
            f"  baseline={baseline[-1]:.4f}, "
            f"plasticity-inspired={plasticity[-1]:.4f}"
        )

    graph_path = plot_history(baseline_runs, plasticity_runs)
    summary_path = write_summary(baseline_runs, plasticity_runs)

    baseline_final = np.array([run[-1] for run in baseline_runs])
    plasticity_final = np.array([run[-1] for run in plasticity_runs])

    print(f"Baseline mean final validation MSE: {baseline_final.mean():.4f}")
    print(f"Plasticity-inspired mean final validation MSE: {plasticity_final.mean():.4f}")
    print(f"Saved graph: {graph_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
