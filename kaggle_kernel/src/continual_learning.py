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
    seeds: tuple[int, ...] = (7, 19, 31)
    input_dim: int = 12
    action_dim: int = 4
    task_size: int = 5000
    val_size: int = 1200
    batch_size: int = 128
    epochs_per_task: int = 14
    lr: float = 3e-4
    replay_size: int = 1200
    replay_fraction: float = 0.45


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


def make_task(size: int, cfg: Config, task_shift: float) -> TensorDataset:
    x = torch.randn(size, cfg.input_dim)
    difficulty = torch.rand(size, 1)
    weights = torch.linspace(-1.0, 1.0, cfg.input_dim * cfg.action_dim).reshape(
        cfg.input_dim, cfg.action_dim
    )
    weights = torch.roll(weights, shifts=int(task_shift), dims=0) / math.sqrt(cfg.input_dim)
    nonlinear = torch.sin(x[:, : cfg.action_dim] * (1.0 + difficulty + task_shift * 0.08))
    action = x @ weights + 0.45 * nonlinear
    action += torch.randn(size, cfg.action_dim) * (0.04 + 0.10 * difficulty)
    return TensorDataset(x.float(), difficulty.float(), action.float())


def eval_loss(model: PolicyNet, data: TensorDataset, cfg: Config) -> float:
    model.eval()
    loader = DataLoader(data, batch_size=cfg.batch_size)
    loss_fn = nn.MSELoss()
    losses: list[float] = []
    with torch.no_grad():
        for x, difficulty, action in loader:
            losses.append(loss_fn(model(x, difficulty), action).item())
    return float(np.mean(losses))


def train_epoch(
    model: PolicyNet,
    optimizer: torch.optim.Optimizer,
    current: TensorDataset,
    cfg: Config,
    replay: TensorDataset | None = None,
) -> None:
    model.train()
    loss_fn = nn.MSELoss()
    x_cur, d_cur, y_cur = current.tensors
    indices = list(range(len(x_cur)))
    random.shuffle(indices)

    replay_tensors = replay.tensors if replay is not None else None
    replay_count = int(cfg.batch_size * cfg.replay_fraction)
    current_count = cfg.batch_size - replay_count if replay_tensors else cfg.batch_size

    for start in range(0, len(indices), current_count):
        batch_indices = torch.tensor(indices[start : start + current_count])
        x = x_cur[batch_indices]
        d = d_cur[batch_indices]
        y = y_cur[batch_indices]

        if replay_tensors is not None:
            xr, dr, yr = replay_tensors
            replay_indices = torch.randint(0, len(xr), (replay_count,))
            x = torch.cat([x, xr[replay_indices]], dim=0)
            d = torch.cat([d, dr[replay_indices]], dim=0)
            y = torch.cat([y, yr[replay_indices]], dim=0)

        optimizer.zero_grad()
        loss = loss_fn(model(x, d), y)
        loss.backward()
        optimizer.step()


def make_replay_memory(data: TensorDataset, cfg: Config) -> TensorDataset:
    tensors = data.tensors
    indices = torch.randperm(len(tensors[0]))[: cfg.replay_size]
    return TensorDataset(*(tensor[indices] for tensor in tensors))


def run_strategy(seed: int, cfg: Config, use_replay: bool) -> tuple[float, float, float]:
    set_seed(seed)
    model = PolicyNet(cfg.input_dim, cfg.action_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    task_a = make_task(cfg.task_size, cfg, task_shift=0.0)
    task_b = make_task(cfg.task_size, cfg, task_shift=5.0)
    val_a = make_task(cfg.val_size, cfg, task_shift=0.0)
    val_b = make_task(cfg.val_size, cfg, task_shift=5.0)

    for _ in range(cfg.epochs_per_task):
        train_epoch(model, optimizer, task_a, cfg)

    loss_a_before_b = eval_loss(model, val_a, cfg)
    replay = make_replay_memory(task_a, cfg) if use_replay else None

    for _ in range(cfg.epochs_per_task):
        train_epoch(model, optimizer, task_b, cfg, replay=replay)

    loss_a_after_b = eval_loss(model, val_a, cfg)
    loss_b_after_b = eval_loss(model, val_b, cfg)
    forgetting = loss_a_after_b - loss_a_before_b
    return forgetting, loss_a_after_b, loss_b_after_b


def plot_results(baseline: np.ndarray, plasticity: np.ndarray) -> Path:
    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / "continual_learning_forgetting.png"
    labels = ["Forgetting on Task A", "Final loss on Task B"]
    baseline_values = [baseline[:, 0].mean(), baseline[:, 2].mean()]
    plasticity_values = [plasticity[:, 0].mean(), plasticity[:, 2].mean()]

    x = np.arange(len(labels))
    width = 0.36
    plt.figure(figsize=(8, 5))
    plt.bar(x - width / 2, baseline_values, width, label="Sequential baseline")
    plt.bar(x + width / 2, plasticity_values, width, label="Replay consolidation")
    plt.xticks(x, labels)
    plt.ylabel("MSE, lower is better")
    plt.title("Continual Robot Learning: Replay Reduces Forgetting")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    return output_path


def write_summary(baseline: np.ndarray, plasticity: np.ndarray) -> Path:
    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / "continual_learning_summary.csv"
    forgetting_reduction = (baseline[:, 0] - plasticity[:, 0]) / baseline[:, 0] * 100.0
    lines = [
        "metric,sequential_baseline,replay_consolidation,relative_change_percent",
        (
            "task_a_forgetting_mean,"
            f"{baseline[:, 0].mean():.6f},"
            f"{plasticity[:, 0].mean():.6f},"
            f"{forgetting_reduction.mean():.2f}"
        ),
        (
            "task_b_final_mse_mean,"
            f"{baseline[:, 2].mean():.6f},"
            f"{plasticity[:, 2].mean():.6f},"
            f"{((baseline[:, 2].mean() - plasticity[:, 2].mean()) / baseline[:, 2].mean() * 100.0):.2f}"
        ),
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def main() -> None:
    cfg = Config()
    baseline_rows = []
    plasticity_rows = []

    for seed in cfg.seeds:
        print(f"Running continual-learning seed {seed}...")
        baseline_rows.append(run_strategy(seed, cfg, use_replay=False))
        plasticity_rows.append(run_strategy(seed, cfg, use_replay=True))

    baseline = np.array(baseline_rows)
    plasticity = np.array(plasticity_rows)
    graph_path = plot_results(baseline, plasticity)
    summary_path = write_summary(baseline, plasticity)

    print(f"Baseline forgetting: {baseline[:, 0].mean():.4f}")
    print(f"Replay consolidation forgetting: {plasticity[:, 0].mean():.4f}")
    print(f"Baseline Task B final MSE: {baseline[:, 2].mean():.4f}")
    print(f"Replay Task B final MSE: {plasticity[:, 2].mean():.4f}")
    print(f"Saved graph: {graph_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
