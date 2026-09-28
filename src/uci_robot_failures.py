from __future__ import annotations

import io
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import requests
import urllib3
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "uci_robot_failures"
OUTPUT_DIR = ROOT / "outputs"
UCI_URL = "https://archive.ics.uci.edu/static/public/138/robot+execution+failures.zip"


@dataclass
class Config:
    seed: int = 7
    batch_size: int = 16
    epochs_per_task: int = 80
    lr: float = 1e-3
    replay_fraction: float = 0.35
    hidden_dim: int = 64


class FailureNet(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def download_dataset() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    marker = DATA_DIR / ".downloaded"
    if marker.exists():
        return DATA_DIR

    try:
        response = requests.get(UCI_URL, timeout=60)
    except requests.exceptions.SSLError:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        response = requests.get(UCI_URL, timeout=60, verify=False)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        archive.extractall(DATA_DIR)
    marker.write_text("downloaded\n", encoding="utf-8")
    return DATA_DIR


def parse_lp_file(path: Path) -> tuple[np.ndarray, list[str]]:
    rows: list[np.ndarray] = []
    labels: list[str] = []
    current_label: str | None = None
    current_values: list[list[float]] = []

    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if re.match(r"^[+-]?\d", line):
            values = [float(item) for item in re.split(r"\s+", line) if item]
            current_values.append(values)
            continue

        if current_label is not None and current_values:
            rows.append(np.array(current_values, dtype=np.float32).reshape(-1))
            labels.append(current_label)
        current_label = line
        current_values = []

    if current_label is not None and current_values:
        rows.append(np.array(current_values, dtype=np.float32).reshape(-1))
        labels.append(current_label)

    if not rows:
        raise RuntimeError(f"No robot failure samples parsed from {path}")
    return np.stack(rows), labels


def load_tasks() -> tuple[tuple[np.ndarray, list[str]], tuple[np.ndarray, list[str]]]:
    data_dir = download_dataset()
    files = {path.name.lower(): path for path in data_dir.rglob("*.data")}
    lp1 = files.get("lp1.data")
    lp2 = files.get("lp2.data")
    if lp1 is None or lp2 is None:
        raise RuntimeError(f"Expected lp1.data and lp2.data inside {data_dir}")
    return parse_lp_file(lp1), parse_lp_file(lp2)


def encode_labels(labels_a: list[str], labels_b: list[str]) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    names = sorted(set(labels_a) | set(labels_b))
    mapping = {name: idx for idx, name in enumerate(names)}
    y_a = np.array([mapping[label] for label in labels_a], dtype=np.int64)
    y_b = np.array([mapping[label] for label in labels_b], dtype=np.int64)
    return y_a, y_b, mapping


def standardize(train_a: np.ndarray, train_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    joined = np.concatenate([train_a, train_b], axis=0)
    mean = joined.mean(axis=0, keepdims=True)
    std = joined.std(axis=0, keepdims=True) + 1e-6
    return (train_a - mean) / std, (train_b - mean) / std


def split_dataset(x: np.ndarray, y: np.ndarray, seed: int) -> tuple[TensorDataset, TensorDataset]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(x))
    split = max(1, int(len(indices) * 0.7))
    train_idx = indices[:split]
    test_idx = indices[split:]
    train = TensorDataset(torch.from_numpy(x[train_idx]).float(), torch.from_numpy(y[train_idx]))
    test = TensorDataset(torch.from_numpy(x[test_idx]).float(), torch.from_numpy(y[test_idx]))
    return train, test


def accuracy(model: FailureNet, data: TensorDataset, cfg: Config) -> float:
    model.eval()
    loader = DataLoader(data, batch_size=cfg.batch_size)
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            pred = model(x).argmax(dim=1)
            correct += int((pred == y).sum())
            total += len(y)
    return correct / max(1, total)


def train_task(
    model: FailureNet,
    optimizer: torch.optim.Optimizer,
    current: TensorDataset,
    cfg: Config,
    replay: TensorDataset | None = None,
) -> None:
    loss_fn = nn.CrossEntropyLoss()
    replay_x: torch.Tensor | None = None
    replay_y: torch.Tensor | None = None
    if replay is not None:
        replay_x, replay_y = replay.tensors

    for _ in range(cfg.epochs_per_task):
        model.train()
        loader = DataLoader(current, batch_size=cfg.batch_size, shuffle=True)
        for x, y in loader:
            if replay_x is not None and replay_y is not None:
                replay_count = max(1, int(len(x) * cfg.replay_fraction))
                replay_idx = torch.randint(0, len(replay_x), (replay_count,))
                x = torch.cat([x, replay_x[replay_idx]], dim=0)
                y = torch.cat([y, replay_y[replay_idx]], dim=0)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()


def run_public_dataset_experiment() -> dict[str, float]:
    cfg = Config()
    set_seed(cfg.seed)

    (x_a, labels_a), (x_b, labels_b) = load_tasks()
    y_a, y_b, label_map = encode_labels(labels_a, labels_b)
    x_a, x_b = standardize(x_a, x_b)
    train_a, test_a = split_dataset(x_a, y_a, cfg.seed)
    train_b, test_b = split_dataset(x_b, y_b, cfg.seed + 1)

    model_base = FailureNet(x_a.shape[1], len(label_map), cfg.hidden_dim)
    opt_base = torch.optim.AdamW(model_base.parameters(), lr=cfg.lr)
    train_task(model_base, opt_base, train_a, cfg)
    acc_a_before_base = accuracy(model_base, test_a, cfg)
    train_task(model_base, opt_base, train_b, cfg)
    acc_a_after_base = accuracy(model_base, test_a, cfg)
    acc_b_base = accuracy(model_base, test_b, cfg)

    set_seed(cfg.seed)
    model_replay = FailureNet(x_a.shape[1], len(label_map), cfg.hidden_dim)
    opt_replay = torch.optim.AdamW(model_replay.parameters(), lr=cfg.lr)
    train_task(model_replay, opt_replay, train_a, cfg)
    acc_a_before_replay = accuracy(model_replay, test_a, cfg)
    train_task(model_replay, opt_replay, train_b, cfg, replay=train_a)
    acc_a_after_replay = accuracy(model_replay, test_a, cfg)
    acc_b_replay = accuracy(model_replay, test_b, cfg)

    results = {
        "baseline_task_a_before": acc_a_before_base,
        "baseline_task_a_after": acc_a_after_base,
        "baseline_task_b": acc_b_base,
        "replay_task_a_before": acc_a_before_replay,
        "replay_task_a_after": acc_a_after_replay,
        "replay_task_b": acc_b_replay,
        "baseline_forgetting": acc_a_before_base - acc_a_after_base,
        "replay_forgetting": acc_a_before_replay - acc_a_after_replay,
    }
    save_outputs(results)
    return results


def save_outputs(results: dict[str, float]) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    csv_path = OUTPUT_DIR / "uci_robot_failures_summary.csv"
    csv_path.write_text(
        "\n".join(["metric,value"] + [f"{key},{value:.6f}" for key, value in results.items()])
        + "\n",
        encoding="utf-8",
    )

    labels = ["Task A after Task B", "Task B final"]
    baseline = [results["baseline_task_a_after"], results["baseline_task_b"]]
    replay = [results["replay_task_a_after"], results["replay_task_b"]]
    x = np.arange(len(labels))
    width = 0.36
    plt.figure(figsize=(8, 5))
    plt.bar(x - width / 2, baseline, width, label="Sequential baseline")
    plt.bar(x + width / 2, replay, width, label="Replay consolidation")
    plt.xticks(x, labels)
    plt.ylabel("Accuracy, higher is better")
    plt.title("Public Dataset: UCI Robot Execution Failures")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "uci_robot_failures_public_dataset.png", dpi=180)


def main() -> None:
    results = run_public_dataset_experiment()
    print("Public dataset: UCI Robot Execution Failures")
    for key, value in results.items():
        print(f"{key}: {value:.4f}")
    print(f"Saved summary: {OUTPUT_DIR / 'uci_robot_failures_summary.csv'}")
    print(f"Saved graph: {OUTPUT_DIR / 'uci_robot_failures_public_dataset.png'}")


if __name__ == "__main__":
    main()
