from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from continual_learning import Config, run_strategy


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"


def run_sweep() -> list[dict[str, float]]:
    base_cfg = Config()
    replay_fractions = [0.0, 0.10, 0.20, 0.30, 0.45]
    rows: list[dict[str, float]] = []

    baseline_results = np.array(
        [run_strategy(seed, base_cfg, use_replay=False) for seed in base_cfg.seeds]
    )
    baseline_forgetting = float(baseline_results[:, 0].mean())
    baseline_task_b = float(baseline_results[:, 2].mean())

    rows.append(
        {
            "replay_fraction": 0.0,
            "forgetting_mean": baseline_forgetting,
            "task_b_mse_mean": baseline_task_b,
            "forgetting_reduction_percent": 0.0,
        }
    )

    for fraction in replay_fractions[1:]:
        cfg = replace(base_cfg, replay_fraction=fraction)
        results = np.array([run_strategy(seed, cfg, use_replay=True) for seed in cfg.seeds])
        forgetting = float(results[:, 0].mean())
        task_b = float(results[:, 2].mean())
        rows.append(
            {
                "replay_fraction": fraction,
                "forgetting_mean": forgetting,
                "task_b_mse_mean": task_b,
                "forgetting_reduction_percent": (
                    (baseline_forgetting - forgetting) / baseline_forgetting * 100.0
                ),
            }
        )

    return rows


def write_csv(rows: list[dict[str, float]]) -> Path:
    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / "replay_sweep.csv"
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def plot_sweep(rows: list[dict[str, float]]) -> Path:
    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / "replay_sweep_tradeoff.png"

    fractions = [row["replay_fraction"] for row in rows]
    forgetting = [row["forgetting_mean"] for row in rows]
    task_b = [row["task_b_mse_mean"] for row in rows]

    fig, ax1 = plt.subplots(figsize=(8.5, 5))
    ax1.plot(fractions, forgetting, marker="o", linewidth=2, label="Task A forgetting")
    ax1.set_xlabel("Replay fraction during Task B training")
    ax1.set_ylabel("Task A forgetting, lower is better")
    ax1.grid(alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(fractions, task_b, marker="s", linewidth=2, color="#b45309", label="Task B MSE")
    ax2.set_ylabel("Task B final MSE, lower is better")

    lines = ax1.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="upper center")
    plt.title("Replay Consolidation Tradeoff")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    return output_path


def main() -> None:
    rows = run_sweep()
    csv_path = write_csv(rows)
    graph_path = plot_sweep(rows)

    best_balanced = min(rows, key=lambda row: row["forgetting_mean"] + row["task_b_mse_mean"])
    print(f"Saved sweep CSV: {csv_path}")
    print(f"Saved sweep graph: {graph_path}")
    print(
        "Best balanced replay fraction: "
        f"{best_balanced['replay_fraction']:.2f} "
        f"(forgetting={best_balanced['forgetting_mean']:.4f}, "
        f"task_b_mse={best_balanced['task_b_mse_mean']:.4f})"
    )


if __name__ == "__main__":
    main()
