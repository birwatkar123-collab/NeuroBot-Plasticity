from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def _get_nested(example: dict[str, Any], path: tuple[str, ...]) -> Any | None:
    current: Any = example
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _flatten_state(example: dict[str, Any]) -> np.ndarray | None:
    candidates = [
        ("steps", "observation", "state"),
        ("steps", "observation", "proprio"),
        ("steps", "observation", "robot_state"),
    ]
    for path in candidates:
        value = _get_nested(example, path)
        if value is not None:
            arr = _to_numpy(value)
            return arr.reshape(arr.shape[0], -1).astype(np.float32)
    return None


def _flatten_action(example: dict[str, Any]) -> np.ndarray | None:
    value = _get_nested(example, ("steps", "action"))
    if value is None:
        return None
    arr = _to_numpy(value)
    return arr.reshape(arr.shape[0], -1).astype(np.float32)


def extract_bridge_subset(
    output_path: Path,
    data_dir: str | None,
    split: str,
    max_episodes: int,
    max_steps_per_episode: int,
) -> Path:
    import tensorflow_datasets as tfds

    builder_kwargs = {"data_dir": data_dir} if data_dir else {}
    dataset = tfds.load("bridge", split=split, shuffle_files=True, **builder_kwargs)

    states: list[np.ndarray] = []
    difficulties: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    episodes_used = 0

    for episode in dataset:
        state = _flatten_state(episode)
        action = _flatten_action(episode)
        if state is None or action is None:
            continue

        usable = min(len(state), len(action), max_steps_per_episode)
        if usable < 2:
            continue

        state = state[:usable]
        action = action[:usable]
        difficulty = np.linspace(0.0, 1.0, usable, dtype=np.float32).reshape(-1, 1)

        states.append(state)
        actions.append(action)
        difficulties.append(difficulty)
        episodes_used += 1

        if episodes_used >= max_episodes:
            break

    if not states:
        raise RuntimeError(
            "No usable BridgeData episodes were found. Check that the TFDS bridge "
            "dataset is downloaded and that the observation/action keys match."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        states=np.concatenate(states, axis=0),
        difficulties=np.concatenate(difficulties, axis=0),
        actions=np.concatenate(actions, axis=0),
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract a small BridgeData V2 subset.")
    parser.add_argument("--output", type=Path, default=Path("data/bridge_subset.npz"))
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--split", type=str, default="train[:1%]")
    parser.add_argument("--max-episodes", type=int, default=40)
    parser.add_argument("--max-steps-per-episode", type=int, default=80)
    args = parser.parse_args()

    output_path = extract_bridge_subset(
        output_path=args.output,
        data_dir=args.data_dir,
        split=args.split,
        max_episodes=args.max_episodes,
        max_steps_per_episode=args.max_steps_per_episode,
    )
    print(f"Saved BridgeData subset: {output_path}")


if __name__ == "__main__":
    main()
