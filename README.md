# NeuroBot-Plasticity

**Neuroplasticity-inspired continual learning for robotics**

NeuroBot-Plasticity studies whether ideas inspired by biological neuroplasticity — especially repetition, memory consolidation, gradual adaptation, and replay — can help robot-learning systems learn new tasks while retaining previously acquired skills.

The central problem is **catastrophic forgetting**: after a model learns Task A, then Task B, then Task C, performance on earlier tasks can deteriorate. This project compares ordinary sequential learning with **replay consolidation**, where selected experiences from older tasks are mixed into later training.

## Research Question

> Can neuroplasticity-inspired replay reduce catastrophic forgetting in robot policies while preserving the ability to learn new tasks?

## Latest Milestone — Real Robot Data A → B → C

The project has now completed successful **Task A → Task B → Task C continual-learning experiments on two public robotics sources**:

1. **BridgeData V2** — semantic continual learning within one manipulation dataset.
2. **Open X-Embodiment** — cross-dataset/domain continual learning across RT-1, Kuka, and Jaco Play.

The successful Kaggle run used **2× Tesla T4 GPUs** and completed in about **3 minutes**.

Kaggle run:

```text
https://www.kaggle.com/code/birwatkar/neurobot-plasticity-task-abc
```

### BridgeData V2 Task Sequence

The Bridge experiment grouped natural-language instructions into three sequential skill families:

- **Task A — object relocation:** put / place / move / transfer / bring / insert
- **Task B — articulated-object manipulation:** open / close
- **Task C — reorientation and complex manipulation:** flip / turn / rotate / fold / sweep / stack / push / pull / slide / pour

The loader scanned **396 episodes**, all with non-empty language instructions, and collected **800 samples per task**.

Example instructions included:

- "put carrot on plate"
- "open microwave"
- "close fridge"
- "flip cup upright"
- "turn faucet front to left"
- "slide the yellow cloth..."

### BridgeData V2 Results

| Metric | Sequential baseline | Replay consolidation |
| --- | ---: | ---: |
| Task A forgetting | 0.0945 | **-0.0028** |
| Task B forgetting | 0.1403 | **0.0342** |
| Mean forgetting | 0.1174 | **0.0157** |
| Final mean MSE | 1.1695 | **1.1039** |

**Relative mean-forgetting reduction with replay: 86.61%.**

The slightly negative Task A forgetting value means Task A validation error improved slightly after later-task training rather than worsening.

### Open X-Embodiment Task Sequence

The OXE experiment used a cross-dataset/domain sequence:

- **Task A:** `fractal20220817_data` (RT-1)
- **Task B:** `kuka`
- **Task C:** `jaco_play`

Each task used **1,500 real trajectory samples** with a common lightweight representation and a standardized 7-D end-effector-compatible action target.

### Open X-Embodiment Results

| Metric | Sequential baseline | Replay consolidation |
| --- | ---: | ---: |
| Task A forgetting | **0.0277** | 0.0365 |
| Task B forgetting | 0.1380 | **0.0324** |
| Mean forgetting | 0.0828 | **0.0344** |
| Final mean MSE | 0.9579 | **0.9203** |

**Relative mean-forgetting reduction with replay: 58.43%.**

Replay did not improve every individual task: OXE Task A forgetting increased slightly, while Task B retention improved substantially. Overall mean forgetting still decreased, illustrating the **stability-plasticity tradeoff** rather than a universally positive replay effect.

## Previous Public-Dataset Result

The earlier public-data proof of concept used **UCI Robot Execution Failures**.

| Metric | Sequential baseline | Replay consolidation |
| --- | ---: | ---: |
| Task A accuracy after Task B | 0.3704 | 0.7407 |
| Task B final accuracy | 0.6000 | 0.6000 |
| Forgetting | 0.4074 | 0.0370 |

This experiment first showed that replay could reduce forgetting on real robot execution data. The BridgeData V2 and Open X-Embodiment experiments extend the idea to richer robot trajectories and an A → B → C protocol.

## Synthetic Continual-Learning Baseline

The project also includes a controlled synthetic environment for method development and replay sweeps.

| Replay fraction | Task A forgetting | Task B final MSE | Forgetting reduction |
| --- | ---: | ---: | ---: |
| 0.00 | 0.9377 | 0.0443 | 0.00% |
| 0.10 | 0.7592 | 0.0581 | 19.04% |
| 0.20 | 0.6070 | 0.0845 | 35.27% |
| 0.30 | 0.4525 | 0.1369 | 51.74% |
| 0.45 | 0.3205 | 0.2169 | 65.82% |

This demonstrated the expected tradeoff: more replay preserved Task A better, but made adaptation to Task B slower.

## Neuroplasticity Mapping

| Neuroscience idea | Machine-learning implementation |
| --- | --- |
| Repetition strengthens useful pathways | repeated training exposures |
| Memory consolidation | replay buffer |
| Gradual skill acquisition | curriculum / sequential learning |
| Error correction | high-error example resampling |
| Transfer between related skills | shared representation across tasks |
| Stability-plasticity balance | retention vs new-task adaptation |

## Project Evolution

The project now includes:

1. Synthetic single-task action prediction.
2. Curriculum learning and error-focused replay.
3. Two-task continual learning.
4. Replay-fraction stability-plasticity sweep.
5. UCI Robot Execution Failures public-data experiment.
6. **BridgeData V2 real-data Task A → B → C experiment.**
7. **Open X-Embodiment cross-dataset Task A → B → C experiment.**

## Current Repository Files

- `src/train.py` — baseline and neuroplasticity-inspired synthetic training
- `src/continual_learning.py` — replay consolidation experiment
- `src/replay_sweep.py` — replay-fraction tradeoff sweep
- `src/uci_robot_failures.py` — UCI Robot Execution Failures experiment
- `src/bridgedata_loader.py` — original BridgeData subset loader
- `src/train_bridge_subset.py` — original BridgeData subset training scaffold
- `kaggle_task_abc/` — successful BridgeData/OXE Task A→B→C Kaggle package
- `termux/` — Termux/Kaggle CLI control scripts
- `REPORT.md` — project report
- `REPRODUCE.md` — reproduction instructions
- `PROJECT_PLAN.md` — project roadmap
- `CITATION.cff` — citation metadata
- `LICENSE` — MIT license

## Kaggle

Original project notebook:

```text
https://www.kaggle.com/code/birwatkar/neurobot-plasticity
```

Successful real-data A → B → C run:

```text
https://www.kaggle.com/code/birwatkar/neurobot-plasticity-task-abc
```

## Interpretation

Across the current experiments, replay consistently reduces **average** catastrophic forgetting, but the benefit varies by task.

The strongest current results are:

- **BridgeData V2:** 86.61% reduction in mean forgetting
- **Open X-Embodiment:** 58.43% reduction in mean forgetting
- **UCI Robot Execution Failures:** forgetting reduced from 0.4074 to 0.0370

These results support continued investigation of replay-based memory consolidation for continual robot learning.

## Important Scope / Limitation

This work is a **lightweight continual-learning policy-regression experiment**. It uses compact state, language, and/or low-resolution visual representations with standardized action targets.

It is **not** full RT-X, Octo, OpenVLA, or large vision-language-action model pretraining, and the current results should not be interpreted as evidence that replay will automatically improve every robot policy or task.

The next research stage should test:

- multiple random seeds and confidence intervals
- larger task samples
- alternative replay fractions and prioritized replay
- Elastic Weight Consolidation (EWC)
- adapter-based continual learning
- stronger visual encoders
- larger policy architectures
- real robot evaluation if hardware becomes available

## Repository Status

**Version 0.3.0 — successful real-data continual-learning milestone**

NeuroBot-Plasticity has progressed from synthetic experiments and a small public-data proof of concept to successful A → B → C experiments on **BridgeData V2** and **Open X-Embodiment**.

## Quick Start

```bash
pip install -r requirements.txt
python src/train.py
python src/continual_learning.py
python src/replay_sweep.py
python src/uci_robot_failures.py
```

## Author

**Gaurav Birwatkar**

## License

MIT
