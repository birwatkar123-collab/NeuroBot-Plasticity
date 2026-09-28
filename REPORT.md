# NeuroBot-Plasticity: First Project Report

## Abstract

This project explores whether principles inspired by neuroplasticity can improve
robot task learning. The first local experiments use synthetic robot-style
state/action data to test replay consolidation, curriculum learning, and
continual learning. The strongest result appears in a continual-learning setup:
replay consolidation reduces forgetting of an earlier task, but slows adaptation
to a new task. This demonstrates the stability-plasticity tradeoff, a useful
research direction before moving to a public robotics dataset such as BridgeData
V2.

## Motivation

Human and animal nervous systems do not learn a task once and remain fixed.
They adapt through repetition, feedback, memory consolidation, and gradual
practice. Robot learning systems face a related problem: they must learn new
tasks while preserving useful older skills. This project translates that idea
into neural-network training methods.

## Research Questions

1. Can curriculum learning and replay memory improve robot-style action
   prediction?
2. Can replay consolidation reduce catastrophic forgetting when a policy learns
   Task A and then adapts to Task B?
3. How does replay amount affect the tradeoff between retaining old skills and
   learning new ones?

## Method

The current local prototype uses synthetic robot-style data. Each sample
contains:

- a state vector
- a difficulty value
- a target action vector

Two training settings are tested.

### Single-Task Learning

A baseline policy is trained with ordinary shuffled batches. The
neuroplasticity-inspired policy uses:

- curriculum learning from easier to harder samples
- replay of high-error examples
- extra weighting for harder examples

### Continual Learning

The policy first learns Task A. It then learns Task B.

Two strategies are compared:

- sequential baseline: train on Task A, then Task B only
- replay consolidation: train on Task B while replaying examples from Task A

Forgetting is measured as:

```text
Task A loss after Task B training - Task A loss before Task B training
```

Lower forgetting is better.

## Results

### Single-Task Training

| Method | Final validation MSE |
| --- | ---: |
| Baseline | 0.0481 |
| Curriculum + replay | 0.0519 |

The single-task result does not beat the baseline. This is still useful: replay
and curriculum are not automatically better when the task distribution is fixed
and simple.

### Continual Learning

| Metric | Sequential baseline | Replay consolidation |
| --- | ---: | ---: |
| Task A forgetting | 0.9377 | 0.3205 |
| Task B final MSE | 0.0443 | 0.2169 |

Replay consolidation reduces Task A forgetting by about 65.81%, but increases
Task B error. This is the central finding: replay protects old knowledge while
making new-task adaptation slower.

### Replay Sweep

| Replay fraction | Task A forgetting | Task B final MSE | Forgetting reduction |
| --- | ---: | ---: | ---: |
| 0.00 | 0.9377 | 0.0443 | 0.00% |
| 0.10 | 0.7592 | 0.0581 | 19.04% |
| 0.20 | 0.6070 | 0.0845 | 35.27% |
| 0.30 | 0.4525 | 0.1369 | 51.74% |
| 0.45 | 0.3205 | 0.2169 | 65.82% |

As replay fraction increases, forgetting decreases steadily. However, Task B
final error increases. This forms a clear stability-plasticity curve.

## Interpretation

The project supports a useful hypothesis: neuroplasticity-inspired replay is
most valuable when the robot must keep old skills while learning new ones. It is
less useful as a simple replacement for ordinary training on one fixed task.

The result is not "replay is always better." The result is more precise:

> Replay improves memory stability, but too much replay reduces plasticity for
> the new task.

That makes the next research problem clear: find replay schedules that protect
old skills without slowing new learning too much.

## Limitations

- The current results use synthetic robot-style data, not real robot rollouts.
- The action-prediction task is simpler than real manipulation.
- The current model uses vector states, not raw images.
- The replay strategy is basic random rehearsal.

## Next Experiments

1. Replace synthetic data with a small BridgeData V2 subset.
2. Try replay schedules instead of fixed replay fractions.
3. Compare random replay with prioritized replay.
4. Add Elastic Weight Consolidation as another neuroplasticity-inspired method.
5. Add image features from robot camera observations.

## Publication Plan

1. Publish the local prototype on GitHub.
2. Publish the reproducible notebook on Kaggle.
3. Add BridgeData V2 results.
4. Create a short Hugging Face Space or static demo showing the plots.
5. Write a blog-style article explaining the stability-plasticity tradeoff in
   robot learning.
