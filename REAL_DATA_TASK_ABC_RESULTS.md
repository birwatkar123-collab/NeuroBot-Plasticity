# Real-Data Task A → B → C Results

Verified Kaggle run:
https://www.kaggle.com/code/birwatkar/neurobot-plasticity-task-abc

## BridgeData V2

- Samples: 800 per task
- Task A: object relocation
- Task B: articulated-object open/close
- Task C: reorientation and complex manipulation
- Baseline mean forgetting: 0.1173898545
- Replay mean forgetting: 0.0157140291
- Relative forgetting reduction: 86.6138%
- Baseline final mean MSE: 1.169483995
- Replay final mean MSE: 1.103861718

## Open X-Embodiment

Sequence:
fractal20220817_data → kuka → jaco_play

- Samples: 1500 per task
- Baseline mean forgetting: 0.0828434669
- Replay mean forgetting: 0.0344376475
- Relative forgetting reduction: 58.4305%
- Baseline final mean MSE: 0.9579039780
- Replay final mean MSE: 0.9202817010

## Interpretation

Replay substantially reduced average catastrophic forgetting in both experiments. The benefit was task-dependent: for OXE, Task A forgetting increased slightly with replay while Task B forgetting decreased strongly.

These are lightweight continual-learning policy-regression experiments, not full RT-X/Octo/OpenVLA-scale policy training.
