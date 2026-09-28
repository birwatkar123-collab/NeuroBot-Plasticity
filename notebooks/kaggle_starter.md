# Kaggle Notebook Starter

Use this as the structure for your first public Kaggle notebook.

## 1. Title

Neuroplasticity-Inspired Robot Training with Curriculum Learning and Replay

## 2. Problem

Robots often require large amounts of task data to learn reliable behavior.
This project tests whether neuroplasticity-inspired training principles can
improve robot action prediction.

## 3. Neuroscience Inspiration

- repetition strengthens useful behavior
- gradual practice helps learning
- memory replay helps retain older skills
- error-focused practice supports adaptation

## 4. Machine Learning Implementation

- baseline behavior cloning
- curriculum learning from easy to hard samples
- replay buffer for previous hard examples
- high-error resampling

## 5. Experiment

Run:

```python
!python src/train.py
!python src/continual_learning.py
!python src/replay_sweep.py
```

Show:

```python
from IPython.display import Image
Image("/kaggle/working/NeuroBot-Plasticity/outputs/learning_curves.png")
Image("/kaggle/working/NeuroBot-Plasticity/outputs/continual_learning_forgetting.png")
Image("/kaggle/working/NeuroBot-Plasticity/outputs/replay_sweep_tradeoff.png")
```

## 6. Next Dataset Upgrade

Replace the synthetic data function with a loader for a small BridgeData V2
subset. Keep the same baseline-vs-plasticity comparison so the research
question remains clear.

Example commands:

```python
!python src/bridgedata_loader.py --output data/bridge_subset.npz --split "train[:1%]" --max-episodes 40
!python src/train_bridge_subset.py --data data/bridge_subset.npz
```

If you attach a Kaggle dataset that already contains a TensorFlow Datasets cache,
pass the cache path:

```python
!python src/bridgedata_loader.py --data-dir /kaggle/input/YOUR_TFDS_DIR --output data/bridge_subset.npz
```

## 7. Expected Discussion

The plasticity-inspired method may not always beat the baseline immediately.
That is still scientifically useful: it lets you discuss where curriculum,
replay size, task ordering, and hard-example sampling need tuning.

The continual-learning result is the stronger first story: replay consolidation
reduces forgetting of an older robot task, while creating a tradeoff with fast
adaptation to the new task.

The replay sweep makes this publishable because it shows the relationship
between replay amount and the stability-plasticity tradeoff.
