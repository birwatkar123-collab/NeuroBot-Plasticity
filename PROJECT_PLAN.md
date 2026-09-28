# Project Plan

## Phase 1: Runnable Prototype

Goal: prove the experiment design works end to end.

- Build a robot-style action prediction task.
- Train a baseline neural network.
- Train a neuroplasticity-inspired version.
- Compare validation loss.
- Save a result graph.
- Add a continual-learning experiment to measure forgetting.
- Add replay-fraction sweep to quantify the stability-plasticity tradeoff.

## Phase 2: Public Dataset Integration

Recommended first dataset: BridgeData V2.

Why BridgeData V2:

- robot manipulation focused
- task/language labels
- easier to explain than very large mixed datasets
- more realistic for a first Kaggle project

Integration plan:

1. Download or mount a small BridgeData V2 subset.
2. Convert observations into model inputs:
   - image features or low-resolution images
   - robot state
   - action vectors
   - task label
3. Keep the same training comparison:
   - baseline shuffled training
   - curriculum plus replay training

Implemented starter scripts:

- `src/bridgedata_loader.py` extracts state/action pairs from a local TFDS/RLDS
  BridgeData cache.
- `src/train_bridge_subset.py` trains the baseline-vs-plasticity comparison on
  the extracted `.npz` subset.

## Phase 3: Kaggle Version

Create a Kaggle notebook that:

- installs dependencies
- loads a small dataset subset
- trains both models
- displays plots
- explains the neuroplasticity mapping
- optionally extracts a BridgeData subset from an attached Kaggle dataset/TFDS cache

## Phase 4: Public Release

Publish:

- GitHub repository
- Kaggle notebook
- Hugging Face demo
- short article explaining the method and results

## Evaluation Metrics

- validation mean squared error
- final loss difference
- training stability
- forgetting after training on harder tasks
- generalization to held-out difficulty levels
- stability-plasticity tradeoff between old-task retention and new-task learning

## Current Experimental Finding

The first continual-learning result shows that replay consolidation reduces
forgetting of Task A after training on Task B. It also increases the final loss
on Task B, which creates a useful research problem: how much replay protects old
skills without slowing adaptation too much?

Replay sweep result:

- replay fraction `0.10`: 19.04% less forgetting
- replay fraction `0.20`: 35.27% less forgetting
- replay fraction `0.30`: 51.74% less forgetting
- replay fraction `0.45`: 65.82% less forgetting

The result forms a clean stability-plasticity curve: more replay means better
retention of old skills, but slower adaptation to the new task.

Next tuning ideas:

- reduce replay fraction
- use replay only every few batches
- weight new-task examples more heavily
- test prioritized replay instead of random replay
- compare rehearsal, elastic weight consolidation, and adapter-based fine-tuning

## Neuroplasticity Mapping

| Neuroscience principle | Machine learning implementation |
| --- | --- |
| Repetition strengthens pathways | repeated exposure to successful trajectories |
| Gradual skill acquisition | curriculum learning |
| Memory consolidation | replay buffer |
| Error correction | high-error example resampling |
| Transfer | pretraining then fine-tuning on related tasks |
