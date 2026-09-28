# GitHub Publishing Checklist

## 1. Create Repository

Create a new GitHub repository named:

```text
NeuroBot-Plasticity
```

Recommended settings:

- visibility: public
- license: MIT
- README: do not auto-generate, because this repo already has one

## 2. Add Remote

After creating the empty GitHub repository, run:

```bash
git remote add origin https://github.com/YOUR_USERNAME/NeuroBot-Plasticity.git
git branch -M main
git push -u origin main
```

## 3. Repository Description

Suggested description:

```text
Neuroplasticity-inspired robot task learning with replay consolidation and continual-learning experiments.
```

## 4. Topics

Suggested GitHub topics:

```text
robot-learning, neuroplasticity, continual-learning, replay-buffer, pytorch, robotics, machine-learning
```

## 5. First Release

Create release `v0.1.0` after pushing.

Release title:

```text
Local prototype: replay consolidation and stability-plasticity tradeoff
```

Release notes:

```text
Initial local prototype with synthetic robot-style experiments, continual-learning evaluation, replay-fraction sweep, report, reproduction guide, and Kaggle starter notebook.
```
