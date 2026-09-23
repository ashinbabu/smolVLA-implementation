# SmolVLA Implementation

This repository contains two isolated SmolVLA experiments:

- `v1/`: ALOHA insertion fine-tuning and evaluation
- `v2/`: LeIsaac SO101 pick-and-place fine-tuning and evaluation launchers

## v1: ALOHA

The ALOHA implementation, debugging notes, evaluation wrappers, and report are
under `v1/`. The local LeRobot checkout and trained checkpoints are intentionally
not committed because they are large generated/dependency artifacts.

## v2: LeIsaac

The v2 workflow is documented in [`v2/README.md`](v2/README.md). It includes:

- environment verification
- LeIsaac dataset inspection
- LeRobot v2.1-to-v3.0 conversion
- SmolVLA fine-tuning launcher
- IsaacLab evaluation launcher

The trained model checkpoints are excluded from Git. The latest local v2
checkpoint is produced under `v2/outputs/train/`.

## Environment

The scripts use the existing `smolvla` Conda environment for training and the
separate `leisaac` environment plus IsaacLab/Isaac Sim for simulation
evaluation. See `v2/README.md` for the current setup and known prerequisites.