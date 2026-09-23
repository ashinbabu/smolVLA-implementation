# Copilot Implementation Plan: SmolVLA Fine-Tuning on LeIsaac

## Context for Copilot
Generate Python scripts and bash commands to fine-tune and evaluate the SmolVLA model on the LeIsaac dataset. The user already has a configured conda environment named `smolvla` with core dependencies (PyTorch, LeRobot, etc.) installed from a previous ALOHA implementation.
**Crucial Constraints:**
- Do NOT generate environment creation or package installation commands (no `pip install` or `conda install`).
- Scope all generated files and artifacts to a completely new working directory to isolate this project from the previous ALOHA runs.

---

## Step 1: Workspace Initialization and Verification
**Goal:** Set up the isolated directory and verify the existing `smolvla` environment is ready for GPU-accelerated LeIsaac rollouts.

1. **Workspace Setup:** Generate bash commands to create a new directory (e.g., `~/smolvla_leisaac_workspace`) and navigate into it.
2. **Environment Activation:** Output the command to activate the existing environment (`conda activate smolvla`).
3. **Dependency Checker (`verify_env.py`):** Write a Python script to import and verify the versions of `torch` (checking `cuda.is_available()`), `lerobot`, `transformers`, and the `omni.isaac.lab` (IsaacLab) namespaces. The script should print a clear status report of these existing libraries.

---

## Step 2: LeIsaac Dataset Inspection
**Goal:** Verify the specific observation and action schemas of the LeIsaac SO101 dataset before starting the training loop.

1. **Target Dataset:** Use the Hugging Face repository ID `LightwheelAI/leisaac-pick-orange` (or `CoRL2026-CSI/IsaacLab-SO101-Phase1PlusPhase2-pull_cube-100episode-10fps` as an alternative).
2. **Inspection Script (`inspect_leisaac_data.py`):** Write a script using LeRobot's dataset API to load the dataset and print:
   - The exact string used for the natural language task instruction.
   - The dimensions and names of the image inputs (e.g., `observation.images.wrist`, `observation.images.top`).
   - The dimensionality of the state and action vectors.

---

## Step 3: Fine-Tuning Execution Script
**Goal:** Construct the training command leveraging the existing LeRobot training pipeline.

1. **Training Bash Script (`run_leisaac_training.sh`):** Write the full `lerobot-train` CLI command.
2. **Required Flags:**
   - Policy type: `smolvla` initialized with pretrained weights (`--policy.load_vlm_weights=true`).
   - Freeze the vision encoder to preserve pretrained visual features and fit the training on a single GPU.
   - Dataset: `LightwheelAI/leisaac-pick-orange`.
   - Output directory: Map to a local folder within the new workspace (e.g., `outputs/train/leisaac_pick_orange`).
   - Hardware: Ensure it is configured for CUDA execution.

---

## Step 4: GPU-Accelerated Parallel Evaluation
**Goal:** Evaluate the fine-tuned checkpoint using IsaacLab's high-throughput parallel environments.

1. **Evaluation Script (`run_isaac_evaluation.sh`):** Write the evaluation command pointing to the newly trained checkpoint.
2. **Parallel Configuration:** Set the evaluation batch size to run multiple environments concurrently (e.g., `--eval.batch_size=64` or higher, depending on VRAM) to leverage IsaacLab's GPU-accelerated physics engine.
3. **Logging & Rendering:** Ensure the script is configured to capture rollout metrics, log the success rate, and save MP4 videos of the simulated parallel episodes into an `eval_videos` directory.