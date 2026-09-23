# SmolVLA v2: LeIsaac fine-tuning

This directory is isolated from `v1`, which contains the ALOHA experiment.
The v2 pipeline targets a LeRobot-format LeIsaac SO101 dataset and reuses the
preserved local LeRobot source from `../v1/lerobot/src` without modifying v1.

## First run

Activate the existing environment, then run the checks from this directory:

```bash
conda activate smolvla
python verify_env.py
python inspect_leisaac_data.py
```

The checker automatically adds the preserved local LeRobot source to `PYTHONPATH`.
The inspector is the data-contract gate. Confirm the task instruction, camera
keys, state shape, and action shape before starting training.

The initial target dataset currently reports LeRobot v2.1 format. Convert its
local cache before inspecting or training:

```bash
chmod +x convert_leisaac_dataset.sh
./convert_leisaac_dataset.sh
python inspect_leisaac_data.py --no-videos
```

Conversion defaults to `--push-to-hub=false` and does not alter the public Hub
dataset. Set `DATASET_ROOT` to convert a specific local dataset directory.

## Training

```bash
chmod +x run_leisaac_training.sh run_isaac_evaluation.sh
./run_leisaac_training.sh
```

For the converted default dataset, the launcher automatically uses
`$HOME/.cache/huggingface/lerobot/LightwheelAI/leisaac-pick-orange`. Set
`DATASET_ROOT` when using another local dataset.

Override settings with environment variables, for example:

```bash
DATASET_ID=CoRL2026-CSI/IsaacLab-SO101-Phase1PlusPhase2-pull_cube-100episode-10fps \
STEPS=100 BATCH_SIZE=4 ./run_leisaac_training.sh
```

## Evaluation

The training launcher is directly compatible with the local LeRobot CLI. The
IsaacLab evaluation entry point varies by the LeIsaac task package and is not
present in the preserved v1 repository, so the evaluation launcher requires its
command explicitly:

```bash
export ISAACLAB_EVAL_CMD='python path/to/leisaac_evaluate.py'
BATCH_SIZE=64 EPISODES=256 ./run_isaac_evaluation.sh
```

For evaluators with different argument names, use placeholders so the
launcher does not append incompatible flags:

```bash
export ISAACLAB_EVAL_CMD='python path/to/evaluate_policy.py --checkpoint {checkpoint} --num_envs {batch_size} --episodes {episodes} --output_dir {output_dir} --seed {seed}'
./run_isaac_evaluation.sh
```

The launcher records the command and log under `eval_videos/`. The external
evaluator must implement the task's actual IsaacLab environment, policy action
adapter, success metric, and video recorder.

The `smolvla` environment must provide PyAV (`av`) for LeRobot dataset video
handling. The current environment has PyAV installed; `python verify_env.py`
checks this dependency before training.
