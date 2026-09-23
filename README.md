# SmolVLA Implementation

This repository contains two SmolVLA experiments built around LeRobot:

- **v1 / ALOHA**: fine-tuning and evaluation on the simulated ALOHA insertion task.
- **v2 / LeIsaac**: fine-tuning on the LeIsaac SO101 pick-and-place dataset, with an IsaacLab evaluation launcher.

The experiments are intentionally kept separate so the ALOHA results remain reproducible while the LeIsaac pipeline evolves.

## Repository Layout

```text
.
├── v1/
│   ├── eval/                  ALOHA evaluation wrappers and recorded metrics
│   ├── SMOLVLA_REPORT_DRAFT.md
│   ├── smolla_lelsaac.md      Original LeIsaac implementation plan
│   └── verify.py
├── v2/
│   ├── convert_leisaac_dataset.sh
│   ├── inspect_leisaac_data.py
│   ├── run_leisaac_training.sh
│   ├── run_isaac_evaluation.sh
│   ├── verify_env.py
│   └── README.md
├── .gitignore
└── README.md
```

The local LeRobot checkout, trained checkpoints, optimizer states, videos, and caches are not committed. They are large generated or third-party artifacts.

## Requirements

Two Conda environments are used because training and IsaacLab simulation have different dependency stacks.

| Environment | Purpose | Required components |
| --- | --- | --- |
| `smolvla` | Dataset inspection and SmolVLA training | Python 3.12, PyTorch with CUDA, LeRobot 0.6.2 source, Transformers 5.5.4, PEFT 0.20.0, PyAV 15.1.0 |
| `leisaac` | IsaacLab simulation evaluation | Python 3.11, Isaac Sim 5.1.0, IsaacLab 2.3.0, LeIsaac 0.4.0, LeRobot async inference dependencies (`grpcio`, `protobuf`) |

The commands below describe the tested arrangement. Package versions can differ, but the LeRobot policy-server and LeIsaac client must use compatible APIs.

### Training Environment: `smolvla`

Activate it before running the v2 data and training scripts:

```bash
conda activate smolvla
cd /path/to/smolVLA/v2
python verify_env.py
```

The checker verifies PyTorch/CUDA, LeRobot, Transformers, PEFT, and PyAV. The training launcher expects a compatible LeRobot source checkout. Set it explicitly when the checkout is outside this repository:

```bash
export LEROBOT_SRC=/path/to/lerobot/src
```

The public repository does not vendor LeRobot. The original local layout used `v1/lerobot/src`.

### Evaluation Environment: `leisaac`

The evaluation environment must contain Isaac Sim, IsaacLab, and LeIsaac. The LeIsaac project documents the installation of IsaacLab 2.3.0 and exposes the optional dependencies used by its policy client:

```bash
conda activate leisaac
cd /path/to/leisaac
pip install -e "source/leisaac[isaaclab,lerobot-async]"
```

Follow the IsaacLab and Isaac Sim installation instructions for the specific machine before running this command. The evaluator entry point is supplied by the external LeIsaac checkout, not this repository.

## v2 LeIsaac Workflow

### 1. Inspect the environment

```bash
cd /path/to/smolVLA/v2
conda activate smolvla
python verify_env.py
```

### 2. Convert and inspect the dataset

The default dataset is `LightwheelAI/leisaac-pick-orange`. The current LeRobot tooling requires the dataset in v3.0 format:

```bash
./convert_leisaac_dataset.sh
python inspect_leisaac_data.py --no-videos
```

The verified dataset contains 60 episodes, two camera inputs (`front` and `wrist`), and 6-dimensional state and action vectors. Conversion uses `--push-to-hub=false` and writes to the local Hugging Face LeRobot cache.

### 3. Fine-tune SmolVLA

```bash
./run_leisaac_training.sh
```

The default configuration uses CUDA, batch size 32, 20,000 steps, pretrained SmolVLA VLM weights, a frozen vision encoder, and expert-only training. Override settings with environment variables:

```bash
STEPS=100 BATCH_SIZE=4 OUTPUT_DIR=/tmp/smolvla_smoke ./run_leisaac_training.sh
```

Checkpoints are written to `outputs/train/leisaac_pick_orange/`, which is ignored by Git.

### 4. Evaluate in IsaacLab

The LeIsaac evaluator uses LeRobot asynchronous inference. Start a compatible LeRobot SmolVLA policy server in the `smolvla` environment, then run the LeIsaac `policy_inference.py` entry point in the `leisaac` environment. The policy server must load:

```text
outputs/train/leisaac_pick_orange/checkpoints/020000/pretrained_model
```

The included launcher is a configurable adapter for task-specific IsaacLab evaluators:

```bash
export ISAACLAB_EVAL_CMD='python /path/to/evaluate_policy.py --checkpoint {checkpoint} --num_envs {batch_size} --episodes {episodes} --output_dir {output_dir} --seed {seed}'
./run_isaac_evaluation.sh
```

See [`v2/README.md`](v2/README.md) for evaluator details and placeholder handling. IsaacLab evaluation is not runnable from this repository alone because the simulator and LeIsaac task package are external dependencies.

## v1 ALOHA Results

The v1 experiment fine-tuned SmolVLA for 20,000 steps on `lerobot/aloha_sim_insertion_human` and achieved 2 successful insertions out of 20 simulated episodes, or a 10% success rate. The full report is in [`v1/SMOLVLA_REPORT_DRAFT.md`](v1/SMOLVLA_REPORT_DRAFT.md).

## Reproducibility Notes

- Record the dataset revision, random seed, checkpoint, and environment versions for each run.
- Do not interpret training loss as closed-loop task success; evaluate rollouts separately.
- The LeIsaac language instruction must match the dataset task text: `Grab orange and place into plate` for the default dataset.
- Generated checkpoints and videos are intentionally excluded from Git. Store them separately or publish them through an artifact store.