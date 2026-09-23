#!/usr/bin/env python3
"""Run SmolVLA evaluation on the ALOHA insertion environment."""

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs"
    / "train"
    / "smolvla_aloha_insertion"
    / "checkpoints"
    / "020000"
    / "pretrained_model"
)
DEFAULT_RESULTS = PROJECT_ROOT / "eval" / "results" / "020000"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    results = args.results.expanduser().resolve()
    results.mkdir(parents=True, exist_ok=True)

    command = [
        "python",
        str(PROJECT_ROOT / "eval" / "aloha_eval_entrypoint.py"),
        f"--policy.path={checkpoint}",
        "--env.type=aloha",
        "--env.task=AlohaInsertion-v0",
        f"--eval.n_episodes={args.episodes}",
        "--eval.batch_size=1",
        "--eval.use_async_envs=false",
        f"--policy.device={'cuda' if os.environ.get('CUDA_VISIBLE_DEVICES') != '' else 'cpu'}",
        "--policy.use_amp=false",
        f"--output_dir={results}",
        "--job_name=smolvla_aloha_insertion_020000",
        f"--seed={args.seed}",
    ]

    (results / "command.txt").write_text(shlex.join(command) + "\n", encoding="utf-8")
    log_path = results / "evaluation.log"
    env = os.environ.copy()
    source_root = str(PROJECT_ROOT / "lerobot" / "src")
    env["PYTHONPATH"] = source_root + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("MUJOCO_GL", "egl")

    print(f"Running {args.episodes} episode(s) with checkpoint: {checkpoint}")
    print(f"Results directory: {results}")
    print(f"Full log: {log_path}")
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.run(
            [sys.executable, *command[1:]],
            cwd=PROJECT_ROOT / "lerobot",
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )

    if process.returncode != 0:
        raise SystemExit(
            f"Evaluation failed with exit code {process.returncode}. "
            f"See {log_path}."
        )

    print("Evaluation completed successfully.")


if __name__ == "__main__":
    main()
