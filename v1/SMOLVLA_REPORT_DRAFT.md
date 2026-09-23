# SmolVLA Implementation and Fine-Tuning Report

## 1. Overview

This project implemented and evaluated SmolVLA, a lightweight vision-language-action model for robotic manipulation. The implementation was built on top of the LeRobot framework and evaluated on the simulated ALOHA insertion task.

The main objective was to fine-tune SmolVLA on the `lerobot/aloha_sim_insertion_human` dataset and evaluate whether the resulting policy could insert a peg into a socket in simulation.

The final pipeline successfully:

- Loaded the SmolVLA architecture and pretrained VLM weights.
- Loaded the ALOHA insertion dataset.
- Fine-tuned the model for 20,000 steps.
- Saved and reloaded the resulting checkpoint.
- Evaluated the policy in the ALOHA MuJoCo simulator.
- Produced rollout videos and quantitative success metrics.

The final corrected evaluation achieved a 10% success rate over 20 simulated episodes.

## 2. Software and Hardware Environment

The project used the LeRobot repository with native SmolVLA support.

Main software components:

- Python environment: Conda environment named `smolvla`
- LeRobot version: `0.6.2`
- PyTorch version: `2.11.0`
- Transformers version: `5.5.4`
- PEFT version: `0.20.0`
- ALOHA simulator: `gym-aloha`
- Physics engine: MuJoCo
- Rendering backend: EGL for headless GPU rendering

The training and evaluation were run on an NVIDIA H100 NVL GPU. Training used CUDA, while MuJoCo evaluation used headless EGL rendering because the machine did not have an X11 display available.

## 3. SmolVLA Architecture

SmolVLA combines a pretrained vision-language model with an action expert. The policy receives:

- An image observation from the ALOHA top camera.
- The current 14-dimensional robot state.
- A natural-language task instruction.

It predicts a 14-dimensional robot action. The dataset and simulator use the following observation and action structure:

- Image: `observation.images.top`, originally 480 x 640 RGB.
- State: `observation.state`, dimension 14.
- Action: `action`, dimension 14.
- Task instruction: `Insert the peg into the socket.`

Images are resized internally to 512 x 512 with padding. State and action values use mean-standard-deviation normalization, while visual inputs use the identity normalization mode.

## 4. Dataset

The training dataset was:

```text
lerobot/aloha_sim_insertion_human
```

Dataset statistics:

- Episodes: 50
- Frames: 25,000
- Camera inputs: one top camera
- State dimension: 14
- Action dimension: 14
- Task instruction: `Insert the peg into the socket.`

The dataset was used without a held-out evaluation split. Therefore, the final evaluation was performed in the ALOHA simulator rather than on a separate dataset split.

## 5. Initial Implementation and Verification

The initial verification script loaded the public base model `lerobot/smolvla_base`. It was then updated to load a local checkpoint and accept a checkpoint path as a command-line argument.

The checkpoint verification confirmed that the saved model could be loaded and contained approximately 450 million parameters.

A local evaluation wrapper was created at:

```text
/home/ashin/smolVLA/eval/run_evaluation.py
```

The wrapper handles:

- Checkpoint selection.
- Number of evaluation episodes.
- Result directory creation.
- CUDA selection.
- Headless MuJoCo configuration.
- Saving logs, metrics, commands, and rollout videos.

## 6. Debugging and Configuration Corrections

Several configuration issues were identified during development.

### 6.1 Dependency mismatch

The base environment contained an incompatible PEFT installation. The required PEFT package was installed in the `smolvla` Conda environment rather than in the base environment.

### 6.2 Incorrect VLM initialization

The first training configuration used:

```json
"load_vlm_weights": false
```

This created the VLM architecture without loading the pretrained SmolVLM weights. The run completed, but it was not a proper fine-tune of the pretrained SmolVLA model.

### 6.3 Incorrect policy loading mode

Using `--policy.path=lerobot/smolvla_base` loaded the complete base policy configuration, including its SO100 feature schema. This conflicted with the ALOHA dataset, which uses one top camera and 14-dimensional state and action vectors.

The corrected approach was:

```bash
--policy.type=smolvla
--policy.pretrained_path=lerobot/smolvla_base
--policy.load_vlm_weights=true
```

This allowed the dataset to define the ALOHA input and output features while initializing the model from pretrained SmolVLA weights.

### 6.4 Task-language mismatch during evaluation

The dataset uses the instruction:

```text
Insert the peg into the socket.
```

The ALOHA simulator exposed only the task name:

```text
insertion
```

LeRobot therefore passed `insertion` to the policy during evaluation. This did not match the language used during training and resulted in an initial evaluation score of 0% with zero reward.

A local evaluation entry point was added at:

```text
/home/ashin/smolVLA/eval/aloha_eval_entrypoint.py
```

This entry point supplies the exact dataset instruction to the environment before launching LeRobot evaluation.

## 7. Fine-Tuning Configuration

The corrected training run used:

```bash
conda run -n smolvla lerobot-train \
  --policy.type=smolvla \
  --policy.pretrained_path=lerobot/smolvla_base \
  --policy.load_vlm_weights=true \
  --policy.push_to_hub=false \
  --dataset.repo_id=lerobot/aloha_sim_insertion_human \
  --dataset.video_backend=pyav \
  --batch_size=32 \
  --steps=20000 \
  --save_freq=5000 \
  --policy.scheduler_decay_steps=20000 \
  --output_dir=outputs/train/smolvla_aloha_insertion_pretrained \
  --job_name=smolvla_aloha_insertion_pretrained \
  --policy.device=cuda \
  --policy.freeze_vision_encoder=true \
  --policy.train_expert_only=true
```

Important training settings:

- Training steps: 20,000
- Batch size: 32
- Vision encoder: frozen
- Action expert: trained
- VLM weights: loaded from the pretrained SmolVLA model
- Checkpoint frequency: every 5,000 steps
- Learning-rate decay steps: 20,000

The final checkpoint was saved at:

```text
outputs/train/smolvla_aloha_insertion_pretrained/checkpoints/020000/pretrained_model
```

The `last` checkpoint pointer correctly referenced checkpoint `020000`.

## 8. Training Results

The corrected training run completed successfully.

- Total steps: 20,000
- Training duration: approximately 39 minutes 53 seconds
- Final training loss: approximately 0.012
- Total model parameters: approximately 450 million
- Learnable parameters: approximately 100 million

The low training loss shows that the model fit the demonstration data well. However, training loss alone does not guarantee successful closed-loop behavior in the simulator, so a separate rollout evaluation was required.

## 9. Evaluation Procedure

The policy was evaluated using the LeRobot ALOHA environment:

```text
AlohaInsertion-v0
```

Evaluation settings:

- Episodes: 20
- Batch size: 1
- Maximum episode length: 400 steps
- Device: CUDA
- Rendering: headless MuJoCo EGL
- Task instruction: `Insert the peg into the socket.`

The evaluator saved:

- Aggregate metrics in `eval_info.json`.
- Full logs in `evaluation.log`.
- The executed command in `command.txt`.
- Rollout videos in the `videos` directory.

## 10. Final Evaluation Results

The corrected evaluation achieved:

| Metric | Result |
|---|---:|
| Number of episodes | 20 |
| Successful episodes | 2 |
| Success rate | 10% |
| Average episode reward | 216.25 |
| Average maximum reward | 2.05 |
| Total evaluation time | 223 seconds |
| Average episode time | 11.15 seconds |

The two successful rollout videos were:

- `eval/results/pretrained_020000_tasktext/videos/aloha_0/eval_episode_5.mp4`
- `eval/results/pretrained_020000_tasktext/videos/aloha_0/eval_episode_6.mp4`

The complete result file is:

```text
eval/results/pretrained_020000_tasktext/eval_info.json
```

## 11. Interpretation

The final result demonstrates that the complete SmolVLA fine-tuning and evaluation pipeline works end to end. The policy is capable of completing the ALOHA insertion task in some episodes, but its performance is not yet reliable.

The improvement from 0% to 10% after correcting the task instruction shows that language conditioning was an important part of the evaluation contract. The policy also achieved nonzero reward in many unsuccessful episodes, indicating partial progress toward the insertion task.

The result should not be described as state-of-the-art performance. It is better described as a working proof-of-concept implementation with measurable but limited task success.

## 12. Limitations

The current implementation has several limitations:

1. The evaluation used only 20 episodes, so the success-rate estimate has substantial statistical uncertainty.
2. No held-out dataset split was used during training.
3. Only one camera view was used.
4. The vision encoder was frozen during fine-tuning.
5. The training dataset contained 50 demonstrations, which may not cover enough task variation.
6. The final success rate was only 10%.
7. The simulator and dataset action conventions should be validated further before drawing conclusions about real-robot performance.
8. The current evaluation wrapper uses a local task-description override specific to the ALOHA insertion task.

## 13. Future Work

Potential improvements include:

- Train with a larger and more varied demonstration dataset.
- Collect additional object and initial-pose variations.
- Evaluate more than 20 episodes using fixed and documented random seeds.
- Unfreeze the vision encoder and compare performance.
- Compare checkpoints at 5k, 10k, 15k, and 20k steps.
- Add a held-out validation split and track validation action loss.
- Inspect successful and unsuccessful rollout videos systematically.
- Compare the fine-tuned checkpoint against the base SmolVLA model using the same task instruction.
- Validate action scaling and joint ordering against the simulator implementation.
- Evaluate the policy on a real robot only after simulator behavior is reliable.

## 14. Conclusion

SmolVLA was successfully integrated with LeRobot, fine-tuned on the ALOHA insertion dataset, and evaluated in the MuJoCo simulator. The final corrected experiment loaded pretrained VLM weights, used the correct ALOHA feature schema, supplied the correct natural-language task instruction, and completed 20,000 training steps.

The final policy achieved 2 successful insertions out of 20 simulated episodes, corresponding to a 10% success rate. This demonstrates a working end-to-end implementation while also showing that further data, training, and evaluation improvements are needed for robust performance.
