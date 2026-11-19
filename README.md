<p align="center">
  <a href="https://www.perceptron.inc/" target="_blank" rel="noopener">
    <img src="./assets/banner-light.svg" alt="Perceptron" width="680" />
  </a>
</p>

<div align="center">
  <h1>Isaac 0.5</h1>
  <h3>An open embodied foundation model for robotics</h3>
</div>

<p align="center">
  <a href="https://pub-d90b81cad7254a1aa6b148ac18153c0c.r2.dev/isaac-0.5.pdf"><strong>Paper</strong></a> ·
  <a href="https://huggingface.co/PerceptronAI/Isaac-0.5"><strong>Model &amp; weights</strong></a> ·
  <a href="https://www.perceptron.inc/" target="_blank"><strong>Website</strong></a> ·
  <a href="https://discord.gg/fgBeaACQzE" target="_blank"><strong>Community</strong></a>
</p>

---

Isaac 0.5 is a 36-billion-parameter sparse model for video understanding, embodied reasoning, and robot control. It reads images, video, language instructions, robot state, and past actions. It can answer questions about video, point to and track objects, estimate task progress, and generate robot actions.

We trained Isaac on data from more than 35 robot systems, 100,000 hours of robot experience, one million hours of general video, and three trillion multimodal tokens. Video understanding, spatial grounding, task progress, future-percept prediction, and control were trained together from the start.

The model has separate interfaces for continuous and discrete control. Continuous actions come from a Flow expert and a 36-block diffusion transformer. Discrete actions use a separate vocabulary of 2,048 FAST tokens. During closed-loop control, Isaac predicts the next action chunk while the current one is running, using the latest observation and the commands already sent to the robot.

We also measured how general video changes the amount of robot data needed during training. At a held-out action-loss target of 2.50, increasing general video from 1,000 hours to one million hours reduced the measured teleoperation requirement from about 5,900 hours to 28 hours.

## Using this repository

On Linux, install Git, uv, and a Rust toolchain, and use a CUDA-capable NVIDIA GPU. Clone Isaac with its pinned LeRobot submodule. The policy type is `perceptron_isaac`.

```bash
git clone --recurse-submodules https://github.com/perceptron-ai-inc/isaac.git
cd isaac/lerobot
uv venv --python 3.12
source .venv/bin/activate
uv pip install -c constraints/perceptron_isaac-cu128.txt \
  --index https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match \
  '.[perceptron_isaac,training]'
```

The constraints select the CUDA 12.8 build of PyTorch and the `perceptron_isaac` extra installs the pinned mHarmony runtime. For LIBERO evaluation, install `.[perceptron_isaac,training,libero]` in the same command.

Find checkpoints and inference examples in the model cards. [Isaac 0.5 on Hugging Face](https://huggingface.co/PerceptronAI/Isaac-0.5)

## Fine-tune

From `isaac/lerobot`, set `POLICY_DIR` to the `recorded_policy` directory in the downloaded YAM or SO-100/101 checkpoint and `DATASET_REPO_ID` to a compatible LeRobot dataset. This example uses two GPUs and CPU offload. It saves a checkpoint at the end of training.

```bash
STEPS=10000
torchrun --standalone --nproc_per_node=2 ../scripts/finetune.py \
  --policy.path="$POLICY_DIR" --dataset.repo_id="$DATASET_REPO_ID" \
  --steps="$STEPS" --policy.max_train_steps="$STEPS" --batch_size=1 \
  --dataset.video_backend=pyav --parallelism.dp_shard=-1 \
  --accelerator.mixed_precision=bf16 --accelerator.fsdp.cpu_offload=true \
  --accelerator.fsdp.reduce_dtype=float32 --policy.push_to_hub=false \
  --env_eval_freq=0 --eval_steps=0
```

For a one-step smoke test, set `STEPS=1` and add `--policy.optimizer_warmup_steps=0`. If your dataset uses different camera keys, pass a mapping from dataset keys to policy keys with `--rename_map`.

## Documentation

- [Policy and runtime guide](./lerobot/src/lerobot/policies/perceptron_isaac/README.md)
- [LeRobot user guide](./lerobot/AGENT_GUIDE.md)

## License

The code is available under the [Apache License 2.0](./LICENSE). Model weights and artifacts use the terms published in the Hugging Face repository.

For deployment support, contact [support@perceptron.inc](mailto:support@perceptron.inc).
