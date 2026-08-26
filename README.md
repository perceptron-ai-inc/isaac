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

The Isaac training and inference code lives in a pinned LeRobot submodule:

```text
isaac/
├── lerobot/       # Isaac 0.5 training and inference integration
├── LICENSE
└── README.md
```

Clone the repository:

```bash
git clone --recurse-submodules https://github.com/perceptron-ai-inc/isaac.git
cd isaac/lerobot
```

## Installation

In LeRobot, the policy type is named `perceptron_isaac`. From the `lerobot` directory:

```bash
uv sync --locked --extra perceptron_isaac
uv run python -c "from lerobot.policies.perceptron_isaac import PerceptronIsaacPolicy"
```

For the CUDA runtime:

```bash
uv sync --locked --extra perceptron_isaac_cuda
```

For fine-tuning:

```bash
uv sync --locked --extra perceptron_isaac --extra training --extra peft
uv run lerobot-train --help
```

mHarmony and TensorStream are maintained separately from this repository. The `perceptron_isaac` extra does not yet declare the mHarmony runtime, so a clean checkout is not enough for rendering, training, or inference. The provenance and policy guides document this boundary and the current deployment requirements.

Model resources are hosted at [PerceptronAI/Isaac-0.5](https://huggingface.co/PerceptronAI/Isaac-0.5).

## Documentation

- [Fine-tuning guide](./lerobot/docs/PERCEPTRON_ISAAC_FINETUNING_GUIDE.md)
- [Policy and runtime guide](./lerobot/src/lerobot/policies/perceptron_isaac/README.md)
- [Source provenance and package boundaries](./lerobot/docs/PERCEPTRON_ISAAC_PROVENANCE.md)
- [LeRobot user guide](./lerobot/AGENT_GUIDE.md)

## License

The code is available under the [Apache License 2.0](./LICENSE). Model weights and artifacts use the terms published in the Hugging Face repository.

For deployment support, contact [support@perceptron.inc](mailto:support@perceptron.inc).
