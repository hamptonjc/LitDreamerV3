# Dreamer-V3 (Re-implementation in PyTorch🔥 / Lightning⚡)

A clean re-implementation of the model-based reinforcement learning algorithm [Dreamer-V3](https://arxiv.org/abs/2301.04104) by Hafner et al. in PyTorch & Lightning (for eductional purposes).

<p align="center">
  <video src="https://github.com/user-attachments/assets/cfa8c76b-c648-42ff-9eaf-5de4c913b961" width="700" controls></video>
</p>

## Usage

1. Setup a config YAML file for the training run. See the ```configurations/``` sub-directory for examples & starting points.

2. Execute training via ```$ python runner.py -cfg /path/to/cfg.yml```

## Features

- 1D & 2D Observations

- Continuous & Discrete Action Spaces

- Custom Concurrent Data Collection (i.e., multiple environments & decoupled from algorithm updates)

- Custom TorchRL ReplayBuffer storage for compressed experiences for efficient data collection.

- BF16 Mixed-precision training & torch.compile support

- Extensive logging to TensorBoard (e.g., in-depth statistic, replay videos)

- Various network sizes (S/M/L/XL/XXL)

- Custom Docker image for execution 🐋


## References

1. [Dreamer-V3 Paper](https://arxiv.org/abs/2301.04104)

2. [Dreamer-V3 Repo](https://github.com/danijar/dreamerv3)

3. [Dreamer-V3 PyTorch](https://github.com/NM512/dreamerv3-torch)

4. [SheepRL](https://github.com/Eclectic-Sheep/sheeprl)

5. [EclecticSheep Dreamer V3 Blog Post](https://eclecticsheep.ai/2023/08/10/dreamer_v3.html)
