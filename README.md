# WGAN-GP: Wasserstein GAN with Gradient Penalty
![Python](https://img.shields.io/badge/Python-3.7%2B-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-1.7%2B-orange?style=for-the-badge&logo=pytorch)
![Weights & Biases](https://img.shields.io/badge/Weights%20%26%20Biases-Enabled-yellow?style=for-the-badge&logo=weightsandbiases)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![image](https://github.com/user-attachments/assets/00681b1a-988d-4d72-88a7-f4ca3bc7a89d)

This project implements a Wasserstein GAN with Gradient Penalty (WGAN-GP) for generating realistic face images using the CelebA dataset.
## Features
- WGAN-GP architecture
- CelebA dataset integration
- Weights & Biases logging
- Modular code structure
- Checkpoint saving and loading
## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/SanshruthR/wgan-gp.git
   cd wgan-gp
   ```
2. Install the required packages:
   ```bash
   pip install torch torchvision tqdm matplotlib wandb
   ```
3. Set up your Weights & Biases account and obtain your API key.
## Usage
1. Update the `config.py` file with your desired parameters.
2. Replace `<ENTER YOUR KEY HERE>` in `main.py` with your W&B API key.
3. Run the training script:
   ```bash
   python main.py
   ```
## Project Structure
- `main.py`: Main script to run the training process
- `model.py`: Contains Generator and Critic model architectures
- `dataset.py`: CelebA dataset loader
- `engine.py`: Training loop and utilities
- `utils.py`: Helper functions
- `config.py`: Configuration parameters
## Results and Reports
For detailed results and visualizations, check out our [Weights & Biases report](https://api.wandb.ai/links/sanshruthr-misc/r0ra4ybu).
| Name            | crit_loss (Max) | gen_loss (Max) | crit_loss (Min) | gen_loss (Min) | Runtime | GPU Type           | batch_size | epoch | lr     | model   | optimizer | crit_loss    | gen_loss    |
|-----------------|-----------------|----------------|-----------------|----------------|---------|--------------------|------------|-------|--------|----------|-----------|--------------|-------------|
| decent-water-2   | -5.9919         | 139.1709       | -93.3319        | -32.2147       |  4.71 hours  | Tesla P100-PCIE-16GB| 200        | 200   | 0.001  | wgan gp  | adam      | -62.4255     | 114.6643    |
| flowing-energy-1 | 10.8647         | 94.0787        | -56.2638        | 0.0535         |  2.51 hours   | Tesla P100-PCIE-16GB| 100        | 100   | 0.0001 | wgan gp  | adam      | -49.3754     | 75.1708     |
## Acknowledgements
- [Wasserstein GAN paper](https://arxiv.org/abs/1701.07875)
- [CelebA Dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

