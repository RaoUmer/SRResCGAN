# Super-Resolution Residual Convolutional Generative Adversarial Network (SRResCGAN)
A PyTorch implementation of the [SRResCGAN](https://github.com/RaoUmer/SRResCGAN) model as described in the paper [Deep Generative Adversarial Residual Convolutional Networks for Real-World Super-Resolution](https://arxiv.org/). This work is participated in the [NTIRE 2020](https://data.vision.ee.ethz.ch/cvl/ntire20/) RWSR challenges on the [Real-World Super-Resolution](https://arxiv.org/).

#### Abstract
Most current deep learning based single image super-resolution (SISR) methods focus on  designing deeper / wider models to learn the non-linear mapping between low-resolution (LR) inputs and the high-resolution (HR) outputs from a large number of paired (LR/HR) training data. They usually take as assumption that the LR image is a bicubic down-sampled version of the HR image. However, such degradation process is not available in real-world settings i.e. inherent sensor noise, stochastic noise, compression artifacts, possible mismatch between image degradation process and camera device. It reduces significantly the performance of current SISR methods due to real-world image corruptions. To address these problems, we propose a deep Super-Resolution Residual Convolutional Generative Adversarial Network (SRResCGAN) to follow the real-world degradation settings by adversarial training the model with pixel-wise supervision in the HR domain from its generated LR counterpart. The proposed network exploits the residual learning by minimizing the energy-based objective function with powerful image regularization and convex optimization techniques. We demonstrate our proposed approach in quantitative and qualitative experiments that generalize robustly to real input and it is easy to deploy for other down-scaling operators and mobile/embedded devices.

#### Pre-trained Models
| |[DSGAN](https://github.com/ManuelFritsche/real-world-sr/tree/master/dsgan)|[SRResCGAN](https://github.com/RaoUmer/SRResCGAN)|
|---|:---:|:---:|
|NTIRE2020 RWSR|[Source-Domain-Learning](https://github.com/RaoUmer/SRResCGAN)|[SR-learning](https://github.com/RaoUmer/SRResCGAN)|

#### BibTeX
    @inproceedings{UmerCVPRW2020,
        title={Deep Generative Adversarial Residual Convolutional Networks for Real-World Super-Resolution},
        author={Rao Muhammad Umer and Gian Luca Foresti and Christian Micheloni},
        booktitle={CVPR Workshops},
        year={2020},
        }

## Quick Test
#### Dependencies
- [Python 3.7](https://www.anaconda.com/distribution/) (version >= 3.0)
- [PyTorch >= 1.0](https://pytorch.org/) (CUDA version >= 8.0 if installing with CUDA.)
- Python packages:  `pip install numpy opencv-python`

#### Test models
describe here later.

## SRResCGAN Architecture
#### Overall Representative diagram
<p align="center">
  <img height="120" src="figs/srrescgan.png">
</p>

#### SR Generator Network
<p align="center">
  <img height="180" src="figs/generator.png">
</p>

## Quantitative Results
describe here later.
| <sub>Dataset (HR/LR pairs)</sub> | <sub>SR methods</sub> | <sub>#Params</sub> | <sub>PSNR&#x2191;</sub> | <sub>SSIM&#x2191;</sub> | <sub>LPIPS&#x2193;</sub> | <sub>Artifacts</sub> |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| <sub>-</sub>| <sub>-</sub>| <sub>-</sub> |<sub>-</sub>|<sub>-</sub>|<sub>-</sub>|<sub>-</sub>|

#### The NTIRE2020 RWSR Challenge Results
| <sub>Team</sub> | <sub>PSNR&#x2191;</sub> | <sub>SSIM&#x2191;</sub> | <sub>LPIPS&#x2193;</sub> | <sub>MOS&#x2193;</sub> |
|:---:|:---:|:---:|:---:|:---:|
| <sub>-</sub>| <sub>-</sub> |<sub>-</sub>|<sub>-</sub>|<sub>-</sub>|

## Visual Results
describe here later.

## Code Acknowledgement
describe here later.
