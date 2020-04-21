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
describe here later.

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
| <sub>[SRCNN](http://mmlab.ie.cuhk.edu.hk/projects/SRCNN.html)</sub>| <sub>291</sub>| <sub>30.48/0.8628</sub> |<sub>27.50/0.7513</sub>|<sub>26.90/0.7101</sub>|<sub>24.52/0.7221</sub>|<sub>27.58/0.8555</sub>|
| <sub>[EDSR](https://github.com/thstkdgus35/EDSR-PyTorch)</sub> | <sub>DIV2K</sub> | <sub>32.46/0.8968</sub> | <sub>28.80/0.7876</sub> | <sub>27.71/0.7420</sub> | <sub>26.64/0.8033</sub> | <sub>31.02/0.9148</sub> |
| <sub>[RCAN](https://github.com/yulunzhang/RCAN)</sub> |  <sub>DIV2K</sub> | <sub>32.63/0.9002</sub> | <sub>28.87/0.7889</sub> | <sub>27.77/0.7436</sub> | <sub>26.82/ 0.8087</sub>| <sub>31.22/ 0.9173</sub>|
|<sub>RRDB(ours)</sub>| <sub>DF2K</sub>| <sub>**32.73/0.9011**</sub> |<sub>**28.99/0.7917**</sub> |<sub>**27.85/0.7455**</sub> |<sub>**27.03/0.8153**</sub> |<sub>**31.66/0.9196**</sub>|

#### The NTIRE2020 RWSR Challenge Results

## Visual Results
describe here later.

## Code Acknowledgement
describe here later.
