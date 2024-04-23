# MSSPT

## Overview
This repository contains the latest version (v2) of the code to reproduce the experiments and results of the paper "[Pre-Training Music Classification Models via Music Source Separation](https://arxiv.org/pdf/2310.15845.pdf)", submitted to EUSIPCO-2024. In short, we propose pre-training U-Nets with a music source separation objective, and then appending a classification frontend, in order to jointly train them in downstream music classification tasks. Experimental results in two widely used music classification datasets, [Magna-Tag-A-Tune](https://mirg.city.ac.uk/codeapps/the-magnatagatune-dataset) and [FMA](https://github.com/mdeff/fma) indicate that the proposed strategy can prove beneficial for music classification tasks.
Additional features, compared to [v1](https://github.com/cgaroufis/MSSPT/tree/v1), include:
- Incorporation of the AST backend, apart from convolutional ones.
- Support for pre-training independently the backend network, or using a pre-trained set of weights for the backend for fine-tuning.
 
Pre-initialized U-Nets, as well as backbone networks (initialized at both ImageNet-derived weights and dataset-specific ones), are made available within this repository. Code for accessing fine-tuned joint models will be made available within the next few days; in the meantime, you can access the models included in v1 from [here](https://github.com/cgaroufis/MSSPT/tree/v1).

## Framework details
![Screenshot](assets/arch_overview2.pdf)
The proposed framework (depicted in the figure above for the case of a convolutional backend, blue rectangle) is inspired by the [TUne+](https://archives.ismir.net/ismir2022/paper/000007.pdf) architecture, modified to fit a traditional supervised learning framework and adapted into the STFT domain. In essence, it consists of a U-Net network (red rectangle) followed by a classification frontend (green rectangle); the U-Net and the frontend are connected via a convolutional adaptation module (grey rectangle). 
- The U-Net is pre-trained with various music source separation objectives, and is based on the baseline architecture described [here](https://arxiv.org/pdf/2109.05418.pdf).
- For the classification network we experimented with both convolutional and Transformer-based backends. The convolutional frontend is a modification of the VGG-like frontend developed by [Won et al.](https://arxiv.org/pdf/2006.00751.pdf), with 2-stem convolutions at each resolution; for the Transformer case, we use the [AST](https://arxiv.org/pdf/2104.01778.pdf) backend, which follows a typical Transformer encoder architecture.
- Finally, the adaptation module consists of 1x1 convolutions, to align the feature map dimensions, followed by either spectral downsampling (CNN frontend) or a patchification operation with appropriately sized kernels (AST frontend).

