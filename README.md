# MSSPT

## Overview
This repository contains the latest version (v2) of the code to reproduce the experiments and results of the paper "[Pre-Training Music Classification Models via Music Source Separation](https://arxiv.org/pdf/2310.15845.pdf)", submitted to EUSIPCO-2024. In short, we propose pre-training U-Nets with a music source separation objective, and then appending a classification frontend, in order to jointly train them in downstream music classification tasks. Experimental results in two widely used music classification datasets, [Magna-Tag-A-Tune](https://mirg.city.ac.uk/codeapps/the-magnatagatune-dataset) and [FMA](https://github.com/mdeff/fma) indicate that the proposed strategy can prove beneficial for music classification tasks.
Additional features, compared to [v1](https://github.com/cgaroufis/MSSPT/tree/v1), include:
- Incorporation of the AST backend, apart from convolutional ones.
- Support for pre-training independently the backend network, or using a pre-trained set of weights for the backend for finetuning.
Code for accessing pre-trained models will be made available within the next few days; in the meantime, you can access the models included in v1 from [here](https://github.com/cgaroufis/MSSPT/tree/v1).
