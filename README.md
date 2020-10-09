# FC-DCNN 
![pytorch](https://img.shields.io/badge/pytorch-v1.2.0-green.svg?style=plastic)
![cuda](https://img.shields.io/badge/cuda-v10.0-green.svg?style=plastic)
![cudnn](https://img.shields.io/badge/cudnn-v7.4.2-green.svg?style=plastic)
![python](https://img.shields.io/badge/python-v3.6.10-green.svg?style=plastic)
![colab](https://img.shields.io/badge/online-demo-green.svg?style=plastic)

![Teaser image](./docs/Header.png)
##### A densely connected neural network for stereo estimation
Dominik Hirner, Friedrich Fraundorfer

An implementation of a lightweight fully-convolutional densely connected neural network method for disparity estimation.
This method has been submitted to the **ICPR 2020** conference and is currently being peer-reviewed.
A demo of this network is available online. 
[colab](https://colab.research.google.com/drive/1i5k-YTCsmQC8IIURBh802oKeyZP_ZSHG)

The whole project is in pure python 3.6 with the exception of the jupyter-notebook inference script. For performace reasons the post-processing is done in cython which can be installed via: 

  $ pip install Cython

This repository contains

- jupyter notebooks for training and inference of disparity via a stereo-pair
- pure python3.6 code for training and inference
- trained weights for many publicly available datasets

The network was trained and tested on a GeForce RTX 2080 with 11GB of RAM.
The code is released under the BSD 2-Clause license. Please cite our paper (link) if you use code from this repository in your work.

## Trained weights

[Middlebury](https://drive.google.com/file/d/1DiuY47dnM4PUgzDg8SQZyZUyC09pVoDO/view?usp=sharing) |
[Kitti](https://drive.google.com/file/d/1Nce9yJeAs7u8Y1GpbJpgzP_RbLg9S3Rs/view?usp=sharing) |
[ETH3D](https://drive.google.com/file/d/19QxmKWHNgNnjOUFBPnVgGG3r3mkHY1nW/view?usp=sharing) |
