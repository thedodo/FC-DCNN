# FC-DCNN 
![pytorch](https://img.shields.io/badge/pytorch-v1.2.0-green.svg?style=plastic)
![cuda](https://img.shields.io/badge/cuda-v10.0-green.svg?style=plastic)
![cudnn](https://img.shields.io/badge/cudnn-v7.4.2-green.svg?style=plastic)
![python](https://img.shields.io/badge/python-v3.6.10-green.svg?style=plastic)

![Teaser image](./docs/Header.png)
##### A densely connected neural network for stereo estimation
Dominik Hirner, Friedrich Fraundorfer

Implementation of our lightweight fully-convolutional densely connected neural network method for disparity estimation.
This method has been submitted to the **ICPR 2020** conference and is currently being peer-reviewed.

Upon acceptance this repository will contain:

* Python3 and pytorch 1.2.0 implementation of our FC-DCNN network
* trained weights for:
  * Kitti
  * Middlebury
  * ETH3D 
* python3 script for the full post-processing pipeline as described in our paper

The network was trained and tested on a GeForce RTX 2080 with 11GB of RAM.
The code is released under the BSD 2-Clause license. Please cite our paper (link) if you use code from this repository in your work.
