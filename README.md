# effcc_distilled
This is an official implementation of the IEEE EDGE 2023 paper titled 'Improved Knowledge Distillation for Crowd Counting on IoT Devices'.
This source code utilizes coding from timm to build the efficientnet lite2, lite0.5, and lite0.25 models for the student model.
The teacher model is adapted from the implementation of 'Crowd Counting Using Deep Learning in Edge Devices'.
The Bayesian loss we used is a modified version obtained from https://github.com/ZhihengCV/Bayesian-Crowd-Counting, which implements the 'Bayesian Loss for Crowd Count Estimation with Point Supervision'.
The dataset we used can be downloaded from http://www.crowd-counting.com/.
This source code has been tested with pytorch 1.4.0, Python 3.7.4, and torchvision 0.5.0. Additionally, we used the 'fosscuda/2019b' module in the our HPC system. Please ensure that you install the correct CUDA version for your own environment.
You may want to correct the path of dataset and model in the python scripts, sorry for the mess of the coding.
