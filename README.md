# Brain Tumor Segmentation ECE542-Project

This project is in progress

## Goals
Create a model to accurately segment a brain tumor from an MRI.
- Experiment with varying Computer Vision Architectures, namely UNet (3D and 2D)
- Create a generalizable system to takes in varying image resolution inputs, while retaining prediction quality.

## Outline
This project includes an customizable implementation of UNet using pytorch as well as scripts/notebooks for training and analysis.
Project
    - experiments
        - in progress
    - scripts
        - in progress
    - unet
        - layer.py: block level implementation
        - model.py: architecture level implementation
    - utils
        - data_utils.py: Dataset classes and other data manipulation functions

## Dependencies
- pytorch
- numpy
- matplotlib
- nibabel

## Dataset Citations
Many thanks to the University of Pennsylvania Perelman School of Medicine for making this data public.

[1] B. H. Menze, A. Jakab, S. Bauer, J. Kalpathy-Cramer, K. Farahani, J. Kirby, et al. "The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS)", IEEE Transactions on Medical Imaging 34(10), 1993-2024 (2015) DOI: 10.1109/TMI.2014.2377694 (opens in a new window)

[2] S. Bakas, H. Akbari, A. Sotiras, M. Bilello, M. Rozycki, J.S. Kirby, et al., "Advancing The Cancer Genome Atlas glioma MRI collections with expert segmentation labels and radiomic features", Nature Scientific Data, 4:170117 (2017) DOI: 10.1038/sdata.2017.117(opens in a new window)

[3] S. Bakas, M. Reyes, A. Jakab, S. Bauer, M. Rempfler, A. Crimi, et al., "Identifying the Best Machine Learning Algorithms for Brain Tumor Segmentation, Progression Assessment, and Overall Survival Prediction in the BRATS Challenge", arXiv preprint arXiv:1811.02629 (2018)

[4] S. Bakas, H. Akbari, A. Sotiras, M. Bilello, M. Rozycki, J. Kirby, et al., "Segmentation Labels and Radiomic Features for the Pre-operative Scans of the TCGA-GBM collection", The Cancer Imaging Archive, 2017. DOI: 10.7937/K9/TCIA.2017.KLXWJJ1Q(opens in a new window)

[5] S. Bakas, H. Akbari, A. Sotiras, M. Bilello, M. Rozycki, J. Kirby, et al.,(opens in a new window) "Segmentation Labels and Radiomic Features for the Pre-operative Scans of the TCGA-LGG collection", The Cancer Imaging Archive, 2017. DOI: 10.7937/K9/TCIA.2017.GJQ7R0EF

This Project was completed as part of coursework at NCSU 
