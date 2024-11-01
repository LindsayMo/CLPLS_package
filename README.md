# CLPLS_package

## Overview
Deciphering the cellular abundance in spatial transcriptomics is crucial for revealing spatial architecture of cellular heterogeneity within tissues. However, some of the current spatial sequencing technologies are in low resolutions, leading to, each spot has multiple heterogeneous cells. Additionally, current spatial deconvolution methods lack the ability to utilize multi-modality information such as gene expression and chromatin accessibility from single-cell multi-omics data. We introduce CLPLS, an efficient spatial deconvolution method that combines graph contrastive learning and partial least squares regression to estimate the cell-type composition at each spatial location. Moreover, CLPLS is a flexible method that it can be extended to integrating spatial transcriptomics data with single-cell multi-omics data, enabling the exploration of the spatial landscape of epigenomics previously not visible by existing methods.

## Requirements

* anndata==0.10.5
* networkx==2.8.4
* numpy==1.23.5
* pandas==1.5.3
* scanpy==1.9.8
* scikit_learn==1.2.1
* scipy==1.10.0
* torch==2.3.0
* tqdm==4.64.1

## Data

The preprocessed data used in our experiments have been uploaded and are freely available at https://drive.google.com/drive/folders/12Ma5hciEt454Cohkh2EqUUm49xk9KTpi?usp=sharing.

## Tutorial
For the step-by-step tutorials, we have released them at https://github.com/LindsayMo/CLPLS_package/tree/main/experiment.




## Citation
