# DD-IRSENet: Dual-domain Iterative residual sqeze-excitaton network for sparse-view reconstruction of MRI

Xiongchao Chen, Zhigang Peng

![image](IMAGE/IRSENet.png)

This repository contains the PyTorch implementation of Dual-domain Iterative residual sqeze-excitaton network for sparse-view reconstruction of MRI

### Citation
If you use this code for your research or project, please cite: \
\
"..."

 ### Environment and Dependencies
 Requirements:
 * Python 3.8.13
 * PyTorch 1.11.0
 * numpy 1.22.3
 * tqdm 4.64.0
 * opencv 4.4.0
 * scikit-image 0.17.2
 * hypy 3.1.0
 * pyyaml 6.0

Our code has been tested with Python 3.8.13, PyTorch 1.11.0, CUDA: 10.0.130 on Ubuntu 18.04.6 

### Dataset Setup
    .
    Data
    ├── train_480                # contain training files
    |   ├── recon_001.h5
    |       ├── kspace.mat  
    |       ├── kspace_sv.mat
    |       ├── recon_origin.mat
    |       ├── recon_origin_sv.mat
    |       ├── mask.mat
    |       ├── recon_rss.mat
    |       ├── recon_rss_sv.mat
    |   └── ...  
    |
    ├── valid                # contain validation files
    |   ├── recon_001.h5
    |       ├── kspace.mat  
    |       ├── kspace_sv.mat
    |       ├── recon_origin.mat
    |       ├── recon_origin_sv.mat
    |       ├── mask.mat
    |       ├── recon_rss.mat
    |       ├── recon_rss_sv.mat
    |   └── ... 
    |
    └── test                 # contain testing files
    |   ├── recon_001.h5
    |       ├── kspace.mat  
    |       ├── kspace_sv.mat
    |       ├── recon_origin.mat
    |       ├── recon_origin_sv.mat
    |       ├── mask.mat
    |       ├── recon_rss.mat
    |       ├── recon_rss_sv.mat
    |   └── ... 
    └── ...  
 
 where \
`kspace`: full-view kspace data with a size of Num x W x H. \
`kspace_sv`: downsampled sparse-view kspace data with a size of Num x W x H. \
`recon_origin`: iverse fourier transfrom of 'kspace'
`recon_origin_sv`: iverse fourier transfrom of 'kspace_sv'




 
 
