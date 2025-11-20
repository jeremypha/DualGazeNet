<div align = 'center'>
<h1>DGNet</h1>
<h3>DualGazeNet: A Biologically Inspired Dual-Gaze Query Network for Salient Object Detection
</h3>
<!-- Yu Zhang, Haoan Ping, Yuchen Li, Zhenshan Bing, Wei He, Fellow, IEEE,
Fuchun Sun, Fellow, IEEE, Alois Knoll, Fellow, IEEE -->

Technical University of Munich

<!-- Paper:([arxiv:2408.04326v1](https://arxiv.org/html/2408.04326v1)) -->
</div>


## Abstract
DualGazeNet is a biologically inspired Transformer framework for salient object detection, designed with dual-path processing inspired by the human visual system. It achieves state-of-the-art performance on five RGB SOD benchmarks as well as 4 COD benchmarks and USOD10 dataset. 

## Overview
<p align="center">
  <img src="DGN_Overview.png" alt="arch" width="100%">
</p>


## News
* **`Nov 20, 2025`:**   We released the well-trained weights under different configs for SOD/COD/USOD tasks with various resolutions. We also provide the corresponding datasets, the pretrained backbone weights and their prediction maps from both our models and other SOTA models.

* **`Nov 17, 2025`:**   We released DGNet codes.

<!-- * **`Nov 17, 2025`:**  We released our paper on [arXiv](). -->


<!-- <p align="center">
  <img src="assets/overall.png" alt="arch" width="80%">
</p> -->

## Usage

### Installation

#### Step 1:
Clone this repository

```bash
git clone https://github.com/jeremypha/DualGazeNet.git
cd DualGazeNet
```

#### Step 2:

##### Create a new conda environment

```bash
conda create --name dgnet
conda activate dgnet
```

##### Install Dependencies
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu<your cuda version>
pip install -r requirements.txt
```

##### Set up Datasets
````
-- datasets
    |-- SOD
    |    |-- DUTS-TR
    |    |   |-- im
    |    |   |-- gt
    |    |-- DUTS-TE
    |    |   |-- im
    |    |   |-- gt
    |    |-- DUT-OMRON
    |    |   |-- im
    |    |   |-- gt
    |    |-- ECSSD
    |    |   |-- im
    |    |   |-- gt
    |    |-- HKU-IS
    |    |   |-- im
    |    |   |-- gt
    |    |-- PASCAL-S
    |    |   |-- im
    |    |   |-- gt
    |-- COD
    |-- USOD

````
All datasets are publicly available from their official sources: [DUTS](http://saliencydetection.net/duts/), [DUT-OMRON](http://saliencydetection.net/dut-omron/), [HKU-IS](https://i.cs.hku.hk/~yzyu/research/deep_saliency.html), [ECSSD](https://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/dataset.html), and [PASCAL-S](https://cbs.ic.gatech.edu/salobj/).

For convenience, we provide **pre-configured versions** with consistent formatting in our [BaiduNetDisk Folder](https://pan.baidu.com/s/11-eVRNce7thlrZKXtyzgug?pwd=84bw), which also includes datasets for **COD** and **USOD** tasks.

##### Train

Download [Pretrained Backbones](https://pan.baidu.com/s/1Ovd7A_fVi7uxjeY2Z3IzjA?pwd=f116) and save it in ./weights

run ./scripts/train.sh

##### Evaluation and Predicted Saliency Map 

Model weights and corresponding prediction maps for all configurations are available for download. Access the full dataset in our [Google Drive Folder](https://drive.google.com/drive/folders/1t9_lETBB1H-1uc-mIukHFlTe7dzbAaRU?usp=drive_link), or retrieve specific items individually from the following table.

| Task | Backbone | Resolution | Params(M) | FLOPs(G) | FPS | Checkpoint | Saliency Map |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| SOD | Hiera-L | 512×512 | 247.56 | 238.52 | 43 | [checkpoint](https://drive.google.com/file/d/1yrZqmhb0wNuvmOVb8GfBVUkRjBK-F6hi/view?usp=drive_link) |  [Results](https://drive.google.com/file/d/1ndHItAE1U9xEPMQPXHCuXn7VSCUlpiPQ/view?usp=drive_link) |
| SOD | Hiera-L | 352×352 | 247.56 | 139.07 | 45 | [checkpoint](https://drive.google.com/file/d/17GSLdBkXw2MaX-9EOVU2UisCldMFuECl/view?usp=drive_link) |  [Results](https://drive.google.com/file/d/1RvOBBJ0ypnnTBFzVi1MgEbM-vN8TPZ5n/view?usp=drive_link) |
| SOD | Hiera-L | 224×224 | 247.56 | 48.59 | 46 | [checkpoint](https://drive.google.com/file/d/1t1UPzjkBYSQfoTv2voI3Dbz-b3XErLrc/view?usp=drive_link) |  [Results](https://drive.google.com/file/d/1HCCBoJgQw5EEl1ABE7M0XAOAJz2BFDDl/view?usp=drive_link) |
| SOD | Hiera\*-L | 512×512 | 162.32 | 217.11 | 48 | [checkpoint](https://drive.google.com/file/d/1WteXk2Qx7Erh0cKnQOiW3yeuCKvlXIfC/view?usp=drive_link) |  [Results](https://drive.google.com/file/d/1ndHItAE1U9xEPMQPXHCuXn7VSCUlpiPQ/view?usp=drive_link) |
| SOD | Hiera\*-L | 352×352 | 162.32 | 126.27 | 50 | [checkpoint](https://drive.google.com/file/d/1NlE2mahsCWrq14OrnY37js2GlJeJ--x-/view?usp=drive_link) |  [Results](https://drive.google.com/file/d/1RvOBBJ0ypnnTBFzVi1MgEbM-vN8TPZ5n/view?usp=drive_link) |
| SOD | Hiera\*-L | 224×224 | 162.32 | 44.19 | 52 | [checkpoint](https://drive.google.com/file/d/1lkKwkq9SH_uZ-lYv3lhkkyo9MOJ11hh1/view?usp=drive_link) |  [Results](https://drive.google.com/file/d/1HCCBoJgQw5EEl1ABE7M0XAOAJz2BFDDl/view?usp=drive_link) |
| SOD | Hiera-B | 512×512 | 91.92 | 102.78 | 61 | [checkpoint](https://drive.google.com/file/d/1ffmAzkHG2CW2woMkd3bvp20QnEdWYETw/view?usp=drive_link) |  [Results](https://drive.google.com/file/d/1zuz9lvB_fX1TXPKg9a9rkYLrikHgEAAX/view?usp=drive_link) |
| SOD | Hiera-B | 352×352 | 91.92 | 47.95 | 64 | [checkpoint](https://drive.google.com/file/d/12nHB1ssEuNGyW-DqUAwouhzITJzrR-ys/view?usp=drive_link) |  [Results](https://drive.google.com/file/d/1LXlydXF_ixFuGhUGhGRqX9CHxw30kZ29/view?usp=drive_link) |
| SOD | Hiera-B | 224×224 | 91.92 | 17.89 | 69 | [checkpoint](https://drive.google.com/file/d/1fhRfFLkt9AlmRBn7Oayz7n3IXsHAhQxn/view?usp=drive_link) |  [Results](https://drive.google.com/file/d/1ZDvLlcZgPI1OktAe0D3yhzvjzzrqKNZK/view?usp=drive_link) |
| SOD | Hiera\*-B | 512×512 | 49.23 | 83.47 | 72 | [checkpoint](https://drive.google.com/file/d/1yE-X8u1PQUHSs9EvBMsALbwyZBaJuM29/view?usp=drive_link) |  [Results](https://drive.google.com/file/d/1jSWIKtamwFobkIb3UQwfazNdLH9pDxXu/view?usp=drive_link) |
| SOD | Hiera\*-B | 352×352 | 49.23 | 39.13 | 77 | [checkpoint](https://drive.google.com/file/d/1NaSONPsHWTKAqxYk7BFgLdIhPYes_YPT/view?usp=drive_link) |  [Results](https://drive.google.com/file/d/15X6E3XKVJZT_8ZIhA2if6AwQnZ6mFsGs/view?usp=drive_link) |
| SOD | Hiera\*-B | 224×224 | 49.23 | 18.86 | 78 | [checkpoint](https://drive.google.com/file/d/1nxVqLB6ikzy56UMQEeYfE3l6BA1jE-a5/view?usp=drive_link) |  [Results](https://drive.google.com/file/d/1u1Bo7EvEk08vy8vHAC_1YWA9eWjYoZYp/view?usp=drive_link) |
| COD | Hiera-L | 512×512 | 247.56 | 238.52 | 43 | [checkpoint](https://drive.google.com/file/d/1MiwAQzPIrgIHXYf_-Z3TrkhqCPS_ERY-/view?usp=drive_link) |  [Results](https://drive.google.com/file/d/1eGowhy68GqZtI8PM8FX_vvPrImEanh4l/view?usp=drive_link) |
| USOD | Hiera-L | 512×512 | 247.56 | 238.52 | 43 | [checkpoint](https://drive.google.com/file/d/1NtHbcfPhRXq6p7TUmA4YwFelEDpVeymT/view?usp=drive_link) |  [Results](https://drive.google.com/file/d/15m8PFNGzORmS8QU0HE8P7wEro_GzfWiB/view?usp=drive_link) |


run ./scripts/inference.sh

## Quantitative Comparison
<p align="center">
  <img src="Quantitative_Comparison_SOD.png" alt="arch" width="100%">
</p>
<p align="center">
  <img src="Quantitative_Comparison_COD.png" alt="arch" width="100%">
</p>
<p align="center">
  <img src="Quantitative_Comparison_USOD.png" alt="arch" width="100%">
</p>

Here you can download saliency maps of SOD/COD/USOD tasks from other awesome models: [BaiduNetDisk](https://pan.baidu.com/s/12qQV7aBJCzPc1pVTDsO_QQ?pwd=fq3u)




## Citation

```

```
