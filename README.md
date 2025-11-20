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


## Overview

## News
* **`Nov 17, 2025`:**   We released DGNet codes, the well-trained weights under different configs for SOD/COD/USOD tasks with various resolutions. We also provide the corresponding datasets, the pretrained backbone weights and their prediction maps from both our models and other SOTA models. They can all be obtained in my [Google Drive Folder]()

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

For convenience, we provide **pre-configured versions** with consistent formatting in our [Google Drive Folder](https://drive.google.com/drive/folders/1CNz9q5MHSsYV9jgqjK-tPd2ZxZIMjUfU?usp=drive_link), which also includes datasets for **COD** and **USOD** tasks.

##### Train

Download [Pretrained Backbones](https://drive.google.com/drive/folders/118CyQETQtxCrIo9aejBpBdIT_L06tiSy?usp=drive_link) and save it in ./weights

run ./scripts/train.sh

```bash
sh ./scripts/train.sh
```

Example train.sh is demonstrated as followed
```
python train.py \
    --mode train \
    --task SOD \
    --backbone L \
    --input_size 512 \
    --device cuda \
    --batch_size_train 8 \
    --max_epoch_num 100 \
    --model_save_fre 3 \
    --eval_interval 3 \
    --output_dir "./output" \
    --resume_cpt "" \
```

##### Evaluation and Predicted Saliency Map 

Download our pretrained checkpoint or train your own model!

| Task | Backbone | Resolution | Params(M) | FLOPs(G) | FPS | Checkpoint | Saliency Map |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| SOD | Hiera-L | 512×512 | 247.56 | 238.52 | 43 | [checkpoint]() |  [Results]() |
| SOD | Hiera-L | 352×352 | 247.56 | 139.07 | 45 | [checkpoint]() |  [Results]() |
| SOD | Hiera-L | 224×224 | 247.56 | 48.59 | 46 | [checkpoint]() |  [Results]() |
| SOD | Hiera\*-L | 512×512 | 162.32 | 217.11 | 48 | [checkpoint]() |  [Results]() |
| SOD | Hiera\*-L | 352×352 | 162.32 | 126.27 | 50 | [checkpoint]() |  [Results]() |
| SOD | Hiera\*-L | 224×224 | 162.32 | 44.19 | 52 | [checkpoint]() |  [Results]() |
| SOD | Hiera-B | 512×512 | 91.92 | 102.78 | 61 | [checkpoint]() |  [Results]() |
| SOD | Hiera-B | 352×352 | 91.92 | 47.95 | 64 | [checkpoint]() |  [Results]() |
| SOD | Hiera-B | 224×224 | 91.92 | 17.89 | 69 | [checkpoint]() |  [Results]() |
| SOD | Hiera\*-B | 512×512 | 49.23 | 83.47 | 72 | [checkpoint]() |  [Results]() |
| SOD | Hiera\*-B | 352×352 | 49.23 | 39.13 | 77 | [checkpoint]() |  [Results]() |
| SOD | Hiera\*-B | 224×224 | 49.23 | 18.86 | 78 | [checkpoint]() |  [Results]() |
| SOD | PVT-v2-B5 | 384×384 | 92.94 | 47.16 | 36 | [checkpoint]() |  [Results]() |
| SOD | Swin-B | 320×320 | 115.56 | 43.64 | 58 | [checkpoint]() |  [Results]() |
| COD | Hiera-L | 512×512 | 247.56 | 238.52 | 43 | [checkpoint]() |  [Results]() |
| USOD | Hiera-L | 512×512 | 247.56 | 238.52 | 43 | [checkpoint]() |  [Results]() |


run ./scripts/inference.sh


Here you can download saliency maps of SOD/COD/USOD tasks from other awesome models: [BaiduNetDisk](https://pan.baidu.com/s/12qQV7aBJCzPc1pVTDsO_QQ?pwd=fq3u)

## Citation

```

```
