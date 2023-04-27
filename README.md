# Video Frame Interpolation with Densely Queried Bilateral Correlation

## Introduction

This repository is the official implementation of the IJCAI 2023 paper "Video Frame Interpolation with Densely Queried Bilateral Correlation". [[paper](https://arxiv.org/abs/2304.13596)]

## Requirements

1. `torch` is necessary. The code has been developed with `torch1.12.1`.
2. Install other requirements as:
    ```bash
    pip install -r requirements.txt
    ```


## Benchmarking

Download our [pretrained model](https://drive.google.com/file/d/1_cCLh5Tz6aRjde6-siesXOP3iBju3Fa4/view?usp=sharing).

Download [Vimeo90K dataset](http://toflow.csail.mit.edu/).

Download [SNU_FILM dataset](https://myungsub.github.io/CAIN/).

Download [UCF101 dataset](https://liuziwei7.github.io/projects/VoxelFlow).

Download [MiddleBury Other dataset](https://vision.middlebury.edu/flow/data/).

Make your downloaded files structured like below:

```bash
.
├── configs
├── datas
├── datasets
│   ├── middlebury
│   │   ├── other-data
│   │   └── other-gt-interp
│   ├── snu_film
│   │   ├── test
│   │   ├── test-easy.txt
│   │   ├── test-extreme.txt
│   │   ├── test-hard.txt
│   │   └── test-medium.txt
│   ├── ucf101
│   │   ├── 1
│   │   ├── 1001
│   │   ...
│   │   ├── 981
│   │   └── 991
│   └── vimeo_triplet
│       ├── readme.txt
│       ├── sequences
│       ├── tri_testlist.txt
│       └── tri_trainlist.txt
├── pretrained
│   └── 510000.pth
├── experiments
├── losses
├── models
├── utils
├── validate
├── train.py
├── test.py
└── val.py
```

Run benchmarking by following commands:
```bash
python val.py --config configs/benchmarking/vimeo.yaml --gpu_id 0
python val.py --config configs/benchmarking/middlebury.yaml --gpu_id 0
python val.py --config configs/benchmarking/ucf101.yaml --gpu_id 0
python val.py --config configs/benchmarking/snu_film.yaml --gpu_id 0
```

To enable the augmented test (**"Ours-Aug"** in the paper), uncomment the `val_aug: [T,R]` line in the configuration files.


## Training

The model was trained on the Vimeo90K-triplet training split.

Run the following command for training:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port 9999 train.py --config configs/train.yaml
```

## Testing on a Custom Image Pair

First specify the path of the model weights in `configs/test.yaml`.

Then you can test the model on a customized image pair as:

```bash
python test.py --config configs/test.yaml --im0 <path to im0> --im1 <path to im1> --output_dir <path to output folder>
```
