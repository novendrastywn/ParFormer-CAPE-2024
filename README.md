
# ParFormer: Vision Transformer Baseline with Parallel Local Global Token Mixer and Convolution Attention Patch Embedding
The Implementation of ParFormer: Vision Transformer Baseline with Parallel Local Global Token Mixer and Convolution Attention Patch Embedding 

[arXiv](https://arxiv.org/abs/2403.15004) | [PDF](https://arxiv.org/pdf/2403.15004.pdf)

<img width="1633" alt="ParFormerFramework" src="https://github.com/novendrastywn/ParFormer-CAPE-2024/assets/31612686/73ab3406-81c1-4370-b8be-f634a5ee4705">


## ImageNet  

|  Model Name  | Resolution | Params | GFLOPs | @Top-1 | Download |
|--------------|------------|:------:|:------:|:------:|:--------:|
| ParFormer-B1 |  224X224   |  11M   |  1.5   |  80.5  | [model](https://huggingface.co/novendrastywn/dl/resolve/main/ParFormer/ParFormer_b1_224.pth) |
| ParFormer-B2 |  224X224   |  23M   |  3.4   |  82.1  | [model](https://huggingface.co/novendrastywn/dl/resolve/main/ParFormer/ParFormer_b2_224.pth) |
| ParFormer-B3 |  224X224   |  34M   |  6.5   |  83.1  | [model](https://huggingface.co/novendrastywn/dl/resolve/main/ParFormer/ParFormer_b3_224.pth) |


### Prerequisites
`conda` virtual environment is recommended. 
```
conda install pytorch torchvision cudatoolkit=11.8 -c pytorch
pip install timm==0.6.13
pip install wandb
pip install fvcore
```

### Data preparation

Download and extract ImageNet train and val images from http://image-net.org/. The training and validation data are expected to be in the `train` folder and `val` folder respectively:
```
|-- /path/to/imagenet/
    |-- train
    |-- val
```

### Single machine multi-GPU training

We provide an example training script `train_imnet.sh` using PyTorch distributed data parallel (DDP). 

To train ParFormer-B1 on an 2-GPU machine:

```
sh train_imnet.sh parformer_b1 2
```

Tips: specify your data path and experiment name in the script! 
