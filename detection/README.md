# Applying ParFormer to Object Detection

Our detection code is developed on top of [MMDetection v2.13.0](https://github.com/open-mmlab/mmdetection/tree/v2.13.0) and [PVT Series](https://github.com/whai362/PVT/tree/v2/detection).

For details see [ParFormer: Vision Transformer Baseline with Parallel Local Global Token Mixer and Convolution Attention Patch Embedding]([https://arxiv.org/pdf/2102.12122.pdf](https://arxiv.org/abs/2403.15004)). 

If you use this code for a paper please cite:

ParFormer
```
@article{setyawan2024parformer,
  title={ParFormer: Vision Transformer Baseline with Parallel Local Global Token Mixer and Convolution Attention Patch Embedding},
  author={Setyawan, Novendra and Kurniawan, Ghufron Wahyu and Sun, Chi-Chia and Hsieh, Jun-Wei and Su, Hui-Kai and Kuo, Wen-Kai},
  journal={arXiv preprint arXiv:2403.15004},
  year={2024}
}
```

## Usage

Install [MMDetection v2.13.0](https://github.com/open-mmlab/mmdetection/tree/v2.13.0).

or

```
pip install mmdet==2.13.0 --user
```

Apex (optional):
```
git clone https://github.com/NVIDIA/apex
cd apex
python setup.py install --cpp_ext --cuda_ext --user
```

If you would like to disable apex, modify the type of runner as `EpochBasedRunner` and comment out the following code block in the configuration files:
```
fp16 = None
optimizer_config = dict(
    type="DistOptimizerHook",
    update_interval=1,
    grad_clip=None,
    coalesce=True,
    bucket_size_mb=-1,
    use_fp16=True,
)
```

## Data preparation

Prepare COCO according to the guidelines in [MMDetection v2.13.0](https://github.com/open-mmlab/mmdetection/tree/v2.13.0).


## Results and models


## Evaluation
To evaluate ParFormer-B1 + Mask RCNN on COCO val2017 on a single node with 2 gpus run:
```
dist_test.sh configs/mask_rcnn_parformer_b1_fpn_1x_coco.py /path/to/checkpoint_file 2 --out results.pkl --eval bbox
```

## Training
To train ParFormer-B1 + Mask RCNN on COCO train2017 on a single node with 2 gpus for 12 epochs run:

```
dist_train.sh configs/mask_rcnn_parformer_b1_fpn_1x_coco.py 2
```

## Demo
```
python demo.py demo.jpg /path/to/config_file /path/to/checkpoint_file
```


## Calculating FLOPS & Params

```
python get_flops.py configs/gfl_pvt_v2_b2_fpn_3x_mstrain_fp16.py
```
This should give
```
Input shape: (3, 1280, 800)
Flops: 260.65 GFLOPs
Params: 33.11 M
```

# License
This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.
