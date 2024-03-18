#!/usr/bin/env bash
export NCCL_LL_THRESHOLD=0
dataset_path='dataset_path'

PORT=${PORT:-888z} GPUS=1 batch_size=128 epochs=30
dataset=IMNET

MODEL=ParFormer_B1  OUTPUT_DIR=./checkpoints/finetune/$MODEL
python -m torch.distributed.launch --nproc_per_node=$GPUS main.py \
--model $MODEL --drop_path 0.1 \
--data_set $dataset --data_path=$dataset_path --print_freq=10 \
--batch_size $batch_size --epochs=$epochs --num_workers=10 \
--opt lamb --lr 1e-4 --warmup_epochs=0 --update_freq 4 \
--head_init_scale 0.001 --cutmix 0 --mixup 0 \
--enable_wandb true --project=ParFormer_Finetune \
--finetune ${checkpoint/Model.pth}\
--input_size 384 \
--output_dir $OUTPUT_DIR








    




