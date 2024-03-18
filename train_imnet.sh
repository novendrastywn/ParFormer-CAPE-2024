export NCCL_LL_THRESHOLD=0
dataset_path='/home/ndr/Documents/ImageDataset/ImageNet-1K'

PORT=${PORT:-8888} GPUS=1 batch_size=128 epochs=300
dataset=IMNET

MODEL=convmext_alto OUTPUT_DIR=./checkpoints/$MODEL
python -m torch.distributed.launch --nproc_per_node=$GPUS main.py \
--model $MODEL --drop_path 0.0001 --model_ema true --model_ema_eval true --auto_resume=true \
--data_set $dataset --data_path=$dataset_path --update_freq=4 \
--batch_size $batch_size --epochs=$epochs --num_workers=10 \
--opt adamW --lr 4e-3 --warmup_epochs=20 --input_size 224 \
--enable_wandb false --project=ParFormer \
--output_dir $OUTPUT_DIR
