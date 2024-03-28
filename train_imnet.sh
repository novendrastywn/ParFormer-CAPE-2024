export NCCL_LL_THRESHOLD=0
dataset_path='dataset_path'

model=$1
gpus=$2

PORT=${PORT:-8888} batch_size=128 epochs=300
dataset=IMNET

OUTPUT_DIR=./checkpoints/$MODEL
python -m torch.distributed.launch --nproc_per_node=$gpus main.py \
--model $model --drop_path 0.0001 --model_ema true --model_ema_eval true --auto_resume=true \
--data_set $dataset --data_path=$dataset_path --update_freq=4 \
--batch_size $batch_size --epochs=$epochs --num_workers=10 \
--opt adamW --lr 4e-3 --warmup_epochs=20 --input_size 224 \
--enable_wandb false --project=${Project_Name} \
--output_dir $OUTPUT_DIR
