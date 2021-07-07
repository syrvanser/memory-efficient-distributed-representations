#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate medr

epochs=30
learning_rate=0.00005
batch_size=256
embedding_dimensions=32
num_negatives=7
l2_norm=1e-05
mlp_1=64
mlp_2=32
mlp_3=16
dataset=100k
exp_dir=mf
model=mf
seed=0
continue_from_checkpoint=1
python main.py --batch_size=$batch_size --epochs=$epochs --learning_rate=$learning_rate --seed=$seed --embedding_dimensions=$embedding_dimensions --l2_norm=$l2_norm --num_negatives=$num_negatives --mlp_1=$mlp_1 --mlp_2=$mlp_2 --mlp_3=$mlp_3 --dataset=$dataset --exp_dir=$exp_dir --model=$model --continue_from_checkpoint=$continue_from_checkpoint