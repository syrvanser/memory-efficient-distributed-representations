#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate medr

epochs=50
learning_rate=0.001
batch_size=512
embedding_dimensions=32
num_negatives=4
l2_norm=0.00002
mlp_1=64
mlp_2=32
dataset=1m
exp_dir=dpq_1m
model=dpq
seed=0
k=32
d=16
shared_centroids=true

python main.py --batch_size=$batch_size --epochs=$epochs --learning_rate=$learning_rate --seed=$seed --embedding_dimensions=$embedding_dimensions --k=$k --d=$d --shared_centroids=$shared_centroids --num_negatives=$num_negatives --mlp_1=$mlp_1 --mlp_2=$mlp_2 --dataset=$dataset --exp_dir=$exp_dir --model=$model