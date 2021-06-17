#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate medr

epochs=20
learning_rate=0.0001
batch_size=512
embedding_dimensions=32
num_negatives=4
l2_norm=1e-05
mlp_1=64
mlp_2=32
mlp_3=16
dataset=1m
exp_dir=dpq_1m
model=dpq
seed=0
k=32
d=32
shared_centroids=false

python main.py --batch_size=$batch_size --epochs=$epochs --learning_rate=$learning_rate --seed=$seed --embedding_dimensions=$embedding_dimensions --k=$k --d=$d --shared_centroids=$shared_centroids --num_negatives=$num_negatives --mlp_1=$mlp_1 --mlp_2=$mlp_2 --mlp_3=$mlp_3 --dataset=$dataset --exp_dir=$exp_dir --model=$model