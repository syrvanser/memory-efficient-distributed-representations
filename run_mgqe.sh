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
exp_dir=mgqe_100k
model=mgqe
seed=0
k=32
d=32

python main.py --batch_size=$batch_size --epochs=$epochs --learning_rate=$learning_rate --seed=$seed --embedding_dimensions=$embedding_dimensions --k=$k --d=$d --shared_centroids=$shared_centroids --num_negatives=$num_negatives --mlp_1=$mlp_1 --mlp_2=$mlp_2 --dataset=$dataset --exp_dir=$exp_dir --model=$model