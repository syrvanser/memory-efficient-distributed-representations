#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate medr

epochs=50
learning_rate=0.001
batch_size=512
embedding_dimensions=8
num_negatives=4
l2_norm=0.00002
mlp_1=32
mlp_2=16
mlp_3=8
dataset=100k
exp_dir=dpq
model=dpq
seed=0
k=32
d=8
shared_centroids=true

python main.py --batch_size=$batch_size --epochs=$epochs --learning_rate=$learning_rate --seed=$seed --embedding_dimensions=$embedding_dimensions --l2_norm=$l2_norm --num_negatives=$num_negatives --mlp_1=$mlp_1 --mlp_2=$mlp_2 --mlp_3=$mlp_3 --dataset=$dataset --exp_dir=$exp_dir --model=$model