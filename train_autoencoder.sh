#!/bin/bash

gpus=1
node_num=1
single_gpus=`expr $gpus / $node_num`

cpus=12

while true
do 
  PORT=$((((RANDOM<<15)|RANDOM)%49152 + 10000))
  break
done
echo $PORT

srun -p videood --kill-on-bad-exit=1 --ntasks-per-node=$single_gpus --time=43200 --cpus-per-task=$cpus -N $node_num -o train_job/%j.out python -u train.py \
--init_method 'tcp://127.0.0.1:'$PORT \
-c ./configs/autoencoder_kl_gan.yaml \
--world_size $gpus \
--per_cpus $cpus \
--tensor_model_parallel_size 1 \
--outdir '/home/exouser/YuZheng/experiments' \
--desc  'hmdb_autoencoder'

sleep 2
rm -f batchscript-*


