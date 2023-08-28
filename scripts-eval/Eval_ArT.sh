#!/bin/bash
cd ../

##################### batch eval for ArT ###################################
for ((i=602; i<=611; i=i+1));
do 
CUDA_VISIBLE_DEVICES=4 python3 eval_mixNet.py --net FSNet_M --scale 1 --exp_name ArT_mid --checkepoch $i --test_size 960 2880 --dis_threshold 0.4 --cls_threshold 0.8 --mid True
done
