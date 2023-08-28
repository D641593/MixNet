#!/bin/bash
cd ../

##################### batch eval for TD500 ###################################
echo "start" > td500_eval_mid.log
for ((i=155; i<=300; i=i+5))
do 
CUDA_VISIBLE_DEVICES=0 python3 eval_mixNet.py --net FSNet_M --scale 1 --exp_name TD500HUST_mid --checkepoch $i --test_size 640 1024 --dis_threshold 0.3 --cls_threshold 0.85 --mid True
echo "epoch_"$i >> td500_eval_mid.log
bash dataset/msra/eval.sh output/TD500HUST_mid >> td500_eval_mid.log
done


 
