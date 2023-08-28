#!/bin/bash
cd ../

##################### batch eval for Ctw1500 ###################################
echo "start" >> totaltext_mid_eval.log
for ((i=155; i<=600; i=i+5))
do 
CUDA_VISIBLE_DEVICES=1 python3 eval_mixNet.py --net FSNet_M --scale 1 --exp_name Totaltext_mid --checkepoch $i --test_size 640 1024 --dis_threshold 0.3 --cls_threshold 0.85 --mid True
echo "epoch_"$i >> totaltext_mid_eval.log
python3 dataset/total_text/Evaluation_Protocol/Python_scripts/Deteval.py Totaltext_mid --tr 0.7 --tp 0.6 >> totaltext_mid_eval.log
done