#!/bin/bash
cd ../

##################### batch eval for Ctw1500 ###################################
echo "start" > ctw1500_eval.log
for ((i=155; i<=1200; i=i+5))
do 
CUDA_VISIBLE_DEVICES=4 python3 eval_mixNet.py --net FSNet_hor --scale 1 --exp_name Ctw1500 --checkepoch $i --test_size 640 1024 --dis_threshold 0.3 --cls_threshold 0.85
echo "epoch_"$i >> ctw1500_eval.log
python3 dataset/ctw1500/Evaluation_Protocol/ctw1500_eval.py Ctw1500 >> ctw1500_eval.log
# python2 TIoU-metric/curved-tiou/ccw-sortdet.py > /dev/null
# cd TIoU-metric/curved-tiou/output/
# zip result_ctw1500.zip *.txt > /dev/null
# mv result_ctw1500.zip ../
# cd ../
# python2 script.py -g=ctw1500-gt.zip -s=result_ctw1500.zip >> ../../ctw1500_eval.log
# cd ../../
done