#!/bin/bash
cd ../

##################### eval for Ctw1500 with ResNet18 4s ###################################
#CUDA_LAUNCH_BLOCKING=1 python3 eval_textBPN.py --net resnet18 --scale 4 --exp_name Ctw1500 --checkepoch 105 --test_size 640 1024 --dis_threshold 0.35 --cls_threshold 0.85 --gpu 0;

##################### eval for Ctw1500 with ResNet50 1s ###################################
#CUDA_LAUNCH_BLOCKING=1 python3 eval_textBPN.py --net resnet50 --scale 1 --exp_name Ctw1500 --checkepoch 155 --test_size 640 1024 --dis_threshold 0.375 --cls_threshold 0.8 --gpu 0;


##################### eval for Ctw1500 with ResNet50-DCN 1s ###################################
# CUDA_LAUNCH_BLOCKING=1 python3 eval_textBPN.py --net deformable_resnet50 --scale 1 --exp_name Ctw1500 --checkepoch 565 --test_size 640 1024 --dis_threshold 0.3 --cls_threshold 0.925 --gpu 0;


##################### test speed for Ctw1500 ###################################
#CUDA_LAUNCH_BLOCKING=1 python3 eval_textBPN_speed.py --net resnet18 --scale 4 --exp_name Ctw1500 --checkepoch 570 --test_size 640 1024 --dis_threshold 0.25 --cls_threshold 0.85 --gpu 0;


##################### batch eval for Ctw1500 ###################################
echo "start" > ctw1500_eval_mid.log
for ((i=200; i<=300; i=i+1))
do 
CUDA_VISIBLE_DEVICES=4 python3 eval_mixNet.py --net mixTriHRnet_hor --scale 1 --exp_name Ctw1500_mid --checkepoch $i --test_size 640 1024 --dis_threshold 0.3 --cls_threshold 0.85 --mid True
echo "epoch_"$i >> ctw1500_eval_mid.log
python2 TIoU-metric/curved-tiou/ccw-sortdet.py > /dev/null
cd TIoU-metric/curved-tiou/output/
zip result_ctw1500.zip *.txt > /dev/null
mv result_ctw1500.zip ../
cd ../
python2 script.py -g=ctw1500-gt.zip -s=result_ctw1500.zip >> ../../ctw1500_eval_mid.log
cd ../../
done