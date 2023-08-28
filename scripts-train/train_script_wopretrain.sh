#!/bin/bash
cd ../
CUDA_VISIBLE_DEVICES=2 python3 train_textBPN.py --exp_name Totaltext --net FSNeXt_S --batch_size 8 --input_size 640 --max_epoch 600 --optim Adam --lr 0.001 --num_workers 4
CUDA_VISIBLE_DEVICES=2 python3 train_textBPN.py --exp_name Totaltext --net FSNeXt_M --batch_size 8 --input_size 640 --max_epoch 600 --optim Adam --lr 0.001 --num_workers 4
CUDA_VISIBLE_DEVICES=2 python3 train_textBPN.py --exp_name Totaltext --net FSNet_S --batch_size 8 --input_size 640 --max_epoch 600 --optim Adam --lr 0.001 --num_workers 4
CUDA_VISIBLE_DEVICES=2 python3 train_textBPN.py --exp_name Totaltext --net FSNet_M --batch_size 8 --input_size 640 --max_epoch 600 --optim Adam --lr 0.001 --num_workers 4
CUDA_VISIBLE_DEVICES=2 python3 train_textBPN.py --exp_name Totaltext --net FSNet_T --batch_size 8 --input_size 640 --max_epoch 600 --optim Adam --lr 0.001 --num_workers 4
CUDA_VISIBLE_DEVICES=2 python3 train_textBPN.py --exp_name Totaltext --net resneXt50 --batch_size 8 --input_size 640 --max_epoch 600 --optim Adam --lr 0.001 --num_workers 4