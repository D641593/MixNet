#!/bin/bash
cd ../
python3 train_textBPN.py --exp_name Totaltext --net mixTriHRnet --max_epoch 660 --batch_size 8 --gpu 4 --input_size 640 --optim Adam --lr 0.001 --num_workers 4
python3 train_textBPN.py --exp_name Totaltext --net mixTriHRnet --scale 2 --max_epoch 660 --batch_size 8 --gpu 3 --input_size 640 --optim Adam --lr 0.001 --num_workers 4
python3 train_textBPN.py --exp_name Ctw1500 --net mixTriHRnet --max_epoch 660 --batch_size 8 --gpu 2 --input_size 640 --optim Adam --lr 0.001 --num_workers 4
CUDA_LAUNCH_BLOCKING=1  python3 train_textBPN.py --exp_name TD500 --net mixTriHRnet --max_epoch 1200 --batch_size 8 --gpu 3 --input_size 640 --optim Adam --lr 0.001 --num_workers 4 --load_memory True
CUDA_LAUNCH_BLOCKING=1  python3 train_textBPN.py --exp_name TD500HUST --net mixTriHRnet --max_epoch 1200 --batch_size 8 --gpu 3 --input_size 640 --optim Adam --lr 0.0001 --num_workers 4 --load_memory True
CUDA_LAUNCH_BLOCKING=1  python3 train_textBPN.py --exp_name Totaltext --net Swin_S --max_epoch 660 --batch_size 8 --gpu 3 --input_size 640 --optim Adam --lr 0.001 --num_workers 4 --load_memory True
CUDA_LAUNCH_BLOCKING=1  python3 train_textBPN.py --exp_name TD500 --net Swin_S --max_epoch 1200 --batch_size 8 --gpu 3 --input_size 640 --optim Adam --lr 0.0001 --num_workers 4 --load_memory True


# train for position encoding + encoder
CUDA_LAUNCH_BLOCKING=1 python3 train_textBPN.py --exp_name Totaltext --net resnet50 --scale 1 --max_epoch 660 --batch_size 12 --gpu 4 --input_size 640 --optim Adam --lr 0.0001 --num_workers 4

# pretrain
CUDA_LAUNCH_BLOCKING=1 python3 train_textBPN.py --exp_name Synthtext --net lightmixTriHRnet --scale 1 --max_epoch 3 --batch_size 12 --lr 0.001 --gpu 2 --input_size 640 --save_freq 1 --num_workers 4 


CUDA_VISIBLE_DEVICES=2 python3 train_textBPN.py --exp_name Totaltext --net mixTriHRnet_cbam --max_epoch 600 --batch_size 8 --input_size 640 --optim Adam --lr 0.001 --num_workers 4 --resume ./model/preSynthMLT/TextBPN_mixTriHRnet_cbam_5.pth
# ArT midline finetune
CUDA_VISIBLE_DEVICES=3 python3 train_textBPN.py --exp_name ArT_mid --net mixTriHRnet_cbam --batch_size 8 --input_size 640 --optim Adam --lr 0.0001 --num_workers 4 --resume model/ArT/TextBPN_mixTriHRnet_cbam_600.pth --mid True --start_epoch 0 --max_epoch 30 --save_freq 1

CUDA_VISIBLE_DEVICES=2 python3 train_textBPN.py --exp_name preSynthMLT --net FSNetinter_S --max_epoch 3 --save_freq 1 --batch_size 8 --input_size 640 -im Adam --lr 0.001 --num_workers 4 --display_freq 1000