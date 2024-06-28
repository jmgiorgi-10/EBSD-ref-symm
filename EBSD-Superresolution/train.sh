#!/bin/sh

export CUDA_VISIBLE_DEVICES=0
# export CUDA_LAUNCH_BLOCKING=1

# see argument below: decreased learning rate by an order of magnitude
# reinsert '--prog_patch'

# Consider user larging learning rate, of 0.0002, for Titanium (larger than for Nickel)

python main.py --model 'edsr' --lr 0.0002 --syms_req --syms 'hcp' --val_freq 10 --root_dir '/media/hdd3/jmgiorgi/EBSD-ref-symm' --save 'Ti_edsr_minimum_angle_transformation' --GPU_ID 0 --n_GPUs 1  --dist_type 'minimum_angle_transformation' --patch_size 64 --batch_size 4 --input_dir '/media/hdd3/jmgiorgi/Titanium_all_data' --hr_data_dir 'Train/HR_Images/preprocessed_imgs_all_blocks' --val_lr_data_dir  'Val/LR_Images/X4/preprocessed_imgs_all_Block'  --val_hr_data_dir 'Val/HR_Images/preprocessed_imgs_all_Block' 
