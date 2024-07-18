#!/bin/sh

export CUDA_VISIBLE_DEVICES=3
# export CUDA_LAUNCH_BLOCKING=1

# see argument below: decreased learning rate by an order of magnitude
# reinsert '--prog_patch'

# Consider user larging learning rate, of 0.0002, for Titanium (larger than for Nickel)

####
# weight decay is actually the regularization factor
####

python main.py --weight_decay 1e-5 --model 'han' --epochs 2000 --batch_size 4 --n_feats 32 --lr 1e-3 --syms_req --syms 'fcc' --val_freq 1 --root_dir '/media/hdd3/jmgiorgi/EBSD-ref-symm' --save 'Open_718_MAT_regularized_1e-5' --GPU_ID 0 --n_GPUs 1  --dist_type 'minimum_angle_transformation' --patch_size 64  --input_dir 'media/hdd3/jmgiorgi/fz_reduced/Open_718' --hr_data_dir 'Train/HR_Images' --val_lr_data_dir  'Val/LR_Images/X4/preprocessed_imgs_all_Blocks'  --val_hr_data_dir 'Val/HR_Images/preprocessed_imgs_all_Blocks' 


# python main.py --model 'han' --epochs 2000 --batch_size 1 --n_feats 32 --lr 1e-5 --syms_req --syms 'fcc' --val_freq 1 --root_dir '/media/hdd3/jmgiorgi/EBSD-ref-symm' --save 'Open_718_han_rot_dist_approx_MAT_symmetry' --GPU_ID 0 --n_GPUs 1  --dist_type 'rot_dist_approx_MAT_symmetry' --patch_size 64  --input_dir '/media/hdd3/jmgiorgi/Titanium_all_data' --hr_data_dir 'Train/HR_Images/preprocessed_imgs_all_blocks' --val_lr_data_dir  'Val/LR_Images/X4/preprocessed_imgs_all_Block'  --val_hr_data_dir 'Val/HR_Images/preprocessed_imgs_all_Block' 

