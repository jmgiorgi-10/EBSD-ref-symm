#!/bin/sh

export CUDA_VISIBLE_DEVICES=2
# export CUDA_LAUNCH_BLOCKING=1

python main.py --prog_patch --syms 'fcc' --val_freq 2 --root_dir '/media/hdd3/jmgiorgi/EBSD-ref-symm' --save 'edsr_rot_dist_approx_prog_patch' --GPU_ID 0 --n_GPUs 1 --model 'edsr' --dist_type 'rot_dist_approx' --patch_size 64 --batch_size 4 --input_dir 'media/hdd3/jmgiorgi/fz_reduced/Open_718' --hr_data_dir 'Train/HR_Images' --val_lr_data_dir  'Val/LR_Images/X4/preprocessed_imgs_all_Blocks'  --val_hr_data_dir 'Val/HR_Images/preprocessed_imgs_all_Blocks' 