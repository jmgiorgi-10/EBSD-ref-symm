#!/bin/sh

python test.py --input_dir 'media/hdd3/jmgiorgi/fz_reduced/Open_718' --model 'edsr'  --save 'edsr_rot_dist_approx_prog_patch_lr_div_10' --resume -1  --model_to_load 'model_best' --test_dataset_type 'Test' --test_only  --dist_type 'rot_dist_approx' 
