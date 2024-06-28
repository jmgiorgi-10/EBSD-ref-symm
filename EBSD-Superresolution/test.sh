#!/bin/sh

python test.py --input_dir '/media/hdd3/jmgiorgi/Titanium_all_data' --model 'edsr'  --save 'Ti_edsr_minimum_angle_transformation' --resume -1  --model_to_load 'model_best' --test_dataset_type 'Test' --test_only  --dist_type 'minimum_angle_transformation' 
