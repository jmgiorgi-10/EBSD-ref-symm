#!/bin/bash

modelname="han_rot_dist_approx"
filetype="SR"
datasettype=("Test")
material="Open_718"
sect=("x_block")

# path of Dream3D software
dream3d_path="/media/hdd3/jmgiorgi/DREAM3D-6.5.171-Linux-x86_64/bin"

#path of json file for Dream3D pipeline
json_path="/home/jmgiorgi/EBSD-ref-symm/EBSD-Superresolution/IPF_mapping/pipeline.json"

# path of IPF mapping code
home_path="/home/jmgiorgi/EBSD-ref-symm/EBSD-Superresolution/IPF_mapping"

# path where numpy files (output of models) are saved
file_path="/media/hdd3/jmgiorgi/EBSD-ref-symm/experiments/saved_weights/han_rot_dist_approx/results"
# file_path="/media/hdd3/jmgiorgi/fz_reduced/Open_718/Test/HR_Images"

#path of your source dream3d file as refernece
sourcename="/media/hdd3/jmgiorgi/FCC_Val_02-UPsa.dream3d"

for s in ${sect[@]}; do
   
    echo "Running Numpy to Dream3D"
    python npy_to_dream3d.py --fpath $file_path --data $material --model_name $modelname --file_type $filetype --dataset_type $datasettype --section $s --d3_source $sourcename
    
    echo "Changing Variable in JSON "

    python change_var_in_json.py --fpath $file_path  --data $material --model_name $modelname --file_type $filetype --dataset_type $datasettype --section $s
    
    echo "Running Dream 3D Pipeline"
        
    cd $dream3d_path
    
    ./PipelineRunner -p $json_path

    echo "Running Dream 3D Pipeline"
    
    cd $home_path
  
    python dream3d_to_rgb.py --fpath $file_path --data $material --model_name $modelname --file_type $filetype --dataset_type $datasettype --section $s



done


