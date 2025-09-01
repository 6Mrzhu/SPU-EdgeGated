#!/usr/bin/env bash
script_path=$(cd `dirname $0`; pwd)
cd $script_path

in_data_dir=./data/kitti
gt_data_dir=/dataset/wangjingjing9/3DFR/public_data/PU/PU1K/test/input_2048/gt_8192
num_shape_point=39726

Model="./model/demo/model_20.pth"

python main.py --phase test --ckpt ${Model}  --num_shape_point ${num_shape_point} --test_data  ${in_data_dir}

# cd ./evaluation_code
# bash eval_pu1k.sh
# cd $script_path

# python main.py --phase eval --ckpt ${Model} --num_shape_point ${num_shape_point} --test_data  ${in_data_dir} --gt_path  ${gt_data_dir} 



