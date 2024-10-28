import os
import copy
import shutil

import itertools
import argparse
from collections import OrderedDict


parser = argparse.ArgumentParser()

parser.add_argument("--logdir", type=str, 
    default="", help="Cluster name.")
args = parser.parse_args()

logdir = args.logdir

dir_root = "data_src/kitti360"
dir_data_2d_nvs_drop = os.path.join(dir_root, "data_2d_nvs_drop50")

variants_root = "" 
scans = ['test_00', 'test_01', 'test_02', 'test_03', 'test_04']
test_iters = [200000] * len(scans)
# test_iters = [50000, 60000, 70000, 80000, 70000] 

save_img_dir = os.path.join(logdir, 'test_submit')
if not os.path.exists(save_img_dir):
    os.makedirs(save_img_dir)

for scan, test_iter in zip(scans, test_iters):
    fn_frame_list = os.path.join(dir_data_2d_nvs_drop, f'{scan}.txt')
    with open(fn_frame_list) as f:
        frame_00_list = f.read().splitlines()
    image_paths = frame_00_list         
    id_list = list(range(len(image_paths)))

    for i in id_list: 
        # source
        img_dir_scan = f"test_benchmark_{test_iter}/images"
        fn = 'step-{:04d}-{}.png'.format(i, "coarse_raycolor")
        name = "train"
        source_img_fn = os.path.join(logdir, "train_"+ scan.split("_")[1], img_dir_scan, fn)
        image_path = image_paths[i]
        target_img_fn = os.path.basename(image_path)
        seq_name = image_path.split('/')[0].split('_')[-2]  
        target_img_fn = seq_name+"_"+target_img_fn
        target_img_fn = os.path.join(save_img_dir, target_img_fn) 
        assert os.path.exists(source_img_fn) 
        shutil.copyfile(source_img_fn, target_img_fn)
        # target_img_fn =
    

def extract():
    pass