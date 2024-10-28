#!/bin/bash

# nrCheckpoint="../checkpoints_pointnerf"
# nrDataRoot="../data_src"
# name='chair'
# resume_iter=best #
# data_root="${nrDataRoot}/nerf/nerf_synthetic_colmap/"
# scan="chair"
suffix="${1:-debug}"
backbone_opt="${2:-none}"
point_emb_opt="${3:-mvs}"
backbone_voxel_size_multiplier="${4:-2}"
point_init_emb_opt="${5:-learned}"
num_point_emb_level="${6:-0}"
scan="${7:-chair}"
prob_freq="${8:-10000}"
pyramid_opt="${9:-grid_sampling}"
vsizei="${10:-0.004}"
agg_opt="${11:-pointnet}"
plane_size="${12:-2}"
mid_conf_mode="${13:-learned_normalized}"
point_conf_mode="${14:-1}"
start_plane_idx="${15:-0}"
split="${16:-train}"
global_nerf="${17:-none}"
resample_pnts="${18:-0}"


which_agg_model="viewmlp"
normalize_ray=0

# echo $(($backbone_voxel_size_multiplier * $vsizei))
f_vsizei=$(echo "$vsizei" | bc -l)
backbone_voxel_size=$(echo "$backbone_voxel_size_multiplier  * $f_vsizei" | bc -l) # float multiplication
echo $backbone_voxel_size
echo $which_agg_model

pts0_init=random
load_points=1

# prob_freq=10001 #10000 #2000 #1000 is bad #10001
points2vox_opt=pointnet_ae
# nrCheckpoint="../logs/logs_cur/${suffix}/${agg_opt}_pyramid_${pyramid_opt}_backbone_${backbone_opt}_conf${mid_conf_mode}_${point_conf_mode}_point_emb_opt_${point_emb_opt}_vs_${backbone_voxel_size}_init_${point_init_emb_opt}_emb_level_${num_point_emb_level}_prob_freq_${prob_freq}_planesize_${plane_size}__splane_${start_plane_idx}"
nrCheckpoint="../logs/logs_cur/${suffix}/rand_npts${resample_pnts}_${global_nerf}_${agg_opt}_pyramid_${pyramid_opt}_backbone_${backbone_opt}_conf_${mid_conf_mode}_${point_conf_mode}_point_emb_opt_${point_emb_opt}_vs_${backbone_voxel_size}_init_${point_init_emb_opt}_emb_level_${num_point_emb_level}_prob_freq_${prob_freq}_planesize_${plane_size}__splane_${start_plane_idx}_${split}"


nrDataRoot="../data_src"
name=$scan
resume_iter=best #
data_root="${nrDataRoot}/nerf/nerf_synthetic_colmap/"

feat_grad=1
conf_grad=1
dir_grad=1
color_grad=1
vox_res=320
normview=0
prune_thresh=-1
prune_iter=-1
prune_max_iter=200000

feedforward=0
ref_vid=0
bgmodel="no" #"plane"
depth_occ=1
depth_vid="0"
trgt_id=0
manual_depth_view=1
init_view_num=3
# pre_d_est="${nrCheckpoint}/MVSNet/model_000014.ckpt"
pre_d_est="../checkpoints/MVSNet/model_000014.ckpt"
manual_std_depth=0.0
depth_conf_thresh=0.8
appr_feature_str0="imgfeat_0_0123 dir_0 point_conf"
point_dir_mode="1" # 0 for only at features, 1 for color branch
point_color_mode="1" # 0 for only at features, 1 for color branch
default_conf=0.15

agg_feat_xyz_mode="None"
agg_alpha_xyz_mode="None"
agg_color_xyz_mode="None"
feature_init_method="rand" #"rand" # "zeros"
agg_axis_weight=" 1. 1. 1."
agg_dist_pers=20
radius_limit_scale=4
depth_limit_scale=0
alpha_range=0

vscale=" 2 2 2 "
kernel_size=" 3 3 3 "
query_size=" 3 3 3 "
# vsize=" 0.004 0.004 0.004 " #" 0.005 0.005 0.005 "
vsize=" ${vsizei} ${vsizei} ${vsizei}" #" 0.005 0.005 0.005 "
wcoord_query=-1
z_depth_dim=400
max_o=410000 #2000000
ranges=" -0.721 -0.695 -0.995 0.658 0.706 1.050 "
SR=160
K=4
P=12 #120
NN=2


act_type="LeakyReLU"

agg_intrp_order=2
agg_distance_kernel="linear" #"avg" #"feat_intrp"
weight_xyz_freq=2
weight_feat_dim=8

point_features_dim=32
shpnt_jitter="uniform" #"uniform" # uniform gaussian

# # Smaller MLP 
# apply_pnt_mask=1
# shading_feature_mlp_layer0=1 #2
# shading_feature_mlp_layer1=1 #2
# shading_feature_mlp_layer2=0 #1
# shading_feature_mlp_layer3=2 #1
# shading_alpha_mlp_layer=1
# shading_color_mlp_layer=2
# shading_feature_num=256
# dist_xyz_freq=5
# num_feat_freqs=0
# dist_xyz_deno=0

# which_agg_model="viewmlp"
apply_pnt_mask=1
shading_feature_mlp_layer0=1 #2
shading_feature_mlp_layer1=2 #2
shading_feature_mlp_layer2=0 #1
shading_feature_mlp_layer3=2 #1
shading_alpha_mlp_layer=1
shading_color_mlp_layer=4
shading_feature_num=128
dist_xyz_freq=5
num_feat_freqs=0
dist_xyz_deno=0

raydist_mode_unit=1
dataset_name='nerf_synth360_ft'
pin_data_in_memory=1
model='mvs_points_volumetric'
near_plane=2.0
far_plane=6.0
which_ray_generation='near_far_linear' #'nerf_near_far_linear' #
domain_size='1'
dir_norm=0

which_tonemap_func="off" #"gamma" #
which_render_func='radiance'
which_blend_func='alpha'
out_channels=4

num_pos_freqs=10
num_viewdir_freqs=4 #6

random_sample='random'
random_sample_size=48 #94 #48 # 32 * 32 = 1024
batch_size=1
plr=0.002
lr=0.0005 # 0.0005 #0.00015
lr_policy="iter_exponential_decay"
lr_decay_iters=1000000
lr_decay_exp=0.1

gpu_ids='0'
checkpoints_dir="${nrCheckpoint}/col_nerfsynth/"
resume_dir="${nrCheckpoint}/init/dtu_dgt_d012_img0123_conf_agg2_32_dirclr20"

save_iter_freq=10000
save_point_freq=10000 #301840 #1
maximum_step=200000 #800000

niter=10000 #1000000
niter_decay=10000 #250000
n_threads=1

train_and_test=0 #1
test_num=10
test_freq=10000 #1200 #1200 #30184 #30184 #50000
print_freq=40
test_num_step=10

far_thresh=-1 #0.005
prob_num_step=50
prob_thresh=0.7
prob_mul=0.4
prob_kernel_size=" 1 1 1"
prob_tiers=" 60000 "

zero_epsilon=1e-3
visual_items='coarse_raycolor gt_image '
sparse_loss_weight=0

color_loss_weights=" 1.0 0.0 0.0 "
color_loss_items='ray_masked_coarse_raycolor ray_miss_coarse_raycolor coarse_raycolor'
test_color_loss_items='coarse_raycolor ray_miss_coarse_raycolor ray_masked_coarse_raycolor'

vid=250000

bg_color="white" #"0.0,0.0,0.0,1.0,1.0,1.0"
split="train"

cd run

if [ "$agg_opt" =  "planes" ]; then
    echo "Using plane based representation"
    echo "plane size $plane_size"
    echo "point_features_dim $point_features_dim"
else
    echo "Using pointnet based representation"
    echo "point_features_dim $point_features_dim"
fi


if [ $prob_freq -eq 0 ]; then
    echo "No PR strategy"
    total_step=1 
    pythonfile_name=train_ft_nonstop.py  
else
    pythonfile_name=train_ft.py  
#     total_step=$((($maximum_step - 1) / $prob_freq)) 
    total_step=$((($maximum_step - 1) / $prob_freq)) 
    echo "PR strategy, $total_step, prob_freq $prob_freq"
fi

for i in $(seq 1 $total_step)
do
#python3 gen_pnts.py \
python3 ${pythonfile_name} \
        --normalize_ray $normalize_ray\
        --plane_size $plane_size \
        --backbone_stride $backbone_voxel_size_multiplier \
        --agg_opt $agg_opt \
        --plane_start_idx $start_plane_idx \
        --mid_conf_mode $mid_conf_mode \
        --pyramid_opt $pyramid_opt \
        --backbone_opt $backbone_opt \
        --point_emb_opt $point_emb_opt \
        --backbone_voxel_size $backbone_voxel_size\
        --num_point_emb_level $num_point_emb_level\
        --point_init_emb_opt $point_init_emb_opt\
        --points2vox_opt $points2vox_opt \
        --experiment $name \
        --scan $scan \
        --data_root $data_root \
        --dataset_name $dataset_name \
        --model $model \
        --which_render_func $which_render_func \
        --which_blend_func $which_blend_func \
        --out_channels $out_channels \
        --num_pos_freqs $num_pos_freqs \
        --num_viewdir_freqs $num_viewdir_freqs \
        --random_sample $random_sample \
        --random_sample_size $random_sample_size \
        --batch_size $batch_size \
        --maximum_step $maximum_step \
        --plr $plr \
        --lr $lr \
        --lr_policy $lr_policy \
        --lr_decay_iters $lr_decay_iters \
        --lr_decay_exp $lr_decay_exp \
        --gpu_ids $gpu_ids \
        --checkpoints_dir $checkpoints_dir \
        --save_iter_freq $save_iter_freq \
        --niter $niter \
        --niter_decay $niter_decay \
        --n_threads $n_threads \
        --pin_data_in_memory $pin_data_in_memory \
        --train_and_test $train_and_test \
        --test_num $test_num \
        --test_freq $test_freq \
        --test_num_step $test_num_step \
        --test_color_loss_items $test_color_loss_items \
        --print_freq $print_freq \
        --bg_color $bg_color \
        --split $split \
        --which_ray_generation $which_ray_generation \
        --near_plane $near_plane \
        --far_plane $far_plane \
        --dir_norm $dir_norm \
        --which_tonemap_func $which_tonemap_func \
        --load_points $load_points \
        --resume_dir $resume_dir \
        --resume_iter $resume_iter \
        --feature_init_method $feature_init_method \
        --agg_axis_weight $agg_axis_weight \
        --agg_distance_kernel $agg_distance_kernel \
        --radius_limit_scale $radius_limit_scale \
        --depth_limit_scale $depth_limit_scale  \
        --vscale $vscale    \
        --kernel_size $kernel_size  \
        --SR $SR  \
        --K $K  \
        --P $P \
        --NN $NN \
        --agg_feat_xyz_mode $agg_feat_xyz_mode \
        --agg_alpha_xyz_mode $agg_alpha_xyz_mode \
        --agg_color_xyz_mode $agg_color_xyz_mode  \
        --save_point_freq $save_point_freq  \
        --raydist_mode_unit $raydist_mode_unit  \
        --agg_dist_pers $agg_dist_pers \
        --agg_intrp_order $agg_intrp_order \
        --shading_feature_mlp_layer0 $shading_feature_mlp_layer0 \
        --shading_feature_mlp_layer1 $shading_feature_mlp_layer1 \
        --shading_feature_mlp_layer2 $shading_feature_mlp_layer2 \
        --shading_feature_mlp_layer3 $shading_feature_mlp_layer3 \
        --shading_feature_num $shading_feature_num \
        --dist_xyz_freq $dist_xyz_freq \
        --shpnt_jitter $shpnt_jitter \
        --shading_alpha_mlp_layer $shading_alpha_mlp_layer \
        --shading_color_mlp_layer $shading_color_mlp_layer \
        --which_agg_model $which_agg_model \
        --color_loss_weights $color_loss_weights \
        --num_feat_freqs $num_feat_freqs \
        --dist_xyz_deno $dist_xyz_deno \
        --apply_pnt_mask $apply_pnt_mask \
        --point_features_dim $point_features_dim \
        --color_loss_items $color_loss_items \
        --feedforward $feedforward \
        --trgt_id $trgt_id \
        --depth_vid $depth_vid \
        --ref_vid $ref_vid \
        --manual_depth_view $manual_depth_view \
        --pre_d_est $pre_d_est \
        --depth_occ $depth_occ \
        --manual_std_depth $manual_std_depth \
        --visual_items $visual_items \
        --appr_feature_str0 $appr_feature_str0 \
        --init_view_num $init_view_num \
        --feat_grad $feat_grad \
        --conf_grad $conf_grad \
        --dir_grad $dir_grad \
        --color_grad $color_grad \
        --depth_conf_thresh $depth_conf_thresh \
        --bgmodel $bgmodel \
        --vox_res $vox_res \
        --act_type $act_type \
        --point_conf_mode $point_conf_mode \
        --point_dir_mode $point_dir_mode \
        --point_color_mode $point_color_mode \
        --normview $normview \
        --prune_thresh $prune_thresh \
        --prune_iter $prune_iter \
        --sparse_loss_weight $sparse_loss_weight \
        --default_conf $default_conf \
        --prob_freq $prob_freq \
        --prob_num_step $prob_num_step \
        --prob_thresh $prob_thresh \
        --prob_mul $prob_mul \
        --prob_kernel_size $prob_kernel_size \
        --prob_tiers $prob_tiers \
        --alpha_range $alpha_range \
        --ranges $ranges \
        --vid $vid \
        --vsize $vsize \
        --wcoord_query $wcoord_query \
        --max_o $max_o \
        --prune_max_iter $prune_max_iter \
        --far_thresh $far_thresh \
        --global_nerf $global_nerf\
        --resample_pnts $resample_pnts\
        --pts0_init $pts0_init\
        --debug
done
#        --zero_one_loss_items $zero_one_loss_items \
#        --zero_one_loss_weights $zero_one_loss_weights \