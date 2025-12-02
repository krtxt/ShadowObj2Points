#!/bin/bash
# run_training.sh
# export HYDRA_FULL_ERROR=1
# export NCCL_DEBUG=INFO
# export TORCH_DISTRIBUTED_DEBUG=DETAIL 
export TORCH_DISTRIBUTED_DEBUG=OFF
export NCCL_DEBUG=WARN

# 防僵尸进程和卡住
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_DISABLE=1  # 如果网络有问题
export CUDA_LAUNCH_BLOCKING=0  # 设为1调试，0正常运行
export NCCL_TIMEOUT=1800
export TORCH_NCCL_BLOCKING_WAIT=0
# 内存管理
# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True

# NCCL_P2P_DISABLE=1
# NCCL_BLOCKING_WAIT=1

export CUDA_VISIBLE_DEVICES=4,5,6,7

# # 运行训练
# python train.py \
#     model=flow_matching_hand_dit \
#     backbone=ptv3_sparse \
#     datamodule=handencoder_dm_dex \
#     experiments=multi_gpu \
#     optimizer.lr=2e-4 \
#     compile=false \
#     experiment_name=exp_1130_fm_rigid_full \
#     dit.qk_norm=true \
#     dit.qk_norm_type='rms' \
#     batch_size=320 \
#     trainer.precision=bf16-mixed \
#     velocity_strategy=group_rigid \
#     datamodule.num_workers=16 \
#     datamodule.prefetch_factor=2 \
#     datamodule.persistent_workers=false

# python train.py \
#     model=flow_matching_hand_dit \
#     backbone=ptv3_sparse \
#     datamodule=handencoder_dm_dex \
#     experiments=multi_gpu \
#     optimizer.lr=2e-4 \
#     compile=false \
#     experiment_name=exp_1130_fm_direct_free_full \
#     dit.qk_norm=true \
#     dit.qk_norm_type='rms' \
#     batch_size=320 \
#     trainer.precision=bf16-mixed \
#     velocity_strategy=direct_free \
#     datamodule.num_workers=16 \
#     datamodule.prefetch_factor=2 \
#     datamodule.persistent_workers=false 

python train.py \
    model=flow_matching_hand_dit \
    backbone=ptv3_sparse_fourier \
    datamodule=handencoder_dm_dex \
    datamodule.use_scene_normals=true \
    experiments=multi_gpu \
    optimizer.lr=2e-4 \
    compile=true \
    experiment_name=exp_1202_fm_direct_free_full \
    dit.qk_norm=true \
    dit.qk_norm_type='rms' \
    dit.norm_type='rms' \
    dit.activation_fn=swiglu \
    dit.hand_scene_bias.enabled=true \
    batch_size=320 \
    trainer.precision=bf16-mixed \
    velocity_strategy=direct_free \
    model.prediction_target=x \
    datamodule.num_workers=16 \
    datamodule.prefetch_factor=4 \
    datamodule.persistent_workers=false 


# python train.py \
#     model=deterministic_hand_dit \
#     experiment_name=exp_1130_deter_full_stage \
#     loss=deterministic_regression \
#     backbone=ptv3_sparse \
#     datamodule=handencoder_dm_dex \
#     experiments=multi_gpu \
#     optimizer.lr=4e-4 \
#     compile=false \
#     dit.qk_norm=true \
#     dit.qk_norm_type='layer' \
#     batch_size=720 \
#     trainer.precision=bf16-mixed \
#     datamodule.num_workers=16 \
#     datamodule.prefetch_factor=2 \
#     datamodule.persistent_workers=false


# 多卡从指定 checkpoint 恢复训练：
# - 使用 experiments=multi_gpu 开启 DDP 配置
# - 明确指定 trainer.devices=4（对应上面的 CUDA_VISIBLE_DEVICES=4,5,6,7）
# - 直接用 ckpt_path 指向想恢复的 last.ckpt
# - 关闭 auto_resume/auto_resume_config，避免每个 DDP 进程各自去“猜”上一次实验，
#   导致不同 rank 加载到不同的配置/模型，从而出现
#   "DDP expects same model across all ranks" 的错误。

# python train.py \
#   model=flow_matching_hand_dit \
#   datamodule=handencoder_dm_dex \
#   experiments=multi_gpu \
#   experiment_name=exp_1129_fm_direct_free_full \
#   velocity_strategy=direct_free \
#   optimizer.lr=4e-4 \
#   batch_size=360 \
#   trainer.devices=4 \
#   resume_run_dir=outputs/exp_1129_fm_direct_free_full/2025-11-29_21-31-51 \
#   datamodule.num_workers=16 datamodule.prefetch_factor=2 datamodule.persistent_workers=false

# python train.py \
#     resume_run_dir=outputs/exp_1129_fm_rigid_full/2025-11-29_21-34-47 \
#     trainer.devices=4 \
#     datamodule.num_workers=16 datamodule.prefetch_factor=2 datamodule.persistent_workers=false \
#     resume_in_place=false
