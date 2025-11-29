#!/bin/bash
# run_training.sh

# 防僵尸进程和卡住
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL 
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1  # 如果网络有问题
export CUDA_LAUNCH_BLOCKING=0  # 设为1调试，0正常运行
export NCCL_TIMEOUT=1800
export TORCH_NCCL_BLOCKING_WAIT=0
# 内存管理
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True
export CUDA_VISIBLE_DEVICES=3,4,5,6

# 运行训练
python train.py \
    model=flow_matching_hand_dit \
    backbone=ptv3_sparse \
    datamodule=handencoder_dm_dex_debug \
    experiments=multi_gpu \
    optimizer.lr=2e-4 \
    compile=false \
    experiment_name=exp_1129_fm_rigid \
    dit.qk_norm=true \
    batch_size=380 \
    velocity_strategy=group_rigid \
    trainer.precision=bf16-mixed \
    datamodule.num_workers=8 

