#!/bin/bash

export CUDA_VISIBLE_DEVICES=6


python train.py \
    model=flow_matching_hand_dit \
    model.prediction_target=x model.tau_min=1e-5 \
    model.train_time_schedule.name=logit_normal \
    +model.train_time_schedule.params.mu=-0.8 \
    +model.train_time_schedule.params.sigma=1.6 \
    model.train_time_schedule.params.t_min=1e-4 \
    model.train_time_schedule.params.t_max=0.9999 \
    model.sample_schedule=linear \
    backbone=ptv3_sparse_fourier \
    datamodule=handencoder_dm_dex \
    datamodule.use_scene_normals=true \
    datamodule.train_sample_limit=25200 \
    datamodule.val_sample_limit=10800 \
    datamodule.test_sample_limit=10800 \
    optimizer.lr=1e-4 \
    compile=true \
    experiment_name=debug_1204_fm_direct_free_pred_x \
    dit.qk_norm=true \
    dit.qk_norm_type='rms' \
    dit.norm_type='rms' \
    dit.activation_fn=swiglu \
    dit.hand_scene_bias.enabled=true \
    batch_size=360 \
    trainer.precision=bf16-mixed \
    velocity_strategy=direct_free \
    loss.lambda_tangent=0.2 \
    loss.lambda_regression=0.01 \
    loss.regression.weights.collision=1.0 \
    loss.train_num_sample_batches_for_reg_loss=20 \
    loss.weights.loss_tangent=0.2 \
    datamodule.num_workers=16 \
    datamodule.prefetch_factor=4 \
    datamodule.persistent_workers=false 

