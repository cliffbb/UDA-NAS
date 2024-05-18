# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

# dataset settings
dataset_type = "FLAIRDataset"  
data_root = "data/FLAIR1"     
img_norm_cfg = dict(mean=[119.0252, 125.3428, 116.8876], 
                    std=[5.4468, 4.8114, 4.7743], to_rgb=True)
crop_size = (512, 512)  

source_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(512, 512)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    # dict(type='PhotoMetricDistortion'),  # is applied later in dacs.py
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),]

target_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(512, 512)),
    dict(type='RandomCrop', crop_size=crop_size),
    dict(type='RandomFlip', prob=0.5),
    # dict(type='PhotoMetricDistortion'),  # is applied later in dacs.py
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        # MultiScaleFlipAug is disabled by not providing img_ratios
        # and setting flip=False
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=1,
    train=dict(
        type="FLAIR_UDADataset",
        source=dict(# source train set
            type=dataset_type, 
            data_root=data_root,  
            img_dir="images",  
            ann_dir="labels", 
            split = "splits/source.txt", 
            pipeline=source_train_pipeline),
        target=dict(# target train set
            type=dataset_type, 
            data_root=data_root,  
            img_dir="images",  
            ann_dir="labels", 
            split = "splits/target_train.txt", 
            pipeline=target_train_pipeline)),
    val=dict(# target val set
        type=dataset_type,
        data_root=data_root,
        img_dir="images",
        ann_dir="labels",
        split = "splits/target_val.txt",
        pipeline=test_pipeline),
    test=dict(# target test set
        type=dataset_type,
        data_root=data_root,
        img_dir="images/test",
        ann_dir="labels/test",
        split = "splits/target_test.txt",
        pipeline=test_pipeline)
)
