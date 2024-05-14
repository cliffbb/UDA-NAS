#  Configuration for training the network

_base_ = [
    '../_base_/default_runtime.py',
    # MRF-UNet network architecture
    '../_base_/models/mrf_unet.py',
    # FLAIR data loading
    '../_base_/datasets/oem_regionalwise_uda.py',
    # DACS UDA self-training
    '../_base_/uda/dacs.py',
    # AdamW optimizer
    '../_base_/schedules/adamw.py',
    # Linear learning rate warmup 
    '../_base_/schedules/poly10warm.py']

# Random seed
seed = 0

# Modify the backbone architecture config
choices = '5,3,1,0,5,1,6,0,2,6,5,4,4,0,0,0,9,0,4,7,0,4,3,0,7,5' # Net-C1 architecture choices
# choices = '5,0,3,6,5,2,5,5,4,3,5,4,4,2,0,3,6,0,3,5,0,4,9,2,6,5' # Net-C2 architecture choices

embedding_size=16
num_classes = 9

model = dict(
     backbone=dict(choices=choices),
     decode_head=dict(
          in_channels=[embedding_size],
          in_index=[0],
          channels=embedding_size,
          num_classes=num_classes,
          loss_decode=dict(type='RecallCrossEntropyLoss', num_classes=num_classes),),
        #   loss_decode=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0)),
     )

# Modify the DACS config
uda = dict(
    # Consistency regularizer
    consistency_regularizer='confidence_based', # ['confidence_based' | 'energy_based']
    # Increased alpha
    alpha=0.999,
    # Pseudo-label crop
    pseudo_weight_ignore_top=15,
    pseudo_weight_ignore_bottom=120,
    # pallete
    cmap = 'oem',)

# Data 
data = dict(workers_per_gpu=1,  samples_per_gpu=8,)
crop_size = (512, 512) 

# Optimizer hyperparameters
optimizer_config = None
optimizer = dict(_delete_=True, type="AdamW", lr=3e-3, betas=(0.9, 0.999), weight_decay=0.05)
lr_config = dict(_delete_=True,
                 policy="poly",
                 warmup="linear",
                 warmup_iters=1500,
                 warmup_ratio=1e-5,
                 power=1.0,
                 min_lr=0.0,
                 by_epoch=False,
                 )

n_gpus = 1
runner = dict(type='IterBasedRunner', max_iters=140000)

# Logging configuration
checkpoint_config = dict(by_epoch=False, interval=4000, max_keep_ckpts=1) 
evaluation = dict(interval=4000, metric='mIoU') 
